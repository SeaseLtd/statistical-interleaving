import numpy as np
import pandas as pd
import utils


def create_adore_dataset(model_a_preference, is_test):
    print('Reading json file')
    raw_data = pd.read_json('./dataset_from_adore/query_click_user.json')
    raw_data = pd.json_normalize(raw_data['parent_buckets'], record_path=['users'], meta=['val', 'count'],
                                 meta_prefix='query_')
    raw_data.rename(columns={'val': 'userId', 'count': 'click_per_userId', 'query_val': 'queryId',
                             'query_count': 'click_per_query'}, inplace=True)
    raw_data['userId'] = raw_data.groupby('queryId').userId.cumcount() + 1

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'userId': 'int16'})

    if is_test:
        raw_data = raw_data.head(3000)

    print('Fixing click per query')
    sum_clicks = pd.DataFrame(raw_data.groupby('queryId')['click_per_userId'].sum()).reset_index()
    sum_clicks.rename(columns={'click_per_userId': 'click_per_query'}, inplace=True)
    raw_data = pd.merge(raw_data, sum_clicks, how='left', on='queryId')
    raw_data.drop(columns='click_per_query_x', inplace=True)
    raw_data.rename(columns={'click_per_query_y': 'click_per_query'}, inplace=True)

    print('Populating click_per_model_A')
    raw_data['click_per_model_A'] = np.random.randint(raw_data['click_per_userId'] * model_a_preference,
                                                      raw_data['click_per_userId'] + 1)

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_userId': 'int64',
                                'userId': 'int16'})

    return raw_data


def create_primary_dataset(adore_beauty_dataset, model_a_preference, max_clicks_per_user):
    primary_data = adore_beauty_dataset.copy()
    primary_data['click_per_query_primary'] = adore_beauty_dataset['click_per_query'].max()
    primary_data['new_interactions_to_add'] = primary_data['click_per_query_primary'] - primary_data['click_per_query']
    data_to_add = primary_data.drop_duplicates(
        subset=['queryId', 'new_interactions_to_add'], keep='last')[['queryId', 'new_interactions_to_add',
                                                                     'click_per_query']].reset_index(drop=True)
    primary_data.drop(columns=['click_per_query', 'new_interactions_to_add'], inplace=True)
    primary_data.rename(columns={'click_per_query_primary': 'click_per_query'}, inplace=True)

    new_data = utils.generate_new_data(data_to_add, adore_beauty_dataset['click_per_query'].max(),
                                       model_a_preference, max_clicks_per_user)
    primary_data = primary_data.append(new_data, ignore_index=True)

    primary_data = primary_data[['userId', 'click_per_userId', 'queryId', 'click_per_query', 'click_per_model_A']]
    primary_data = primary_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_model_A': 'int64',
                                        'click_per_userId': 'int64', 'userId': 'int16'})
    primary_data['userId'] = primary_data.groupby('queryId').cumcount() + 1

    return primary_data


def create_variation_dataset(primary_data, click_per_query_adore, seed):
    temp_variation_data_frame = primary_data.copy()
    temp_variation_data_frame.drop(columns=['click_per_query'], inplace=True)
    temp_variation_data_frame = pd.merge(temp_variation_data_frame, click_per_query_adore, left_on='queryId',
                                         right_index=True, how='inner')

    interactions_added_data_frames = []
    while temp_variation_data_frame['click_per_query'].values.sum() > 0:
        grouped_by_query_id = temp_variation_data_frame.groupby(['queryId'])
        # Allow or disallow sampling of the same row more than once.
        interactions_added_single_pass = grouped_by_query_id.sample(random_state=seed)
        interactions_added_single_pass[
            'click_per_query_after_addition'] = interactions_added_single_pass[
                                                            'click_per_query'] - interactions_added_single_pass[
            'click_per_userId']

        interactions_added_single_pass['click_per_model_A'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            (interactions_added_single_pass['click_per_model_A'] * interactions_added_single_pass['click_per_query']) /
            interactions_added_single_pass['click_per_userId'], interactions_added_single_pass['click_per_model_A'])
        interactions_added_single_pass['click_per_userId'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            interactions_added_single_pass['click_per_query'], interactions_added_single_pass['click_per_userId'])
        interactions_added_single_pass = interactions_added_single_pass.astype({'click_per_model_A': 'int64'})

        interactions_added_data_frames.append(interactions_added_single_pass)

        # update and reset initial pass data structures
        temp_variation_data_frame = pd.merge(temp_variation_data_frame, interactions_added_single_pass,
                                             left_on='queryId', right_on='queryId', how='left', suffixes=('', '_y'))
        temp_variation_data_frame.drop(columns=['click_per_query'], inplace=True)
        temp_variation_data_frame = temp_variation_data_frame[temp_variation_data_frame.columns[
            ~temp_variation_data_frame.columns.str.endswith('_y')]]
        temp_variation_data_frame.rename(columns={'click_per_query_after_addition': 'click_per_query'}, inplace=True)
        temp_variation_data_frame.drop(interactions_added_single_pass.index, inplace=True)
        temp_variation_data_frame = temp_variation_data_frame.loc[temp_variation_data_frame['click_per_query'] > 0]
        temp_variation_data_frame.reset_index(drop=True, inplace=True)

    variation_data_frame = pd.concat(interactions_added_data_frames, ignore_index=True, sort=True)
    variation_data_frame.drop(columns=['click_per_query_after_addition'], inplace=True)
    variation_data_frame.reset_index(drop=True, inplace=True)
    variation_data_frame.drop(columns=['click_per_query'], inplace=True)
    variation_data_frame = pd.merge(variation_data_frame, click_per_query_adore, left_on='queryId', right_index=True,
                                    how='inner')
    variation_data_frame.reset_index(drop=True, inplace=True)

    variation_data_frame = variation_data_frame[['userId', 'click_per_userId', 'queryId', 'click_per_query',
                                                 'click_per_model_A']]

    return variation_data_frame
