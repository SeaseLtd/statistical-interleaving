import numpy as np
import pandas as pd
import utils


def create_adore_dataset(dataset_path, min_percentage_click_per_user_id, max_percentage_click_per_user_id, seed,
                         is_test):
    print('Reading json file')
    raw_data = pd.read_json(dataset_path)
    raw_data = pd.json_normalize(raw_data['parent_buckets'], record_path=['users'], meta=['val', 'count'],
                                 meta_prefix='query_')
    raw_data.rename(columns={'val': 'userId', 'count': 'click_per_userId', 'query_val': 'queryId',
                             'query_count': 'click_per_query'}, inplace=True)

    print('Populating userId')
    raw_data['userId'] = raw_data.groupby('queryId').userId.cumcount() + 1

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'userId': 'int64'})

    if is_test:
        raw_data = raw_data.head(6000)

    print('Fixing click per query with real dataset values')
    sum_click_per_user_id = pd.DataFrame(raw_data.groupby('queryId')['click_per_userId'].sum()).reset_index()
    sum_click_per_user_id.rename(columns={'click_per_userId': 'click_per_query'}, inplace=True)
    raw_data = pd.merge(raw_data, sum_click_per_user_id, how='left', on='queryId')
    raw_data.drop(columns='click_per_query_x', inplace=True)
    raw_data.rename(columns={'click_per_query_y': 'click_per_query'}, inplace=True)

    print('Populating click_per_model_A')
    np.random.seed(seed)
    raw_data['click_per_model_A'] = np.random.randint(raw_data['click_per_userId'] * min_percentage_click_per_user_id,
                                                      (raw_data['click_per_userId'] * max_percentage_click_per_user_id)
                                                      + 1)

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_userId': 'int64',
                                'userId': 'int64'})

    return raw_data


def create_primary_dataset(adore_beauty_dataset, min_percentage_click_per_user_id, max_percentage_click_per_user_id,
                           max_clicks_per_user, seed):
    primary_data = adore_beauty_dataset.copy()

    print('Computing per query interactions to add')
    primary_data['click_per_query_primary'] = adore_beauty_dataset['click_per_query'].max()
    primary_data['new_interactions_to_add'] = primary_data['click_per_query_primary'] - primary_data['click_per_query']
    data_to_add = primary_data.drop_duplicates(
        subset=['queryId', 'new_interactions_to_add'], keep='last')[['queryId', 'new_interactions_to_add',
                                                                     'click_per_query']].reset_index(drop=True)
    primary_data.drop(columns=['click_per_query', 'new_interactions_to_add'], inplace=True)
    primary_data.rename(columns={'click_per_query_primary': 'click_per_query'}, inplace=True)

    print('Generating additional data')
    new_data = utils.generate_new_data(data_to_add[data_to_add['new_interactions_to_add'] > 0],
                                       adore_beauty_dataset['click_per_query'].max(), min_percentage_click_per_user_id,
                                       max_percentage_click_per_user_id, max_clicks_per_user, seed)
    primary_data = primary_data.append(new_data, ignore_index=True)

    primary_data = primary_data[['userId', 'click_per_userId', 'queryId', 'click_per_query', 'click_per_model_A']]
    primary_data = primary_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_model_A': 'int64',
                                        'click_per_userId': 'int64', 'userId': 'int16'})

    print('Populating userId')
    primary_data['userId'] = primary_data.groupby('queryId').cumcount() + 1

    return primary_data


def create_variation_dataset(primary_data, click_per_query_adore, seed):
    primary_copy_to_steal_from = primary_data.copy()
    primary_copy_to_steal_from.drop(columns=['click_per_query'], inplace=True)
    primary_copy_to_steal_from = pd.merge(primary_copy_to_steal_from, click_per_query_adore, left_on='queryId',
                                          right_index=True, how='inner')

    print('Sampling interactions from primary')
    interactions_added_data_frames = []
    while primary_copy_to_steal_from['click_per_query'].values.sum() > 0:
        grouped_by_query_id = primary_copy_to_steal_from.groupby(['queryId'])
        # Allow or disallow sampling of the same row more than once.
        interactions_added_single_pass = grouped_by_query_id.sample(random_state=seed)
        interactions_added_single_pass[
            'click_per_query_after_addition'] = interactions_added_single_pass[
                                                            'click_per_query'] - interactions_added_single_pass[
            'click_per_userId']

        # If click_per_query_after_addition < 0 we are adding to much clicks.
        # Resize click_per_model_A and click_per_userId in order to have as much clicks as needed.
        interactions_added_single_pass['click_per_model_A'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            (interactions_added_single_pass['click_per_model_A'] * interactions_added_single_pass['click_per_query']) /
            interactions_added_single_pass['click_per_userId'], interactions_added_single_pass['click_per_model_A'])
        interactions_added_single_pass['click_per_userId'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            interactions_added_single_pass['click_per_query'], interactions_added_single_pass['click_per_userId'])
        interactions_added_single_pass = interactions_added_single_pass.round({'click_per_model_A': 0}).astype({
            'click_per_model_A': 'int64'})

        interactions_added_data_frames.append(interactions_added_single_pass)

        # Substitute the current click_per_query with the remaining click_per_query to add.
        # (as it is used to stop the while loop)
        primary_copy_to_steal_from = pd.merge(primary_copy_to_steal_from, interactions_added_single_pass[[
            'queryId', 'click_per_query_after_addition']], on='queryId', how='outer', validate="many_to_one")
        primary_copy_to_steal_from.drop(columns=['click_per_query'], inplace=True)
        primary_copy_to_steal_from.rename(columns={'click_per_query_after_addition': 'click_per_query'}, inplace=True)

        # Remove the interactions added from the to_steal_from dataframe.
        primary_copy_to_steal_from.drop(interactions_added_single_pass.index, inplace=True)
        # Remove the queries that have enough interactions (clicks).
        primary_copy_to_steal_from = primary_copy_to_steal_from.loc[primary_copy_to_steal_from['click_per_query'] > 0]
        primary_copy_to_steal_from.reset_index(drop=True, inplace=True)

    variation_data_frame = pd.concat(interactions_added_data_frames, ignore_index=True, sort=True)
    variation_data_frame.drop(columns=['click_per_query_after_addition', 'click_per_query'], inplace=True)
    variation_data_frame = pd.merge(variation_data_frame, click_per_query_adore, left_on='queryId', right_index=True,
                                    how='inner')
    variation_data_frame.reset_index(drop=True, inplace=True)

    variation_data_frame = variation_data_frame[['userId', 'click_per_userId', 'queryId', 'click_per_query',
                                                 'click_per_model_A']]

    return variation_data_frame
