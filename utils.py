import sys
import scipy.stats
import numpy as np
import pandas as pd


def create_adore_dataset():
    print('Reading json file')
    raw_data = pd.read_json('./dataset_from_adore/query_click_user.json')
    raw_data = pd.json_normalize(raw_data['parent_buckets'], record_path=['users'], meta=['val', 'count'],
                                 meta_prefix='query_')
    raw_data.rename(columns={'val': 'userId', 'count': 'click_per_userId', 'query_val': 'queryId',
                             'query_count': 'click_per_query'}, inplace=True)
    raw_data['userId'] = raw_data.groupby('queryId').userId.cumcount() + 1

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'userId': 'int16'})

    # SOLO PER TEST
    # raw_data = raw_data.head(3000)

    print('Fixing click per query')
    sum_clicks = pd.DataFrame(raw_data.groupby('queryId')['click_per_userId'].sum()).reset_index()
    sum_clicks.rename(columns={'click_per_userId': 'click_per_query'}, inplace=True)
    raw_data = pd.merge(raw_data, sum_clicks, how='left', on='queryId')
    raw_data.drop(columns='click_per_query_x', inplace=True)
    raw_data.rename(columns={'click_per_query_y': 'click_per_query'}, inplace=True)

    print('Populating click_per_model_A')
    clicks_list = list()
    for index in raw_data.index:
        min_data = 0
        max_data = raw_data['click_per_userId'][index]
        clicks = np.random.randint(min_data, max_data + 1)
        clicks_list.append(int(clicks))
    click_per_model_a = pd.Series(clicks_list, index=raw_data.index)
    raw_data['click_per_model_A'] = click_per_model_a

    raw_data = raw_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_userId': 'int64',
                                'userId': 'int16'})

    return raw_data


def elaborate_dataset_for_score(interleaving_dataset):
    print('Computing total_click_per_model_A')
    total_click_per_model_a = pd.DataFrame(interleaving_dataset.groupby(['queryId'])['click_per_model_A'].sum())
    total_click_per_model_a.reset_index(inplace=True)
    total_click_per_model_a.rename(columns={'click_per_model_A': 'total_click_per_model_A'}, inplace=True)
    interleaving_dataset = pd.merge(interleaving_dataset, total_click_per_model_a, on='queryId', how='left')
    interleaving_dataset.drop_duplicates(subset=['queryId', 'click_per_query', 'total_click_per_model_A'],
                                         keep='last', inplace=True)
    interleaving_dataset.drop(columns=['userId', 'click_per_userId', 'click_per_model_A'], inplace=True)

    print('Computing winning model')
    interleaving_dataset['winning_model'] = 2
    interleaving_dataset.loc[interleaving_dataset[
                                 'total_click_per_model_A'] > interleaving_dataset['click_per_query'] / 2,
                             'winning_model'] = 0
    interleaving_dataset.loc[interleaving_dataset[
                                 'total_click_per_model_A'] < interleaving_dataset['click_per_query'] / 2,
                             'winning_model'] = 1

    interleaving_dataset.loc[
        interleaving_dataset['winning_model'] == 1,
        'total_click_per_model_A'] = interleaving_dataset['click_per_query'] - interleaving_dataset[
        'total_click_per_model_A']
    interleaving_dataset.rename(columns={'total_click_per_model_A': 'click_per_winning_model'}, inplace=True)

    return interleaving_dataset


def generate_new_data(data_to_add_stats, click_per_query_max):
    print('Generating random click_per_userId for primary dataset')
    clicks_list = list()
    new_data = pd.DataFrame(columns=['queryId', 'click_per_userId'])
    for index in data_to_add_stats.index:
        if data_to_add_stats.loc[index, 'new_interactions_to_add'] > 0:
            # max_clicks = data_to_add_stats.loc[index, 'new_interactions_to_add'] + 1
            while sum(clicks_list) < data_to_add_stats.loc[index, 'new_interactions_to_add']:
                clicks = np.random.randint(1, 5 + 1)
                clicks_list.append(clicks)
                # max_clicks = max_clicks - clicks + 1
            del clicks_list[-1]
            clicks_list.append(data_to_add_stats.loc[index, 'new_interactions_to_add'] - sum(clicks_list))
            data_to_append = {'queryId': [data_to_add_stats.loc[index, 'queryId']] * len(clicks_list),
                              'click_per_userId': clicks_list,
                              'click_per_query': [click_per_query_max] * len(clicks_list)}
            new_data = new_data.append(pd.DataFrame(data_to_append))
            clicks_list.clear()
    new_data['userId'] = new_data.groupby('queryId').cumcount() + 1

    new_data.reset_index(drop=True, inplace=True)
    print('Populating click_per_model_A')
    clicks_per_user_list = list()
    for index in new_data.index:
        min_data = 0
        max_data = new_data['click_per_userId'][index]
        clicks = np.random.randint(min_data, max_data + 1)
        clicks_per_user_list.append(int(clicks))
    click_per_model_a = pd.Series(clicks_per_user_list, index=new_data.index)
    new_data['click_per_model_A'] = click_per_model_a

    return new_data


def statistical_significance_computation(interactions, overall_diff):
    p = overall_diff
    interactions['cumulative_distribution_left'] = scipy.stats.binom.cdf(interactions.click_per_winning_model,
                                                                         interactions.click_per_query, p)
    interactions['pmf'] = scipy.stats.binom.pmf(interactions.click_per_winning_model, interactions.click_per_query, p)
    interactions['cumulative_distribution_right'] = 1 - interactions[
        'cumulative_distribution_left'] + interactions['pmf']
    interactions['statistical_significance'] = 2 * interactions[[
        'cumulative_distribution_left', 'cumulative_distribution_right']].min(axis=1) + sys.float_info.epsilon
    interactions.drop(columns=['cumulative_distribution_left', 'cumulative_distribution_right', 'pmf'], inplace=True)
    # print('AFTER STATISTICAL SIGNIFICANCE')
    # print(h.heap())
    return interactions


def pruning(interleaving_dataset, percentage_dropped_queries):
    overall_diff = 0.5
    # print('PRUNING')
    # print(h.heap())
    per_query_model_interactions = statistical_significance_computation(interleaving_dataset.copy(), overall_diff)
    # Remove interactions with significance higher than 5% threshold
    queries_before_drop = per_query_model_interactions.shape[0]
    print('Number of queries before drop: ' + str(queries_before_drop))
    per_query_model_interactions = per_query_model_interactions[
        per_query_model_interactions.statistical_significance < 0.05]
    queries_after_drop = per_query_model_interactions.shape[0]
    print('Number of queries after drop: ' + str(queries_after_drop))
    per_query_model_interactions = per_query_model_interactions.drop(columns='statistical_significance')

    print('Dropped queries: ' + str(queries_before_drop - queries_after_drop))
    percentage = (queries_before_drop - queries_after_drop) * 100 / queries_before_drop
    percentage_dropped_queries.append(percentage)
    print('Percentage dropped queries: ' + str(percentage))


    return per_query_model_interactions


def computing_ab_score(interleaving_dataset):
    winner_a = len(interleaving_dataset[interleaving_dataset['winning_model'] == 0])
    winner_b = len(interleaving_dataset[interleaving_dataset['winning_model'] == 1])
    ties = len(interleaving_dataset[interleaving_dataset['winning_model'] == 2])

    # Delta score
    delta_ab = (winner_a + 1 / 2 * ties) / (winner_a + winner_b + ties) - 0.5

    return round(delta_ab, 3)


def same_score(ab_first, ab_second):
    if (ab_first > 0 and ab_second > 0) or (
            ab_first == 0 and ab_second == 0) or (
            ab_first < 0 and ab_second < 0):
        return True
    else:
        return False
