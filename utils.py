import sys
import scipy.stats
import numpy as np
import pandas as pd


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
    # 2 means tie
    # 0 means model A
    # 1 means model B
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


def per_user_distribution(interleaving_dataset):
    print('Computing winning model')
    # 2 means tie
    # 0 means model A
    # 1 means model B
    interleaving_dataset['winning_model'] = 2
    interleaving_dataset.loc[interleaving_dataset[
                                 'click_per_model_A'] > interleaving_dataset['click_per_userId'] / 2,
                             'winning_model'] = 0
    interleaving_dataset.loc[interleaving_dataset[
                                 'click_per_model_A'] < interleaving_dataset['click_per_userId'] / 2,
                             'winning_model'] = 1
    per_query = interleaving_dataset.groupby('queryId')['winning_model'].value_counts()
    print()


def generate_new_data(data_to_add_stats, click_per_query_max, min_percentage_click_per_user_id,
                      max_percentage_click_per_user_id, max_clicks_per_user):
    interactions_added_data_frames = []
    # Setting seed for reproducibility
    np.random.seed(0)
    while data_to_add_stats['new_interactions_to_add'].values.sum() > 0:
        interactions_added_single_pass = pd.DataFrame()
        interactions_added_single_pass['queryId'] = data_to_add_stats['queryId']
        # All the queries have the same total number of clicks.
        interactions_added_single_pass['click_per_query'] = click_per_query_max
        # Generate random click_per_userId
        interactions_added_single_pass['click_per_userId'] = np.random.randint(1, max_clicks_per_user + 1,
                                                                               size=data_to_add_stats.shape[0])
        # Computing remaining clicks to add
        interactions_added_single_pass['new_interactions_to_add'] = data_to_add_stats[
                                                           'new_interactions_to_add'] - interactions_added_single_pass[
                                                           'click_per_userId']
        # If new_interactions_to_add < 0 we are adding to much clicks.
        # Resize click_per_userId in order to have as much clicks as needed.
        interactions_added_single_pass['click_per_userId'] = np.where(
            interactions_added_single_pass['new_interactions_to_add'] < 0,
            data_to_add_stats['new_interactions_to_add'], interactions_added_single_pass['click_per_userId'])
        interactions_added_single_pass = interactions_added_single_pass.astype({'click_per_userId': 'int64'})

        interactions_added_data_frames.append(interactions_added_single_pass)

        # Computing remaining clicks to add
        data_to_add_stats = data_to_add_stats.copy()
        data_to_add_stats.update(interactions_added_single_pass['new_interactions_to_add'])
        # Keep only queries with remaining interactions (clicks) to add.
        data_to_add_stats = data_to_add_stats.loc[data_to_add_stats['new_interactions_to_add'] > 0]

    new_data = pd.concat(interactions_added_data_frames, ignore_index=True, sort=True)
    new_data.drop(columns={'new_interactions_to_add'}, inplace=True)

    print('Populating userId')
    new_data['userId'] = new_data.groupby('queryId').cumcount() + 1

    print('Populating click_per_model_A')
    new_data['click_per_model_A'] = np.random.randint(new_data['click_per_userId'] * min_percentage_click_per_user_id,
                                                      new_data['click_per_userId'] * max_percentage_click_per_user_id +
                                                      1)

    new_data = new_data[['userId', 'click_per_userId', 'queryId', 'click_per_query', 'click_per_model_A']]

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
    print('Computing statistical significance')
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
