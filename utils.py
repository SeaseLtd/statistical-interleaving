import pandas as pd
import numpy as np
import scipy.stats as scistat
import sys


def preprocess_dataset(dataset_path, output_path):
    dataset = pd.read_csv(dataset_path, sep=' ', header=None)
    dataset.drop(columns={138}, inplace=True)
    dataset.replace(to_replace=r'^.*:', value='', regex=True, inplace=True)

    dataset.rename(columns={0: 'relevance', 1: 'queryId'}, inplace=True)
    new_columns_name = {key: value for key, value in zip(range(2, 138), range(1, 137))}
    dataset.rename(columns=new_columns_name, inplace=True)

    dataset = dataset.astype({key: 'float32' for key in range(1, 137)})
    dataset = dataset.astype({'relevance': 'int8', 'queryId': 'int32'})

    store = pd.HDFStore(output_path + '/' + 'processed_train.h5')
    store['processed_train'] = dataset
    store.close()


def load_dataframe(dataset_path):
    dataset_store = pd.HDFStore(dataset_path, 'r')
    dataset = dataset_store['processed_train']
    dataset_store.close()
    return dataset


def generate_set_with_search_demand_curve(dataset):
    total_set_of_queries = dataset.queryId.unique()
    # Search demand curve. First 360 unpopular queries.
    set_of_queries = total_set_of_queries[:360]
    # 14 queries repeated 10 times
    set_of_queries = np.append(set_of_queries, np.repeat(total_set_of_queries[360:374], 10))
    # 2 queries repeated 100 times
    set_of_queries = np.append(set_of_queries, np.repeat(total_set_of_queries[374:376], 100))
    # 1 query repeated 300 times
    set_of_queries = np.append(set_of_queries, np.repeat(total_set_of_queries[376:377], 300))
    return set_of_queries


def compute_ndcg(ranked_list):
    idcg = dcg_at_k(sorted(ranked_list['relevance'], reverse=True), len(ranked_list))
    if not idcg:
        return 0.
    return dcg_at_k(ranked_list['relevance'], len(ranked_list)) / idcg


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def execute_tdi_interleaving(ranked_list_a, ranked_list_b):
    interleaved_list = pd.DataFrame()
    team_a = []
    team_b = []

    ranked_list_a.reset_index(inplace=True)
    ranked_list_b.reset_index(inplace=True)

    remaining_indexes_list_a = list(ranked_list_a['index'])
    remaining_indexes_list_b = list(ranked_list_b['index'])

    while (len(remaining_indexes_list_a) > 0) and (len(remaining_indexes_list_b) > 0):
        random_model_choice = np.random.randint(2, size=1)
        if (len(team_a) < len(team_b)) or ((len(team_a) == len(team_b)) and (random_model_choice == 1)):
            k = remaining_indexes_list_a[0]
            selected_document = pd.DataFrame(ranked_list_a[ranked_list_a['index'] == k])
            selected_document = selected_document.astype({key: 'float32' for key in range(1, 137)})
            selected_document = selected_document.astype({'relevance': 'int8', 'queryId': 'int32'})
            selected_document['model'] = 'a'
            interleaved_list = interleaved_list.append(selected_document)
            team_a.append(k)
        else:
            k = remaining_indexes_list_b[0]
            selected_document = pd.DataFrame(ranked_list_b[ranked_list_b['index'] == k])
            selected_document = selected_document.astype({key: 'float32' for key in range(1, 137)})
            selected_document = selected_document.astype({'relevance': 'int8', 'queryId': 'int32'})
            selected_document['model'] = 'b'
            interleaved_list = interleaved_list.append(selected_document)
            team_b.append(k)
        remaining_indexes_list_a.remove(k)
        remaining_indexes_list_b.remove(k)

    interleaved_list.set_index('index', inplace=True)
    return interleaved_list


def simulate_clicks(interleaved_list, seed):
    clicks_column = pd.DataFrame()
    np.random.seed(seed)

    perfect_model_click_probabilities = {1: 0.2, 2: 0.4, 3: 0.8}
    for key in perfect_model_click_probabilities:
        partial_length = int(len(interleaved_list[interleaved_list['relevance'] == key]))
        probability_click = perfect_model_click_probabilities[key]
        clicks = pd.DataFrame(np.random.choice(2, size=partial_length, p=[1 - probability_click, probability_click]))
        clicks.index = interleaved_list[interleaved_list['relevance'] == key].index
        clicks_column = clicks_column.append(clicks)

    interleaved_list = pd.merge(interleaved_list, clicks_column, how='left', left_index=True, right_index=True)
    interleaved_list.rename(columns={0: 'click'}, inplace=True)
    interleaved_list['click'] = np.where(interleaved_list['relevance'] == 0, 0, interleaved_list['click'])
    interleaved_list['click'] = np.where(interleaved_list['relevance'] == 4, 1, interleaved_list['click'])

    interleaved_list = interleaved_list[interleaved_list['click'] == 1]
    return interleaved_list


def compute_winning_model(interleaved_list, chosen_query_id):
    click_per_a = interleaved_list[interleaved_list['model'] == 'a'].shape[0]
    click_per_b = interleaved_list[interleaved_list['model'] == 'b'].shape[0]
    total_clicks = click_per_a + click_per_b
    if click_per_a > click_per_b:
        return [chosen_query_id, click_per_a, total_clicks, 'a']
    elif click_per_b > click_per_a:
        return [chosen_query_id, click_per_b, total_clicks, 'b']
    else:
        return [chosen_query_id, click_per_a, total_clicks, 't']


def statistical_significance_computation(interactions, overall_diff):
    p = overall_diff
    interactions['cumulative_distribution_left'] = scistat.binom.cdf(interactions.click_per_winning_model,
                                                                     interactions.click_per_query, p)
    interactions['pmf'] = scistat.binom.pmf(interactions.click_per_winning_model, interactions.click_per_query, p)
    interactions['cumulative_distribution_right'] = 1 - interactions[
        'cumulative_distribution_left'] + interactions['pmf']
    interactions['statistical_significance'] = 2 * interactions[[
        'cumulative_distribution_left', 'cumulative_distribution_right']].min(axis=1) + sys.float_info.epsilon
    interactions.drop(columns=['cumulative_distribution_left', 'cumulative_distribution_right', 'pmf'], inplace=True)
    return interactions


def pruning(interleaving_dataset):
    overall_diff = 0.5
    per_query_model_interactions = statistical_significance_computation(interleaving_dataset.copy(), overall_diff)

    # Remove interactions with significance higher than 5% threshold
    queries_before_drop = per_query_model_interactions.shape[0]
    print('------ Number of queries before drop: ' + str(queries_before_drop))
    per_query_model_interactions = per_query_model_interactions[
        per_query_model_interactions.statistical_significance < 0.05]
    queries_after_drop = per_query_model_interactions.shape[0]
    print('------ Number of queries after drop: ' + str(queries_after_drop))
    per_query_model_interactions = per_query_model_interactions.drop(columns='statistical_significance')

    return per_query_model_interactions


def computing_ab_score(interleaving_dataset):
    winner_a = len(interleaving_dataset[interleaving_dataset['winning_model'] == 'a'])
    winner_b = len(interleaving_dataset[interleaving_dataset['winning_model'] == 'b'])
    ties = len(interleaving_dataset[interleaving_dataset['winning_model'] == 't'])

    # Delta score
    delta_ab = (winner_a + 1 / 2 * ties) / (winner_a + winner_b + ties) - 0.5

    return round(delta_ab, 3)
