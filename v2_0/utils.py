import pandas as pd
import numpy as np
import scipy.stats as scistat
import sys
from sklearn.datasets import load_svmlight_file


def print_memory_status(dataset):
    print(dataset.info(memory_usage='deep', verbose=False))


def load_dataframe(dataset_path):
    features, relevance, query_id = load_svmlight_file(dataset_path, query_id=True)
    dataset = pd.DataFrame(features.todense())
    new_columns_name = {key: value for key, value in zip(range(0, 136), range(1, 137))}
    dataset.rename(columns=new_columns_name, inplace=True)
    dataset['relevance'] = relevance
    dataset['queryId'] = query_id
    print_memory_status(dataset)
    print()

    return dataset


def precompute_ranked_table(dataset, max_range_pair, set_of_queries):
    # Create id column
    dataset.reset_index(drop=False, inplace=True)
    dataset.rename(columns={'index': 'query_doc_id'}, inplace=True)

    ranked_table_lists = []
    ndcg_per_query_ranker_list =[]
    for ranker in range(1, max_range_pair):
        for query_index in range(0, len(set_of_queries)):
            chosen_query_id = set_of_queries[query_index]
            query_selected_documents = dataset[dataset['queryId'] == chosen_query_id]
            ranked_list = query_selected_documents.sort_values(by=[ranker], ascending=False)
            ranked_list['ranker'] = ranker
            ranked_table_lists.append(ranked_list)

            ndcg_per_query_ranker = compute_ndcg(ranked_list)
            ndcg_per_query_ranker_list.append([ranker, chosen_query_id, ndcg_per_query_ranker])

    ranked_table = pd.concat(ranked_table_lists, ignore_index=True, sort=True)
    ndcg_ranked_table = pd.DataFrame(ndcg_per_query_ranker_list)
    ndcg_ranked_table.rename(columns={0: 'ranker', 1: 'queryId', 2: 'ndcg'}, inplace=True)

    # Reorder columns
    cols = ranked_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    ranked_table = ranked_table[cols]

    # Set multiindex
    ranked_table.set_index(['ranker', 'query_doc_id'], inplace=True, verify_integrity=True)
    ndcg_ranked_table.set_index(['ranker', 'queryId'], inplace=True, verify_integrity=True)
    return ranked_table, ndcg_ranked_table


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
    return round(dcg_at_k(ranked_list['relevance'], len(ranked_list)) / idcg, 3)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def execute_tdi_interleaving(ranked_list_a, ranked_list_b, seed):
    interleaved_list = pd.DataFrame()
    np.random.seed(seed)
    team_a = []
    team_b = []

    ranked_list_a.reset_index(inplace=True)
    ranked_list_b.reset_index(inplace=True)

    remaining_indexes_list_a = list(ranked_list_a['query_doc_id'])
    remaining_indexes_list_b = list(ranked_list_b['query_doc_id'])

    while (len(remaining_indexes_list_a) > 0) and (len(remaining_indexes_list_b) > 0):
        random_model_choice = np.random.randint(2, size=1)
        if (len(team_a) < len(team_b)) or ((len(team_a) == len(team_b)) and (random_model_choice == 1)):
            k = remaining_indexes_list_a[0]
            selected_document = pd.DataFrame(ranked_list_a[ranked_list_a['query_doc_id'] == k])
            selected_document['model'] = 'a'
            interleaved_list = interleaved_list.append(selected_document)
            team_a.append(k)
        else:
            k = remaining_indexes_list_b[0]
            selected_document = pd.DataFrame(ranked_list_b[ranked_list_b['query_doc_id'] == k])
            selected_document['model'] = 'b'
            interleaved_list = interleaved_list.append(selected_document)
            team_b.append(k)
        remaining_indexes_list_a.remove(k)
        remaining_indexes_list_b.remove(k)

    interleaved_list.set_index('query_doc_id', inplace=True, verify_integrity=True)
    interleaved_list.drop(columns='ranker', inplace=True)
    return interleaved_list


def simulate_clicks(interleaved_list, seed, realistic_model=False):
    clicks_column = pd.DataFrame()
    to_continue_column = pd.DataFrame()
    np.random.seed(seed)

    interleaved_list['new_index'] = np.arange(0, len(interleaved_list))

    if realistic_model:
        click_probabilities = {0: 0.05, 1: 0.1, 2: 0.2, 3: 0.4, 4: 0.8}
        continue_probabilities = {0: 0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}

        for key in continue_probabilities:
            partial_length = int(len(interleaved_list[interleaved_list['relevance'] == key]))
            probability_to_continue = continue_probabilities[key]
            per_row_to_continue = pd.DataFrame(np.random.choice(2, size=partial_length, p=[
                1 - probability_to_continue, probability_to_continue]))
            per_row_to_continue.rename(columns={0: 'to_continue'}, inplace=True)
            per_row_to_continue.index = interleaved_list[interleaved_list['relevance'] == key].index
            to_continue_column = to_continue_column.append(per_row_to_continue)

        to_continue_column.rename(columns={0: 'to_continue'}, inplace=True)
        interleaved_list = pd.merge(interleaved_list, to_continue_column, how='left', left_index=True, right_index=True)

        idx = interleaved_list[interleaved_list['to_continue'] == 0]['new_index']
        idx = idx.iloc[0]
        interleaved_list = interleaved_list.iloc[0:idx + 1]
    else:
        click_probabilities = {1: 0.2, 2: 0.4, 3: 0.8}

    for key in click_probabilities:
        partial_length = int(len(interleaved_list[interleaved_list['relevance'] == key]))
        probability_click = click_probabilities[key]
        clicks = pd.DataFrame(np.random.choice(2, size=partial_length, p=[1 - probability_click, probability_click]))
        clicks.index = interleaved_list[interleaved_list['relevance'] == key].index
        clicks_column = clicks_column.append(clicks)

    clicks_column.rename(columns={0: 'click'}, inplace=True)
    interleaved_list = pd.merge(interleaved_list, clicks_column, how='left', left_index=True, right_index=True)

    if not realistic_model:
        interleaved_list['click'] = np.where(interleaved_list['relevance'] == 0, 0, interleaved_list['click'])
        interleaved_list['click'] = np.where(interleaved_list['relevance'] == 4, 1, interleaved_list['click'])

    interleaved_list = interleaved_list[interleaved_list['click'] == 1]
    interleaved_list.drop(columns={'click', 'new_index'}, inplace=True)
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
    per_query_model_interactions = per_query_model_interactions[
        per_query_model_interactions.statistical_significance < 0.05]
    per_query_model_interactions = per_query_model_interactions.drop(columns='statistical_significance')

    return per_query_model_interactions


def computing_winning_model_ab_score(interleaving_dataset):
    winner_a = len(interleaving_dataset[interleaving_dataset['winning_model'] == 'a'])
    winner_b = len(interleaving_dataset[interleaving_dataset['winning_model'] == 'b'])
    ties = len(interleaving_dataset[interleaving_dataset['winning_model'] == 't'])

    # Delta score
    delta_ab = (winner_a + 1 / 2 * ties) / (winner_a + winner_b + ties) - 0.5

    # Computing winning model for ab_score
    if round(delta_ab, 3) > 0:
        return 'a'
    elif round(delta_ab, 3) < 0:
        return 'b'
    else:
        return 't'


def compute_ndcg_winning_model(list_ndcg_model_a, list_ndcg_model_b):
    avg_ndcg_model_a = sum(list_ndcg_model_a) / len(list_ndcg_model_a)
    avg_ndcg_model_b = sum(list_ndcg_model_b) / len(list_ndcg_model_b)
    if avg_ndcg_model_a > avg_ndcg_model_b:
        return 'a'
    elif avg_ndcg_model_a < avg_ndcg_model_b:
        return 'b'
    else:
        return 't'
