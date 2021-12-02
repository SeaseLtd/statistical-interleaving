import pandas as pd
import numpy as np
import scipy.stats as scistat
import sys
import os
from sklearn.datasets import load_svmlight_file
from pympler.asizeof import asizeof


def print_memory_status(dataset):
    print(dataset.info(memory_usage='deep', verbose=False))


def load_dataframe(dataset_path):
    features, relevance, query_id = load_svmlight_file(dataset_path, query_id=True, dtype="float32")
    dataset = pd.DataFrame(features.todense())
    new_columns_name = {key: value for key, value in zip(range(0, 136), range(1, 137))}
    dataset.rename(columns=new_columns_name, inplace=True)
    dataset['relevance'] = relevance
    dataset['query_id'] = query_id
    dataset['relevance'] = dataset['relevance'].astype('int32')
    dataset['query_id'] = dataset['query_id'].astype('int32')
    print_memory_status(dataset)
    print()

    return dataset


def precompute_ranked_table(dataset, max_range_pair, set_of_queries):
    # Create id column
    dataset.reset_index(drop=False, inplace=True)
    dataset.rename(columns={'index': 'query_doc_id'}, inplace=True)

    ranked_table_lists = []
    ndcg_per_query_ranker_list = []
    for ranker in range(1, max_range_pair):
        for query_index in range(0, len(set_of_queries)):
            chosen_query_id = set_of_queries[query_index]
            query_selected_documents = dataset[dataset['query_id'] == chosen_query_id]
            ranked_list = query_selected_documents.sort_values(by=[ranker], ascending=False)
            ranked_list['ranker'] = ranker
            ranked_table_lists.append(ranked_list)

            ndcg_per_query_ranker = compute_ndcg(ranked_list)
            ndcg_per_query_ranker_list.append([ranker, chosen_query_id, ndcg_per_query_ranker])

    ranked_table = pd.concat(ranked_table_lists, ignore_index=True, sort=True)

    ndcg_ranked_table = pd.DataFrame(ndcg_per_query_ranker_list)
    ndcg_ranked_table.rename(columns={0: 'ranker', 1: 'query_id', 2: 'ndcg'}, inplace=True)

    # Reorder columns
    cols = ranked_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    ranked_table = ranked_table[cols]

    # Set multiindex
    ranked_table.set_index(['ranker', 'query_doc_id'], inplace=True, verify_integrity=True)
    ndcg_ranked_table.set_index(['ranker', 'query_id'], inplace=True, verify_integrity=True)

    print('Precomputed dataframes:')
    print_memory_status(ranked_table)
    print_memory_status(ndcg_ranked_table)
    print()
    return ranked_table, ndcg_ranked_table

def from_solr_aggregations_dataset(solr_aggregation_json_path):
    realistic_industry_long_tail_dataframe = pd.read_json(solr_aggregation_json_path)
    realistic_industry_long_tail_dataframe = pd.json_normalize(realistic_industry_long_tail_dataframe['parent_buckets'], record_path=['users'], meta=['val', 'count'],
                                 meta_prefix='query_')
    realistic_industry_long_tail_dataframe.rename(columns={'val': 'userId', 'count': 'click_per_userId', 'query_val': 'query_id',
                             'query_count': 'click_per_query'}, inplace=True)
    return realistic_industry_long_tail_dataframe


def generate_set_with_search_demand_curve(dataset, realistic_industry_long_tail_dataframe, users_scaling_factor=1):
    query_id_to_users = (realistic_industry_long_tail_dataframe.groupby(['query_id']).size().nlargest(n=1000, keep='first') * users_scaling_factor).astype(int)
    query_id_to_users = query_id_to_users.reset_index()
    long_tail = query_id_to_users.rename(columns={0: "queryExecutions"})
    long_tail = long_tail.loc[long_tail['queryExecutions'] > 0]
    # the /2 here is just to reduce the amount of queries, to reduce the computational stress for the calculus
    # the shape of the long tail is not affected much in comparison to the original long tail we extracted from an industrial example
    long_tail = (long_tail['queryExecutions'].value_counts() / 2).sort_index().astype(int)
    long_tail = long_tail.loc[long_tail > 0]

    total_set_of_queries = dataset['query_id'].unique()
    set_of_queries = []
    query_id_index = 0
    for repetitions, unique_queries_to_repeat in long_tail.items():
        set_of_queries = np.append(set_of_queries, np.repeat(
            total_set_of_queries[query_id_index:query_id_index + unique_queries_to_repeat], repetitions))
        query_id_index = query_id_index + unique_queries_to_repeat
    return np.array(set_of_queries, dtype=int)


def compute_ndcg(ratings_list):
    idcg = dcg_at_k(sorted(ratings_list, reverse=True), 10)
    if not idcg:
        return 0.
    return round(dcg_at_k(ratings_list, 10) / idcg, 3)


def dcg_at_k(rating_list, topK):
    rating_list = np.asfarray(rating_list)[:topK]
    if rating_list.size:
        dcg_array = np.subtract(np.power(2, rating_list), 1) / np.log2(np.arange(2, rating_list.size + 2))
        return np.sum(dcg_array)
    return 0.


def update_index(already_added, index, ranked_list):
    found_element_to_add = False
    while index < len(ranked_list) and not found_element_to_add:
        element_to_check = ranked_list[index]
        if element_to_check in already_added:
            index += 1
        else:
            found_element_to_add = 'true'
    return index


def execute_team_draft_interleaving(ranked_list_a, a_ratings, ranked_list_b, b_ratings):
    interleaved_ratings = np.empty(len(ranked_list_a), dtype=np.dtype('u1'))
    interleaved_models = np.empty(len(ranked_list_a), dtype=np.dtype('u1'))
    elements_same_position = ranked_list_a - ranked_list_b
    already_added = set()
    turn = 0
    index_a = 0
    index_b = 0
    result_index = 0

    while (result_index < len(ranked_list_a)) and index_a < len(ranked_list_a) and \
            index_b < len(ranked_list_b):
        random_model_choice = np.random.randint(2, size=1)
        if (turn == -1) or ((turn == 0) and (random_model_choice == 0)):
            index_a = update_index(already_added, index_a, ranked_list_a)
            already_added.add(ranked_list_a[index_a])
            interleaved_ratings[result_index] = a_ratings[index_a]
            interleaved_models[result_index] = 0
            result_index += 1
            if turn == 0:
                turn = 1
            else:
                turn = 0
            index_a += 1
        else:
            index_b = update_index(already_added, index_b, ranked_list_b)
            already_added.add(ranked_list_b[index_b])
            interleaved_ratings[result_index] = b_ratings[index_b]
            interleaved_models[result_index] = 1
            result_index += 1
            if turn == 0:
                turn = -1
            else:
                turn = 0
            index_b += 1
    # we have only model 0 and model 1, interleaved. Value=2 means the rankings got the same element in same position
    # this will make possible to ignore such clicks
    interleaved_models[np.where(elements_same_position == 0)] = 2
    return np.array([interleaved_ratings, interleaved_models])


def simulate_clicks(interleaved_list, realistic_model=False):
    # from various points of the original paper it's evident clicks were generated only on the top 10 per query
    # see par 5.1 and 6.3
    top_k = 10
    ratings = interleaved_list[0][:top_k]
    interleaved_models = interleaved_list[1][:top_k]
    clicks_column = np.empty(len(ratings), dtype=np.dtype('u1'))
    to_continue_column = np.empty(len(ratings), dtype=np.dtype('u1'))

    if realistic_model:
        # this dictionary models the probability p of clicking a <query,document> with relevance <key>
        # relevance -> probability of click
        click_probabilities = {0: 0.05, 1: 0.1, 2: 0.2, 3: 0.4, 4: 0.8}
        # this dictionary models the probability s of stopping after having clicked a <query,document> with relevance
        # <key> relevance -> probability of click
        continue_probabilities = {0: 1, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
        continue_probabilities_vector = np.vectorize(to_probability_vector)(continue_probabilities, ratings)
        to_continue_column = np.vectorize(np.random.choice)(2, [1 - continue_probabilities_vector,
                                                                continue_probabilities_vector])
        # first we click the document and then we decide to stop or continue, so we always check the first
        to_continue_column = np.roll(to_continue_column, 1)
        to_continue_column[0] = 1
        click_probabilities_vector = np.vectorize(to_probability_click)(click_probabilities, ratings)
        clicks_column = np.vectorize(will_click)(click_probabilities_vector)
        clicks_column = clicks_column * to_continue_column
    else:
        click_probabilities = {1: 0.2, 2: 0.4, 3: 0.8, 4: 1}
        click_probabilities_vector = np.vectorize(to_probability_click)(click_probabilities, ratings)
        clicks_column = np.vectorize(will_click, otypes=[np.dtype('u1')])(click_probabilities_vector)
    return np.array([interleaved_models, clicks_column])


def to_probability_vector(probabilities, rating):
    return probabilities.get(rating)


def to_probability_click(click_probabilities, rating):
    if rating > 0:
        return click_probabilities.get(rating)
    else:
        return rating


def will_click(click_probability):
    if click_probability == 0 or click_probability == 1:
        return click_probability
    else:
        return np.random.choice([0, 1], size=1, p=[1.0 - click_probability, click_probability])


def compute_winning_model(interleaved_list):
    interleaved_models = interleaved_list[0]
    clicks = interleaved_list[1]
    click_count = count_click(interleaved_models, clicks)
    # clicks in click_count[2] are ignored,
    # they represent clicks on results that were at the same position for both rankers
    # see [Paragraph 9] O. Chapelle, T. Joachims, F. Radlinski, and Y. Yue. Large scale validation and analysis of interleaved search evaluation. ACM Transactions on Information Science, 30(1), 2012.
    total_clicks = click_count[0] + click_count[1]
    return np.array([click_count[0], click_count[1], total_clicks], dtype="uint16")


def count_click(interleaved_models, clicks):
    click_count = np.zeros(3, dtype=np.int)
    for idx in range(0, len(clicks)):
        if clicks[idx] == 1:
            click_count[interleaved_models[idx]] += 1
    return click_count


def statistical_significance_computation(queries_with_clicks, zero_hypothesis_probability):
    p = zero_hypothesis_probability
    # probability that given interleaving_total_clicks tries, we get <= interleaving_winner_clicks by chance
    queries_with_clicks['cumulative_distribution_left'] = scistat.binom.cdf(
        queries_with_clicks.interleaving_winner_clicks,
        queries_with_clicks.interleaving_total_clicks, p)
    # probability that given interleaving_total_clicks tries, we get = interleaving_winner_clicks by chance
    queries_with_clicks['pmf'] = scistat.binom.pmf(queries_with_clicks.interleaving_winner_clicks,
                                                   queries_with_clicks.interleaving_total_clicks, p)
    # probability that given interleaving_total_clicks tries, we get >= interleaving_winner_clicks by chance
    queries_with_clicks['cumulative_distribution_right'] = 1 - queries_with_clicks[
        'cumulative_distribution_left'] + queries_with_clicks['pmf']
    # our statistical significance is two tailed, because we have interleaving_winner_clicks which could be Model A or Model B
    queries_with_clicks['statistical_significance'] = 2 * queries_with_clicks[[
        'cumulative_distribution_left', 'cumulative_distribution_right']].min(axis=1) + sys.float_info.epsilon
    queries_with_clicks['statistical_significance'] = np.where(queries_with_clicks['interleaving_winner'] == 2,
                                                               queries_with_clicks['pmf'],
                                                               queries_with_clicks['statistical_significance'])
    queries_with_clicks.drop(columns=['cumulative_distribution_left', 'cumulative_distribution_right', 'pmf'],
                             inplace=True)
    return queries_with_clicks


def pruning(interleaving_dataset):
    # given a click this is the zero hypothesis probability the winner ranker was clicked
    # given a click only ranker A or ranker B is clicked so zero_hypothesis_probability = 0.5
    zero_hypothesis_probability = 0.5
    statistical_significance_computation(interleaving_dataset, zero_hypothesis_probability)

    # Remove interactions with significance higher than 5% threshold
    only_statistical_significant_queries = interleaving_dataset[
        interleaving_dataset.statistical_significance < 0.05]
    only_statistical_significant_queries = only_statistical_significant_queries.drop(columns='statistical_significance')

    return only_statistical_significant_queries


def computing_winning_model_ab_score(interleaving_dataset, statistical_weight=False):
    if statistical_weight:
        interleaving_dataset['statistical_weight'] = 1 - interleaving_dataset['statistical_significance']
        interleaving_dataset_tdi_stats = \
            interleaving_dataset.groupby(['rankerA_id', 'rankerB_id', 'interleaving_winner'])[
                'statistical_weight'].sum()
        interleaving_dataset_tdi_stats = pd.DataFrame(interleaving_dataset_tdi_stats).reset_index()
        interleaving_dataset_tdi_stats = interleaving_dataset_tdi_stats.rename(
            columns={'statistical_weight': 'per_ranker_wins'})
    else:
        interleaving_dataset_tdi_stats = interleaving_dataset.groupby(
            ['rankerA_id', 'rankerB_id', 'interleaving_winner']).size()
        interleaving_dataset_tdi_stats = pd.DataFrame(interleaving_dataset_tdi_stats).reset_index().rename(
            columns={0: 'per_ranker_wins'})

    per_pair_winner = pd.DataFrame(interleaving_dataset_tdi_stats.groupby(['rankerA_id', 'rankerB_id']).apply(
        lambda x: compute_ab_per_group(x))).reset_index()
    if statistical_weight:
        per_pair_winner.rename(columns={0: 'stat_weight_interleaving_winner'}, inplace=True)
    else:
        per_pair_winner.rename(columns={0: 'control_interleaving_winner'}, inplace=True)
    interleaving_dataset = pd.merge(interleaving_dataset, per_pair_winner, how='left', on=['rankerA_id', 'rankerB_id'])

    return interleaving_dataset


def compute_ab_per_group(per_group_interleaving_dataset):
    winner_a = per_group_interleaving_dataset[
        per_group_interleaving_dataset['interleaving_winner'] == 0]['per_ranker_wins']
    if winner_a.empty:
        winner_a = 0
    else:
        winner_a = int(winner_a)
    winner_b = per_group_interleaving_dataset[
        per_group_interleaving_dataset['interleaving_winner'] == 1]['per_ranker_wins']
    if winner_b.empty:
        winner_b = 0
    else:
        winner_b = int(winner_b)
    ties = per_group_interleaving_dataset[
        per_group_interleaving_dataset['interleaving_winner'] == 2]['per_ranker_wins']
    if ties.empty:
        ties = 0
    else:
        ties = int(ties)

    # In the unlikely event that all queries for a pair of rankers have no significance at all p-value=1
    # we can't say anything
    if winner_a + winner_b + ties == 0:
        return 2
    # Delta score
    delta_ab = (winner_a + 1 / 2 * ties) / (winner_a + winner_b + ties) - 0.5

    # Computing winning model for ab_score
    if round(delta_ab, 3) > 0:
        return 0
    elif round(delta_ab, 3) < 0:
        return 1
    else:
        return 2


def clean_folder(output_dir, end_string):
    files_in_directory = os.listdir(output_dir)
    filtered_files = [file for file in files_in_directory if file.endswith(end_string)]
    for file in filtered_files:
        path_to_file = os.path.join(output_dir, file)
        os.remove(path_to_file)
