import pandas as pd
import numpy as np
import scipy.stats as scistat
import sys
import os
from sklearn.datasets import load_svmlight_file


def print_memory_status(dataset):
    print(dataset.info(memory_usage='deep', verbose=False))


def load_dataframe(dataset_path):
    """
    Load the <query,document> judgements as a pandas dataframe
    :param dataset_path: the path of the dataset file
    :return: the related pandas dataframe
    """
    features, relevance, query_id = load_svmlight_file(dataset_path, query_id=True, dtype="float32")
    dataframe = pd.DataFrame(features.todense())
    new_columns_name = {key: value for key, value in zip(range(0, 136), range(1, 137))}
    dataframe.rename(columns=new_columns_name, inplace=True)
    dataframe['relevance'] = relevance
    dataframe['query_id'] = query_id
    dataframe['relevance'] = dataframe['relevance'].astype('int32')
    dataframe['query_id'] = dataframe['query_id'].astype('int32')
    print_memory_status(dataframe)
    print()

    return dataframe


def parse_solr_json_facet_dataset(solr_json_facet_path):
    """
    Load the dataset in the Solr JSON facets format as a pandas dataframe.
    This is useful to get an approximate distribution of queries in a real-world
    long tail scenario.
    :param solr_json_facet_path: the path of the JSON file
    :return: the related pandas dataframe
    """
    realistic_industry_long_tail_dataframe = pd.read_json(solr_json_facet_path)
    realistic_industry_long_tail_dataframe = pd.json_normalize(realistic_industry_long_tail_dataframe['parent_buckets'],
                                                               record_path=['users'], meta=['val', 'count'],
                                                               meta_prefix='query_')
    realistic_industry_long_tail_dataframe.rename(
        columns={'val': 'userId', 'count': 'click_per_userId', 'query_val': 'query_id',
                 'query_count': 'click_per_query'}, inplace=True)
    return realistic_industry_long_tail_dataframe


def get_long_tail_query_set(query_document_pairs, real_word_long_tail_query_distribution, scaling_factor=1):
    """
    Returns an array of repeated queries id.
    Each query is repeated according to a distribution extracted from a real-world
    long tail dataset.
    :param query_document_pairs: the input experimental dataset with <query,document> pairs with judgement
    :param real_word_long_tail_query_distribution: a <query,user> real world distribution,
     each row is a query executed once by a user
    :param scaling_factor: to reduce the amount of repetitions for the queries, but keeping a similarly shaped long tail
    :return: an array of query ids that contains repetitions (a repetition means the query is executed multiple times)
    """
    query_id_to_users = (real_word_long_tail_query_distribution.groupby(['query_id']).size().nlargest(n=1000,
                                                                                                      keep='first') * scaling_factor).astype(
        int)
    query_id_to_users = query_id_to_users.reset_index()
    long_tail = query_id_to_users.rename(columns={0: "queryExecutions"})
    long_tail = long_tail.loc[long_tail['queryExecutions'] > 0]
    # the /2 here is just to reduce the amount of queries, to reduce the computational stress for the calculus
    # the shape of the long tail is not affected much in comparison to the original long tail we extracted from an industrial example
    long_tail = (long_tail['queryExecutions'].value_counts() / 2).sort_index().astype(int)
    long_tail = long_tail.loc[long_tail > 0]

    total_set_of_queries = query_document_pairs['query_id'].unique()
    set_of_queries = []
    query_id_index = 0
    for repetitions, unique_queries_to_repeat in long_tail.items():
        set_of_queries = np.append(set_of_queries, np.repeat(
            total_set_of_queries[query_id_index:query_id_index + unique_queries_to_repeat], repetitions))
        query_id_index = query_id_index + unique_queries_to_repeat
    return np.array(set_of_queries, dtype=int)


def cache_ranked_lists_per_ranker(experiment_results_dataframe, ndcg_top_k, query_document_pairs,
                                  rankers_to_evaluate_count,
                                  set_of_queries):
    """
    Calculates the ranked list of search results for each ranker and query.
    :param experiment_results_dataframe: each row is a <ranker_a, ranker_b> combination
    :param ndcg_top_k: NDCG@k is calculated for each ranker
    :param query_document_pairs: the input experimental dataset with <query,document> pairs with judgement
    :param rankers_to_evaluate_count: the number of ranker to evaluate
    :param set_of_queries: the queries to run for each ranker
    :return: a dictionary with
    <key> = <ranker_id>_<query_id>
    <value> = the ranked list of document ids with relevance judgement rating associated
    """
    ranked_list_cache = {}
    for ranker in range(1, rankers_to_evaluate_count + 1):
        ndcg_per_ranker = []
        for query_id in set_of_queries:
            matching_documents = query_document_pairs.loc[query_document_pairs['query_id'] == query_id]
            # Pandas internally uses quick-sort which means ties don't guarantee an ordering following the original index
            # in our case the index order (the occurrence of the document in the judgement list doesn't mean much)
            # in case you want to enforce index to sort ties:
            # ranked_list = matching_documents.iloc[np.lexsort((matching_documents.index, -matching_documents[ranker].values))]
            ranked_list = matching_documents.sort_values(by=[ranker], ascending=False)
            ranked_list_ids = ranked_list.index.values
            ranked_list_ratings = ranked_list['relevance'].values
            ndcgAtK = compute_ndcg(ranked_list_ratings, ndcg_top_k)
            ranked_list_cache[str(ranker) + '_' + str(query_id)] = [ranked_list_ids, ranked_list_ratings]
            ndcg_per_ranker.append(ndcgAtK)
        avg_ndcg = sum(ndcg_per_ranker) / len(ndcg_per_ranker)
        print('\nRanker[' + str(ranker) + '] AVG NDCG:' + str(avg_ndcg))
        experiment_results_dataframe.loc[
            (experiment_results_dataframe['rankerA_id'] == ranker), 'rankerA_avg_NDCG'] = avg_ndcg
        experiment_results_dataframe.loc[
            (experiment_results_dataframe['rankerB_id'] == ranker), 'rankerB_avg_NDCG'] = avg_ndcg
    return ranked_list_cache


def compute_ndcg(ratings_list, top_k):
    """
    Calculated NDCG@k for the ranked list of ratings
    :param ratings_list: an ordered list of relevance judgements ratings
    :param top_k: only top k elements are considered for the list
    :return: NDCG@k metric value
    """
    if top_k == 0:
        top_k = len(ratings_list)
    ideal_dcg = dcg_at_k(sorted(ratings_list, reverse=True), top_k)
    if not ideal_dcg:
        return 0.
    return round(dcg_at_k(ratings_list, top_k) / ideal_dcg, 3)


def dcg_at_k(rating_list, top_k):
    rating_list = np.asfarray(rating_list)[:top_k]
    if rating_list.size:
        dcg_array = np.subtract(np.power(2, rating_list), 1) / np.log2(np.arange(2, rating_list.size + 2))
        return np.sum(dcg_array)
    return 0.


def execute_team_draft_interleaving(ranked_list_a, a_ratings, ranked_list_b, b_ratings):
    """
    Team Draft Interleaving
    Interleaves two ranker lists selecting at random a team captain per turn.
    It has been introduced in [1] and [2].
    [1] T. Joachims. Optimizing search engines using clickthrough data. KDD (2002)
    [2] T.Joachims.Evaluatingretrievalperformanceusingclickthroughdata.InJ.Franke, G. Nakhaeizadeh, and I. Renz, editors,
    Text Mining, pages 79–96. Physica/Springer (2003)
    :param ranked_list_a: the ordered list of document ids from ranker A
    :param a_ratings: the ordered list of relevance judgements ratings associated to each document id from ranker A
    :param ranked_list_b: the ordered list of document ids from ranker B
    :param b_ratings: the ordered list of relevance judgements ratings associated to each document id from ranker B
    :return: an ordered interleaved list with the relevance judgements and an ordered list of interleaved rankers picks
    """
    interleaved_ratings = np.empty(len(ranked_list_a), dtype=np.dtype('u1'))
    interleaved_rankers = np.empty(len(ranked_list_a), dtype=np.dtype('u1'))
    elements_same_position = ranked_list_a - ranked_list_b
    already_added = set()
    # -1 means we need to draw a model randomly, 0 means model A turn, 1 means model B
    ranker_turn = -1
    index_a = 0
    index_b = 0
    result_index = 0

    while (result_index < len(ranked_list_a)) and index_a < len(ranked_list_a) and \
            index_b < len(ranked_list_b):
        random_ranker_choice = np.random.randint(2, size=1)
        if (ranker_turn == 0) or ((ranker_turn == -1) and (random_ranker_choice == 0)):
            index_a = update_index(already_added, index_a, ranked_list_a)
            already_added.add(ranked_list_a[index_a])
            interleaved_ratings[result_index] = a_ratings[index_a]
            interleaved_rankers[result_index] = 0
            if ranker_turn == -1:  # we drew A , next turn is B
                ranker_turn = 1
            else:
                ranker_turn = -1
            index_a += 1
            result_index += 1
        else:
            index_b = update_index(already_added, index_b, ranked_list_b)
            already_added.add(ranked_list_b[index_b])
            interleaved_ratings[result_index] = b_ratings[index_b]
            interleaved_rankers[result_index] = 1
            if ranker_turn == -1:  # we drew B , next turn is A
                ranker_turn = 0
            else:
                ranker_turn = -1
            index_b += 1
            result_index += 1
    # we have only ranker 0(A) and ranker 1(B), interleaved.
    # Value=2 means the rankings got the same element in same position this will make possible to ignore such clicks
    interleaved_rankers[np.where(elements_same_position == 0)] = 2
    return np.array([interleaved_ratings, interleaved_rankers])


def update_index(already_added, index, ranked_list):
    found_element_to_add = False
    while index < len(ranked_list) and not found_element_to_add:
        element_to_check = ranked_list[index]
        if element_to_check in already_added:
            index += 1
        else:
            found_element_to_add = 'true'
    return index


def simulate_clicks(interleaved_list, top_k, realistic_model, click_distribution_per_rating):
    """
    Simulates users clicking on the search result interleaved lists.
    For more details see paragraph 5.1 and 6.3 from the <reproducibility_target_paper>
    From various points of the <reproducibility_target_paper>
    it's evident clicks were generated only on the top 10 per query, see paragraph 5.1 and 6.3
    :param interleaved_list:
    interleaved_list[0] is the array of ordered ratings(originally associated to the document Ids) in the interleaved ranked list
    interleaved_list[1] is the array of ordered rankers picked by interleaving
    :param top_k:
    clicks are generated only for the top K items in the interleaved list
    :param realistic_model:
    True - a user stops viewing search results after his/her information need is satisfied
    False - a user checks all search results (and potentially click them)
    :return: the simulated clicks associated to the interleaved list of rankers
    """
    ratings = interleaved_list[0]
    interleaved_rankers = interleaved_list[1]
    if top_k > 0:
        ratings = interleaved_list[0][:top_k]
        interleaved_rankers = interleaved_list[1][:top_k]
    # from  <reproducibility_target_paper> paragraph 5.1
    if realistic_model:
        # this dictionary models the probability c of continuing after having clicked a <query,document> with such relevance<key>
        # relevance -> probability of continuing
        # e.g.
        # if the user clicked on a document with relevance 4/4
        # there's a low probability(0.2) of continuing exploring the results, his/her information need is likely satisfied
        continue_probabilities = {0: 1, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
        continue_probabilities_vector = np.vectorize(to_probability_vectorized, otypes=[np.float])(
            continue_probabilities, ratings)
        to_continue_column = np.vectorize(will_click, otypes=[np.dtype('u1')])(continue_probabilities_vector)
        # first we click the document and then we decide to stop or continue, so we always check the first
        # the to_continue_column at index i means if you visualize or not the document at index i
        to_continue_column = np.roll(to_continue_column, 1)
        to_continue_column[0] = 1
        stopping_points = np.where(to_continue_column == 0)
        # this dictionary models the probability p of clicking a <query,document> with relevance <key>
        # relevance -> probability of click
        click_probabilities = {0: 0.05, 1: 0.1, 2: 0.2, 3: 0.4, 4: 0.8}
        click_probabilities_vector = np.vectorize(to_probability_vectorized, otypes=[np.float])(click_probabilities,
                                                                                                ratings)
        clicks_column = np.vectorize(will_click, otypes=[np.dtype('u1')])(click_probabilities_vector)

        stopping_point = identify_stop_after_click(clicks_column, stopping_points[0])
        to_continue_mask_array = np.zeros(len(ratings), dtype=np.dtype('u1'))
        for idx in range(0, stopping_point):
            to_continue_mask_array[idx] = 1
        clicks_column = clicks_column * to_continue_mask_array
    else:
        click_probabilities = {0: 0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1}
        click_probabilities_vector = np.vectorize(to_probability_vectorized, otypes=[np.float])(click_probabilities,
                                                                                                ratings)
        clicks_column = np.vectorize(will_click, otypes=[np.dtype('u1')])(click_probabilities_vector)
    for idx in range(0, len(clicks_column)):
        if interleaved_rankers[idx] != 2:
            click_distribution_per_rating[ratings[idx]] += clicks_column[idx]

    return np.array([interleaved_rankers, clicks_column])


def identify_stop_after_click(clicks_column, stopping_points):
    for stopping_point in stopping_points:
        if clicks_column[stopping_point - 1] == 1:
            return stopping_point
    return len(clicks_column)


def to_probability_vectorized(probability_dictionary, rating):
    return probability_dictionary.get(rating)


def will_click(probability):
    if probability == 0 or probability == 1:
        return probability
    else:
        return np.random.choice([0, 1], size=1, p=[1.0 - probability, probability])


def aggregate_clicks_per_ranker(interleaved_rankers_with_clicks):
    """
    Counts how many clicks ranker A got, how many ranker B got and
    how many clicks must be ignored because they happened on a document id
    that both ranker A and ranker B puts at the same position
    :param interleaved_rankers_with_clicks:
    the simulated clicks associated to the interleaved list of rankers
    :return: the clicks count per ranker
    """
    interleaved_rankers = interleaved_rankers_with_clicks[0]
    clicks = interleaved_rankers_with_clicks[1]
    per_ranker_clicks = count_clicks_per_ranker(interleaved_rankers, clicks)
    # clicks in per_ranker_clicks[2] are ignored,
    # they represent clicks on results that were at the same position for both rankers
    # see
    # [Paragraph 5.3] from the <reproducibility_target_paper>
    # and [Paragraph 9]
    # O. Chapelle, T. Joachims, F. Radlinski, and Y. Yue.
    # Large scale validation and analysis of interleaved search evaluation.
    # ACM Transactions on Information Science, 30(1), 2012.
    total_clicks = per_ranker_clicks[0] + per_ranker_clicks[1]
    return np.array([per_ranker_clicks[0], per_ranker_clicks[1], total_clicks], dtype="uint16")


def count_clicks_per_ranker(interleaved_models, clicks):
    click_count = np.zeros(3, dtype=np.int)
    for idx in range(0, len(clicks)):
        if clicks[idx] == 1:
            click_count[interleaved_models[idx]] += 1
    return click_count


def statistical_significance_computation(queries_with_clicks, zero_hypothesis_probability):
    """
    Calculates the two tailed p-value for each query with clicks.
    This value means to measure the probability of obtaining results as extreme as the ones observed, by chance.
    :param queries_with_clicks:
    for each query it contains the total number of clicks received and the number of clicks the ranker that got the most
    :param zero_hypothesis_probability: the default probability for ranker A to be better of ranker B
    :return:  the two tailed p-value for each query with clicks.
    """
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


def statistical_significance_pruning(experiment_results_dataframe):
    """
    Given the experiment clicks simulation, this function removes the queries for the ranker pairs <ranker_a, ranker_b>
    that don't bring a statistically significant contribute
    :param experiment_results_dataframe: each row is a <ranker_a, ranker_b, query_id> with all the experimental data so far
    :return: only the rows that have a statistically significant contribute
    """
    # given a click this is the zero hypothesis probability the winner ranker was clicked
    # given a click only ranker A or ranker B is clicked so zero_hypothesis_probability = 0.5
    zero_hypothesis_probability = 0.5
    statistical_significance_computation(experiment_results_dataframe, zero_hypothesis_probability)

    # Remove queries with significance higher than 5% threshold
    only_statistical_significant_queries = experiment_results_dataframe[
        experiment_results_dataframe.statistical_significance < 0.05]
    only_statistical_significant_queries = only_statistical_significant_queries.drop(columns='statistical_significance')

    return only_statistical_significant_queries


def computing_winning_ranker_ab_score(experiment_results_dataframe, statistical_weight=False):
    """
    Calculates the ab score to identify the overall winner between two rankers.
    For each pair of rankers <ranker_a,ranker_b> it uses all the available queries.
    :param experiment_results_dataframe: each row is a <ranker_a, ranker_b, query_id> with all the experimental data so far
    :param statistical_weight:
    True - this is our contribution, explained in our paper in details
    False - classic AB score
    :return the winner ranker in each pair, according to the clicks on the interleaved lists
    """
    if statistical_weight:
        experiment_results_dataframe['statistical_weight'] = 1 - experiment_results_dataframe['statistical_significance']
        per_ranker_pair_models_wins = \
            experiment_results_dataframe.groupby(['rankerA_id', 'rankerB_id', 'interleaving_winner'])[
                'statistical_weight'].sum()
        per_ranker_pair_models_wins = pd.DataFrame(per_ranker_pair_models_wins).reset_index()
        per_ranker_pair_models_wins = per_ranker_pair_models_wins.rename(
            columns={'statistical_weight': 'per_ranker_wins'})
    else:
        per_ranker_pair_models_wins = experiment_results_dataframe.groupby(
            ['rankerA_id', 'rankerB_id', 'interleaving_winner']).size()
        per_ranker_pair_models_wins = pd.DataFrame(per_ranker_pair_models_wins).reset_index().rename(
            columns={0: 'per_ranker_wins'})

    per_pair_winner = pd.DataFrame(per_ranker_pair_models_wins.groupby(['rankerA_id', 'rankerB_id']).apply(
        lambda x: compute_ab_per_pair_of_rankers(x))).reset_index()
    if statistical_weight:
        per_pair_winner.rename(columns={0: 'stat_weight_interleaving_winner'}, inplace=True)
    else:
        per_pair_winner.rename(columns={0: 'control_interleaving_winner'}, inplace=True)
    experiment_results_dataframe = pd.merge(experiment_results_dataframe, per_pair_winner, how='left', on=['rankerA_id', 'rankerB_id'])

    return experiment_results_dataframe


def compute_ab_per_pair_of_rankers(rankers_pair_model_wins):
    winner_a = rankers_pair_model_wins[
        rankers_pair_model_wins['interleaving_winner'] == 0]['per_ranker_wins']
    if winner_a.empty:
        winner_a = 0
    else:
        winner_a = float(winner_a)
    winner_b = rankers_pair_model_wins[
        rankers_pair_model_wins['interleaving_winner'] == 1]['per_ranker_wins']
    if winner_b.empty:
        winner_b = 0
    else:
        winner_b = float(winner_b)
    ties = rankers_pair_model_wins[
        rankers_pair_model_wins['interleaving_winner'] == 2]['per_ranker_wins']
    if ties.empty:
        ties = 0
    else:
        ties = float(ties)
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
