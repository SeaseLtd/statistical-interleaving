import gc

import utils
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pympler.asizeof import asizeof


def start_experiment(dataset_path, seed, queries_to_evaluate_count=1000, rankers_to_evaluate_count=136,
                     long_tail_dataset_path=None, long_tail_scaling_factor=1.0, ndcg_top_k=0, click_generation_top_k=0,
                     click_generation_realistic=False, experiment_one_long_tail=False):
    """
    This experiment has been designed to reproduce the Experiment 1 in the <reproducibility_target_paper>.
    All the details of this experiment are described in our paper.
    Let's briefly list the phases of the experiment:
    - It calculates for each ranker and query in the query set,
      the NDCG@k based on the <query,document> with relevance judgements in input
    - It runs Team Draft Interleaving for each ranker_a VS ranker_b pair, on each query from the query set
    - It generates clicks simulating both a perfect and realistic group of users
    - It calculates the AB score using the simulated clicks
    - It compares the AB scores winners with the NDCG winners, to identify the accuracy of the interleaving approach
    :param dataset_path: <query,document> pairs with relevance judgements ratings, in svmlight format
    :param seed: this is the key to reproducibility, it regulates all the randomic events in the experiment
    :param queries_to_evaluate_count: how many queries to evaluate per ranker_a VS ranker_b pair, taken in order from the dataset
    :param rankers_to_evaluate_count: how many ranker to evaluate, each ranker is a feature in the <query,document> dataset
    :param long_tail_dataset_path: a real-world query distribution, expected in the solr JSON facet format
    :param long_tail_scaling_factor: a float value to multiply how many repetitions per query in the long tail
    :param ndcg_top_k: to calculate the ground truth of a ranker, NDCG@k is calculated
    0 means NDCG is calculated over the entire ranked list
    :param click_generation_top_k: clicks are generated up to the top K position of the ranked list
    0 menas clicks are generated over the entire ranked list
    :param click_generation_realistic:
    True - a user stops viewing search results after his/her information need is satisfied
    False - a user checks all search results (and potentially click them)
    :param experiment_one_long_tail:
    True - queries are repeated with a distribution that happens in the real-world
    False - each query is executed once
    :return:
    """
    np.random.seed(seed)
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    # load dataframe
    print('Loading dataframe')
    start_loading = time.time()
    query_document_pairs = utils.load_dataframe(dataset_path)
    end_loading = time.time()
    time_for_loading = end_loading - start_loading
    print('Time for loading dataframe: ' + str(time_for_loading))
    # Remove duplicates
    query_document_pairs = query_document_pairs.drop_duplicates()
    print('Query-Document Pairs: ' + str(len(query_document_pairs)))
    print('Unique Queries: ' + str(len(query_document_pairs['query_id'].unique())))
    print('Avg judged documents per query: ' + str(query_document_pairs['query_id'].value_counts().mean()))
    print('Relevance label distribution: ' + str(query_document_pairs.groupby(['relevance']).size()))
    print(query_document_pairs.info(memory_usage='deep', verbose=False))
    # Retrieve a set of queries to use for the experiment
    print('Selecting queries')
    if experiment_one_long_tail:
        # Parsing a long tail distribution from an input dataset.
        industrial_dataset = utils.parse_solr_json_facet_dataset(long_tail_dataset_path)
        # Scaling the amount of query executions in the long tail, keeping the shape of the tail
        # This is necessary to reduce the quantity of queries in total,
        # as the more the query the more costly the experiment in terms of time and memory
        set_of_queries = utils.get_long_tail_query_set(query_document_pairs, industrial_dataset, long_tail_scaling_factor)
    else:
        # Getting the first <queries_to_evaluate_count> encountered in the dataset.
        # N.B. the order is important here, the order of query IDS as they appear in the query_document_pairs is preserved
        unique_queries_to_repeat = query_document_pairs['query_id'].unique()[:queries_to_evaluate_count]
        repetition_count = long_tail_scaling_factor
        set_of_queries = []
        for query_id in unique_queries_to_repeat:
            set_of_queries = np.append(set_of_queries, np.repeat(
                query_id, repetition_count))
        set_of_queries = np.array(set_of_queries, dtype=int)
    print('\nEach ranker is evaluated on queries: ' + str(len(set_of_queries)))
    # Set up the experiment dataframe, each row is a triple <rankerA,rankerB,query_id>
    # Rankers goes from 1 to 136 max(therefore our range goes from 1 to 137)
    print('\nComputing experiment results dataframe')
    start_computing_experiment_df = time.time()
    experiment_data = []
    print(rankers_to_evaluate_count)
    for ranker_a in range(1, rankers_to_evaluate_count + 1):
        for ranker_b in range(ranker_a + 1, rankers_to_evaluate_count + 1):
            for query_id in set_of_queries:
                experiment_data.append([ranker_a, ranker_b, query_id])
    experiment_results_dataframe = pd.DataFrame(experiment_data, columns=['rankerA_id', 'rankerB_id', 'query_id'])

    # Clean unuseful data structures
    del experiment_data
    gc.collect()

    experiment_results_dataframe['rankerA_avg_NDCG'] = np.nan
    experiment_results_dataframe['rankerB_avg_NDCG'] = np.nan

    experiment_results_dataframe['avg_NDCG_winning_ranker'] = np.nan

    end_computing_experiment_df = time.time()
    time_computing_experiment_df = end_computing_experiment_df - start_computing_experiment_df
    print('Time for computing experiment results dataframe: ' + str(time_computing_experiment_df))

    print('\nComputing ranked lists and NDCG')
    start_lists = time.time()
    # This cache contains:
    # key = <ranker_id>_<query_id>
    # value = ranked list ( each search result has a documentId and relevance label)
    ranked_list_cache = utils.cache_ranked_lists_per_ranker(experiment_results_dataframe, ndcg_top_k, query_document_pairs,
                                                            rankers_to_evaluate_count, set_of_queries)

    # Assign the NDCG winners
    # model A wins -> 0
    indexes_to_change = experiment_results_dataframe.loc[experiment_results_dataframe['rankerA_avg_NDCG'] > experiment_results_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_results_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 0
    # tie -> 2
    indexes_to_change = experiment_results_dataframe.loc[experiment_results_dataframe['rankerA_avg_NDCG'] == experiment_results_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_results_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 2
    # model B wins -> 1
    indexes_to_change = experiment_results_dataframe.loc[experiment_results_dataframe['rankerA_avg_NDCG'] < experiment_results_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_results_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 1
    experiment_results_dataframe.drop(columns=['rankerA_avg_NDCG', 'rankerB_avg_NDCG'], inplace=True)
    experiment_results_dataframe['avg_NDCG_winning_ranker'] = experiment_results_dataframe['avg_NDCG_winning_ranker'].astype(int)

    end_lists = time.time()
    time_for_interleaving = end_lists - start_lists
    print('Time to calculate Ranked Lists and NDCG: ' + str(time_for_interleaving))
    print(experiment_results_dataframe.info(memory_usage='deep', verbose=False))

    print('\nComputing Interleaving')
    start_interleaving = time.time()
    experiment_results_array = experiment_results_dataframe.to_numpy()
    interleaved_ranked_lists = interleave_iteration(experiment_results_array, ranked_list_cache)
    # Clean unuseful data structures
    gc.collect()
    end_interleaving = time.time()
    time_for_interleaving = end_interleaving - start_interleaving
    print('Time for interleaving: ' + str(time_for_interleaving))

    print('\nGenerating Clicks')
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_results_dataframe)))
    print(experiment_results_dataframe.info(memory_usage='deep', verbose=False))
    start_generating_clicks = time.time()
    interleaved_ranked_lists_clicks = clicks_generation_iteration(interleaved_ranked_lists, click_generation_top_k, click_generation_realistic)
    end_generating_clicks = time.time()
    time_generating_clicks = end_generating_clicks - start_generating_clicks
    # Clean unuseful data structures
    del interleaved_ranked_lists
    gc.collect()
    print('Time for generating clicks: ' + str(time_generating_clicks))
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_results_dataframe)))
    print(experiment_results_dataframe.info(memory_usage='deep', verbose=False))

    # Computing the per query winning model/ranker
    print('\nComputing per query sum of clicks')
    start_computing_per_query_winner = time.time()
    aggregated_clicks_per_model = aggregate_interleaving_clicks_per_model_iteration(interleaved_ranked_lists_clicks).T
    experiment_results_dataframe['interleaving_a_clicks'] = pd.Series(aggregated_clicks_per_model[0])
    experiment_results_dataframe['interleaving_b_clicks'] = pd.Series(aggregated_clicks_per_model[1])
    experiment_results_dataframe['interleaving_total_clicks'] = pd.Series(aggregated_clicks_per_model[2])

    end_computing_per_query_winner = time.time()
    time_computing_per_query_winner = end_computing_per_query_winner - start_computing_per_query_winner
    del interleaved_ranked_lists_clicks, aggregated_clicks_per_model
    gc.collect()
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_results_dataframe)))
    print(experiment_results_dataframe.info(memory_usage='deep', verbose=False))
    print('Time for computing per query winning model: ' + str(time_computing_per_query_winner))
    print('Ranker combinations:' + str(experiment_results_dataframe.groupby(['rankerA_id', 'rankerB_id']).ngroups))
    # Removing pairs where no click happened
    experiment_results_dataframe = experiment_results_dataframe[
        experiment_results_dataframe.interleaving_total_clicks > 0]
    print('\nAfter the removal of queries that didn\'t show any click')
    print('Ranker combinations:' + str(experiment_results_dataframe.groupby(['rankerA_id', 'rankerB_id']).ngroups))
    # Calculate Winner
    print('\nCalculate Winner')
    # in the long tail scenario, certain queries are repeated many times, so we must accumulate the clicks
    if experiment_one_long_tail or long_tail_scaling_factor > 1:
        experiment_results_dataframe = pd.DataFrame(experiment_results_dataframe.groupby(['rankerA_id','rankerB_id','query_id']).agg({'avg_NDCG_winning_ranker':'first','interleaving_a_clicks': 'sum', 'interleaving_b_clicks': 'sum', 'interleaving_total_clicks':'sum'})).reset_index()

    experiment_results_dataframe['interleaving_winner_clicks'] = np.where(
        experiment_results_dataframe['interleaving_a_clicks'] > experiment_results_dataframe['interleaving_b_clicks'],
        experiment_results_dataframe['interleaving_a_clicks'],
        experiment_results_dataframe['interleaving_b_clicks'])

    experiment_results_dataframe['interleaving_winner'] = np.where(
        experiment_results_dataframe['interleaving_a_clicks'] > experiment_results_dataframe['interleaving_b_clicks'],
        0,
        1)
    # tie
    experiment_results_dataframe['interleaving_winner'] = np.where(
        experiment_results_dataframe['interleaving_a_clicks'] == experiment_results_dataframe['interleaving_b_clicks'],
        2,
        experiment_results_dataframe['interleaving_winner'])

    experiment_results_dataframe.drop(columns=['interleaving_a_clicks', 'interleaving_b_clicks'], inplace=True)
    # Statistical Pruning
    print('\nPruning')
    start_pruning = time.time()
    only_statistical_significant_queries = utils.statistical_significance_pruning(experiment_results_dataframe)
    experiment_results_dataframe.drop(columns=['interleaving_winner_clicks', 'interleaving_total_clicks'], inplace=True)
    end_pruning = time.time()
    time_pruning = end_pruning - start_pruning
    print('Time for pruning: ' + str(time_pruning))
    print('Only stat relevant rows')
    print(only_statistical_significant_queries.info(memory_usage='deep', verbose=False))

    # Computing standard ab_score
    print('\nComputing standard AB score')
    start_ab = time.time()
    experiment_results_dataframe = utils.computing_winning_ranker_ab_score(experiment_results_dataframe)
    end_ab = time.time()
    time_ab = end_ab - start_ab
    print('Time for ab score: ' + str(time_ab))

    # Computing Statistical weighted ab_score
    print('\nComputing Statistical Weighted AB score')
    start_ab_stat = time.time()
    experiment_results_dataframe = utils.computing_winning_ranker_ab_score(experiment_results_dataframe, True)
    end_ab_stat = time.time()
    time_ab_stat = end_ab_stat - start_ab_stat
    print('Time for ab stat score: ' + str(time_ab_stat))

    experiment_results_dataframe.drop(columns=['query_id', 'interleaving_winner', 'statistical_significance', 'statistical_weight'], inplace=True)
    experiment_results_dataframe = experiment_results_dataframe.drop_duplicates()

    # Check if ndcg agree with ab_score
    control_correctly_guessed_winner_rankers = len(experiment_results_dataframe[
                                                       experiment_results_dataframe['avg_NDCG_winning_ranker'] ==
                                                       experiment_results_dataframe[
                                                           'control_interleaving_winner']])
    accuracy_control_tdi = control_correctly_guessed_winner_rankers / len(experiment_results_dataframe)
    print('\nThe CONTROL approach got: ' + str(control_correctly_guessed_winner_rankers) + '/' + str(
        len(experiment_results_dataframe)) + ' pairs right!')
    print('\nAccuracy of CONTROL approach: ' + str(accuracy_control_tdi))
    # Check if ndcg agree with stat weight ab_score
    stat_weight_correctly_guessed_winner_rankers = len(experiment_results_dataframe[
                                                       experiment_results_dataframe['avg_NDCG_winning_ranker'] ==
                                                       experiment_results_dataframe[
                                                           'stat_weight_interleaving_winner']])
    accuracy_stat_weight_tdi = stat_weight_correctly_guessed_winner_rankers / len(experiment_results_dataframe)
    print('\nThe STAT WEIGHT approach got: ' + str(stat_weight_correctly_guessed_winner_rankers) + '/' + str(
        len(experiment_results_dataframe)) + ' pairs right!')
    print('\nAccuracy of STAT WEIGHT approach: ' + str(accuracy_stat_weight_tdi))

    # Computing pruning ab_score
    if not only_statistical_significant_queries.empty:
        print('\nComputing standard AB score on pruned dataset')
        start_ab_pruning = time.time()
        only_statistical_significant_queries.drop(columns=['interleaving_winner_clicks', 'interleaving_total_clicks'],
                                                  inplace=True)
        only_statistical_significant_queries = utils.computing_winning_ranker_ab_score(
            only_statistical_significant_queries)
        only_statistical_significant_queries.drop(columns=['query_id', 'interleaving_winner'], inplace=True)
        only_statistical_significant_queries = only_statistical_significant_queries.drop_duplicates()
        end_ab_pruning = time.time()
        time_ab_pruning = end_ab_pruning - start_ab_pruning
        print('Time for ab score pruning: ' + str(time_ab_pruning))

        # Check if ndcg agree with pruning ab_score
        statistical_pruning_correctly_guessed_winner_rankers = len(
            only_statistical_significant_queries[
                only_statistical_significant_queries['avg_NDCG_winning_ranker'] == only_statistical_significant_queries[
                    'control_interleaving_winner']])

        accuracy_pruning_tdi = statistical_pruning_correctly_guessed_winner_rankers / len(experiment_results_dataframe)
        print('\nSTAT PRUNING approach got: ' + str(statistical_pruning_correctly_guessed_winner_rankers) + '/' + str(
            len(experiment_results_dataframe)) + ' pairs right!')
        print('\nAccuracy of STAT PRUNING approach: ' + str(accuracy_pruning_tdi))
    else:
        print('\n!!!!!!!!! The pruning removes all the queries for all the rankers !!!!!!!!!!')

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))


def interleave_iteration(dataframe_array, ranked_list_cache):
    interleaving_column_result = []
    for idx in range(0, len(dataframe_array)):
        ranker_a = dataframe_array[idx][0]
        ranker_b = dataframe_array[idx][1]
        query_id = dataframe_array[idx][2]
        # a ranked list is an array [document_ids, ratings]
        # ranked_list_a[0] is the list of ordered document Ids
        # ranked_list_a[1] is the list of ordered ratings
        ranked_list_a = ranked_list_cache[str(ranker_a) + '_' + str(query_id)]
        ranked_list_b = ranked_list_cache[str(ranker_b) + '_' + str(query_id)]
        interleaving_column_result.append(utils.execute_team_draft_interleaving(
            ranked_list_a[0], ranked_list_a[1],
            ranked_list_b[0], ranked_list_b[1]))
        if idx % 100000 == 0:
            print(str(idx)+' interleaving column size: ' + str(asizeof(interleaving_column_result)))
    print('final interleaving column size: ' + str(asizeof(interleaving_column_result)))
    return interleaving_column_result


def clicks_generation_iteration(interleaved_ranked_lists, click_generation_top_k, click_generation_realistic):
    clicks_results = []
    click_distribution_per_rating = np.zeros(5, dtype=np.int)
    for idx in range(0, len(interleaved_ranked_lists)):
        clicks_results.append(utils.simulate_clicks(interleaved_ranked_lists[idx], click_generation_top_k,                                            click_generation_realistic, click_distribution_per_rating))
        if idx % 100000 == 0:
            print(str(idx)+' clicks column size: ' + str(asizeof(clicks_results)))
    print('final clicks column size: ' + str(asizeof(clicks_results)))
    total_clicks = np.sum(click_distribution_per_rating)
    print('Total Clicks: ' + str(total_clicks))
    print('Click Distribution per rating: ' + str(click_distribution_per_rating))
    print('Relevance 0: ' + str(click_distribution_per_rating[0]/total_clicks))
    print('Relevance 1: ' + str(click_distribution_per_rating[1]/total_clicks))
    print('Relevance 2: ' + str(click_distribution_per_rating[2]/total_clicks))
    print('Relevance 3: ' + str(click_distribution_per_rating[3]/total_clicks))
    print('Relevance 4: ' + str(click_distribution_per_rating[4]/total_clicks))
    return clicks_results


def aggregate_interleaving_clicks_per_model_iteration(interleaved_ranked_lists_clicks):
    aggregated_clicks_column = np.empty([len(interleaved_ranked_lists_clicks), 3], dtype="uint16")
    for idx in range(0, len(interleaved_ranked_lists_clicks)):
        aggregated_clicks_column[idx] = utils.aggregate_clicks_per_ranker(interleaved_ranked_lists_clicks[idx])
    print('final clicks column size: ' + str(asizeof(aggregated_clicks_column)))
    return aggregated_clicks_column
