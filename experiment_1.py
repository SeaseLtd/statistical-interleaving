import gc

import utils
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pympler.asizeof import asizeof


# Model A = 0 , Model B = 1, 2 means a tie
def start_experiment(dataset_path, seed, queries_to_evaluate_count=1000, rankers_to_evaluate_count=136,
                     solr_aggregation_json_path=None, users_scaling_factor=1.0, experiment_one_long_tail=False):
    np.random.seed(seed)
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    # load dataframe
    print('Loading dataframe')
    start_loading = time.time()
    fold1_query_document_pairs = utils.load_dataframe(dataset_path)
    end_loading = time.time()
    time_for_loading = end_loading - start_loading
    print('Time for loading dataframe: ' + str(time_for_loading))
    # Remove duplicates
    fold1_query_document_pairs = fold1_query_document_pairs.drop_duplicates()
    print('Query-Document Pairs: ' + str(len(fold1_query_document_pairs)))
    print('Queries: ' + str(len(fold1_query_document_pairs['query_id'].unique())))
    print('Avg judged documents per query: ' + str(fold1_query_document_pairs['query_id'].value_counts().mean()))
    print(fold1_query_document_pairs.info(memory_usage='deep', verbose=False))
    # Retrieve a set of queries with a cardinality of query_set (1000 by default)
    print('Selecting queries')
    if experiment_one_long_tail:
        industrial_dataset = utils.from_solr_aggregations_dataset(solr_aggregation_json_path)
        set_of_queries = utils.generate_set_with_search_demand_curve(fold1_query_document_pairs, industrial_dataset, users_scaling_factor)
    else:
        set_of_queries = fold1_query_document_pairs['query_id'].unique()[:queries_to_evaluate_count]

    print('\nEach ranker is evaluated on queries: ' + str(len(set_of_queries)))
    # Set up the experiment dataframe, each row is a triple <rankerA,rankerB,query_id>
    # Rankers goes from 1 to 136 (therefore our range goes from 1 to 137)
    print('\nComputing experiment dataframe')
    start_computing_experiment_df = time.time()
    experiment_data = []
    print(rankers_to_evaluate_count)
    for ranker_a in range(1, rankers_to_evaluate_count + 1):
        for ranker_b in range(ranker_a + 1, rankers_to_evaluate_count + 1):
            for query_index in range(0, len(set_of_queries)):
                query_id = set_of_queries[query_index]
                experiment_data.append([ranker_a, ranker_b, query_id])
    experiment_dataframe = pd.DataFrame(experiment_data, columns=['rankerA_id', 'rankerB_id', 'query_id'])

    # Clean unuseful data structures
    del experiment_data
    gc.collect()

    experiment_dataframe['rankerA_avg_NDCG'] = np.nan
    experiment_dataframe['rankerB_avg_NDCG'] = np.nan

    experiment_dataframe['avg_NDCG_winning_ranker'] = np.nan

    end_computing_experiment_df = time.time()
    time_computing_experiment_df = end_computing_experiment_df - start_computing_experiment_df
    print('Time for computing experiment df: ' + str(time_computing_experiment_df))

    # Let's add to each row :
    # ranked list for rankerA, ranker list for rankerB, ratings for rankerA, ratings for rankerB and interleaved list
    print('\nComputing ranked lists and NDCG')
    start_lists = time.time()
    ranked_list_cache = {}
    for ranker in range(1, rankers_to_evaluate_count + 1):
        ndcg_per_ranker = []
        for query_index in range(0, len(set_of_queries)):
            chosen_query_id = set_of_queries[query_index]
            query_selected_documents = fold1_query_document_pairs.loc[fold1_query_document_pairs['query_id'] == chosen_query_id]
            ranked_list = query_selected_documents.sort_values(by=[ranker], ascending=False)
            ranked_list_ids = ranked_list.index.values
            ranked_list_ratings = ranked_list['relevance'].values
            ndcg_per_query_ranker = utils.compute_ndcg(ranked_list_ratings)
            ranked_list_cache[str(ranker) + '_' + str(chosen_query_id)] = [ranked_list_ids, ranked_list_ratings]
            ndcg_per_ranker.append(ndcg_per_query_ranker)
        avg_ndcg = sum(ndcg_per_ranker) / len(ndcg_per_ranker)
        print('\nRanker['+str(ranker)+'] AVG NDCG:' + str(avg_ndcg))
        experiment_dataframe.loc[
            (experiment_dataframe['rankerA_id'] == ranker), 'rankerA_avg_NDCG'] = avg_ndcg
        experiment_dataframe.loc[
            (experiment_dataframe['rankerB_id'] == ranker), 'rankerB_avg_NDCG'] = avg_ndcg

    # Clean unuseful data structures
    del query_selected_documents, ranked_list, ranked_list_ids, ranked_list_ratings, ndcg_per_query_ranker, \
        set_of_queries, fold1_query_document_pairs
    gc.collect()

    # model A wins -> 0
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] > experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 0
    # tie -> -1
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] == experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 2
    # model B wins -> 1
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] < experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 1
    experiment_dataframe.drop(columns=['rankerA_avg_NDCG', 'rankerB_avg_NDCG'], inplace=True)
    experiment_dataframe['avg_NDCG_winning_ranker'] = experiment_dataframe['avg_NDCG_winning_ranker'].astype(int)
    end_lists = time.time()
    time_for_interleaving = end_lists - start_lists
    print('Time for ranked lists and ratings: ' + str(time_for_interleaving))
    print(experiment_dataframe.info(memory_usage='deep', verbose=False))

    print('\nComputing Interleaving')
    start_interleaving = time.time()
    dataframe_array = experiment_dataframe.to_numpy()
    interleaving_column = iterate_interleaving(dataframe_array, ranked_list_cache)
    # Clean unuseful data structures
    gc.collect()
    end_interleaving = time.time()
    time_for_interleaving = end_interleaving - start_interleaving
    print('Time for interleaving: ' + str(time_for_interleaving))

    print('\nGenerating Clicks')
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_dataframe)))
    print(experiment_dataframe.info(memory_usage='deep', verbose=False))
    start_generating_clicks = time.time()
    clicked_interleaved = iterate_clicks_generation(interleaving_column)
    end_generating_clicks = time.time()
    time_generating_clicks = end_generating_clicks - start_generating_clicks
    # Clean unuseful data structures
    del interleaving_column
    gc.collect()
    print('Time for generating clicks: ' + str(time_generating_clicks))
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_dataframe)))
    print(experiment_dataframe.info(memory_usage='deep', verbose=False))

    # Computing the per query winning model/ranker
    print('\nComputing per query sum of clicks')
    start_computing_per_query_winner = time.time()
    counted_clicks = iterate_winner_calculation(clicked_interleaved).T
    experiment_dataframe['interleaving_a_clicks'] = pd.Series(counted_clicks[0])
    experiment_dataframe['interleaving_b_clicks'] = pd.Series(counted_clicks[1])
    experiment_dataframe['interleaving_total_clicks'] = pd.Series(counted_clicks[2])

    end_computing_per_query_winner = time.time()
    time_computing_per_query_winner = end_computing_per_query_winner - start_computing_per_query_winner
    del clicked_interleaved, counted_clicks
    gc.collect()
    print('Current memory for the DataFrame: ' + str(asizeof(experiment_dataframe)))
    print(experiment_dataframe.info(memory_usage='deep', verbose=False))
    print('Time for computing per query winning model: ' + str(time_computing_per_query_winner))
    print('Ranker combinations:' + str(experiment_dataframe.groupby(['rankerA_id', 'rankerB_id']).ngroups))
    # Removing pairs where no click happened
    experiment_dataframe = experiment_dataframe[
        experiment_dataframe.interleaving_total_clicks > 0]
    print('\nAfter the removal of queries that didn\'t show any click')
    print('Ranker combinations:' + str(experiment_dataframe.groupby(['rankerA_id', 'rankerB_id']).ngroups))
    # Calculate Winner
    print('\nCalculate Winner')
    if experiment_one_long_tail:
        experiment_dataframe = pd.DataFrame(experiment_dataframe.groupby(['rankerA_id','rankerB_id','query_id']).agg({'avg_NDCG_winning_ranker':'first','interleaving_a_clicks': 'sum', 'interleaving_b_clicks': 'sum', 'interleaving_total_clicks':'sum'})).reset_index()

    experiment_dataframe['interleaving_winner_clicks'] = np.where(
        experiment_dataframe['interleaving_a_clicks'] > experiment_dataframe['interleaving_b_clicks'],
        experiment_dataframe['interleaving_a_clicks'],
        experiment_dataframe['interleaving_b_clicks'])

    experiment_dataframe['interleaving_winner'] = np.where(
        experiment_dataframe['interleaving_a_clicks'] > experiment_dataframe['interleaving_b_clicks'],
        0,
        1)

    experiment_dataframe['interleaving_winner'] = np.where(
        experiment_dataframe['interleaving_a_clicks'] == experiment_dataframe['interleaving_b_clicks'],
        2,
        experiment_dataframe['interleaving_winner'])

    experiment_dataframe.drop(columns=['interleaving_a_clicks', 'interleaving_b_clicks'], inplace=True)
    # Pruning
    print('\nPruning')
    start_pruning = time.time()
    only_statistical_significant_queries = utils.pruning(experiment_dataframe)
    experiment_dataframe.drop(columns=['interleaving_winner_clicks', 'interleaving_total_clicks'], inplace=True)
    end_pruning = time.time()
    time_pruning = end_pruning - start_pruning
    print('Time for pruning: ' + str(time_pruning))
    print('Only stat relevant rows')
    print(only_statistical_significant_queries.info(memory_usage='deep', verbose=False))

    # Computing standard ab_score
    print('\nComputing standard AB score')
    start_ab = time.time()
    experiment_dataframe = utils.computing_winning_model_ab_score(experiment_dataframe)
    end_ab = time.time()
    time_ab = end_ab - start_ab
    print('Time for ab score: ' + str(time_ab))

    # Computing Statistical weighted ab_score
    print('\nComputing Statistical Weighted AB score')
    start_ab_stat = time.time()
    experiment_dataframe = utils.computing_winning_model_ab_score(experiment_dataframe, True)
    end_ab_stat = time.time()
    time_ab_stat = end_ab_stat - start_ab_stat
    print('Time for ab stat score: ' + str(time_ab_stat))

    experiment_dataframe.drop(columns=['query_id', 'interleaving_winner', 'statistical_significance', 'statistical_weight'], inplace=True)
    experiment_dataframe = experiment_dataframe.drop_duplicates()

    # Check if ndcg agree with ab_score
    control_correctly_guessed_winner_rankers = len(experiment_dataframe[
                                                       experiment_dataframe['avg_NDCG_winning_ranker'] ==
                                                       experiment_dataframe[
                                                           'control_interleaving_winner']])
    accuracy_control_tdi = control_correctly_guessed_winner_rankers / len(experiment_dataframe)
    print('\nThe CONTROL approach got: ' + str(control_correctly_guessed_winner_rankers) + '/' + str(
        len(experiment_dataframe)) + ' pairs right!')
    print('\nAccuracy of CONTROL approach: ' + str(accuracy_control_tdi))
    # Check if ndcg agree with stat weight ab_score
    stat_weight_correctly_guessed_winner_rankers = len(experiment_dataframe[
                                                       experiment_dataframe['avg_NDCG_winning_ranker'] ==
                                                       experiment_dataframe[
                                                           'stat_weight_interleaving_winner']])
    accuracy_stat_weight_tdi = stat_weight_correctly_guessed_winner_rankers / len(experiment_dataframe)
    print('\nThe STAT WEIGHT approach got: ' + str(stat_weight_correctly_guessed_winner_rankers) + '/' + str(
        len(experiment_dataframe)) + ' pairs right!')
    print('\nAccuracy of STAT WEIGHT approach: ' + str(accuracy_stat_weight_tdi))

    # Computing pruning ab_score
    if not only_statistical_significant_queries.empty:
        print('\nComputing standard AB score on pruned dataset')
        start_ab_pruning = time.time()
        only_statistical_significant_queries.drop(columns=['interleaving_winner_clicks', 'interleaving_total_clicks'],
                                                  inplace=True)
        only_statistical_significant_queries = utils.computing_winning_model_ab_score(
            only_statistical_significant_queries)
        only_statistical_significant_queries.drop(columns=['query_id', 'interleaving_winner'], inplace=True)
        only_statistical_significant_queries = only_statistical_significant_queries.drop_duplicates()
        end_ab_pruning = time.time()
        time_ab_pruning = end_ab_pruning - start_ab_pruning
        print('Time for ab score pruning: ' + str(time_ab_pruning))

        # Check if ndcg agree with pruning ab_score
        paper_correctly_guessed_winner_rankers = len(
            only_statistical_significant_queries[
                only_statistical_significant_queries['avg_NDCG_winning_ranker'] == only_statistical_significant_queries[
                    'control_interleaving_winner']])

        accuracy_pruning_tdi = paper_correctly_guessed_winner_rankers / len(experiment_dataframe)
        print('\nOUR PAPER approach got: ' + str(paper_correctly_guessed_winner_rankers) + '/' + str(
            len(experiment_dataframe)) + ' pairs right!')
        print('\nAccuracy of OUR PAPER approach: ' + str(accuracy_pruning_tdi))
    else:
        print('\n!!!!!!!!! The pruning removes all the queries for all the rankers !!!!!!!!!!')

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))


def iterate_interleaving(dataframe_array, ranked_list_cache):
    interleaving_column = []
    for idx in range(0, len(dataframe_array)):
        ranker_a = dataframe_array[idx][0]
        ranker_b = dataframe_array[idx][1]
        query_id = dataframe_array[idx][2]
        ranker_a_row = ranked_list_cache[str(ranker_a) + '_' + str(query_id)]
        ranker_b_row = ranked_list_cache[str(ranker_b) + '_' + str(query_id)]
        interleaving_column.append(utils.execute_team_draft_interleaving(
            ranker_a_row[0], ranker_a_row[1],
            ranker_b_row[0], ranker_b_row[1]))
        if idx % 100000 == 0:
            print(str(idx)+' interleaving column size: ' + str(asizeof(interleaving_column)))
    print('final interleaving column size: ' + str(asizeof(interleaving_column)))
    return interleaving_column


def iterate_clicks_generation(interleaving_column):
    clicks_column = []
    for idx in range(0, len(interleaving_column)):
        clicks_column.append(utils.simulate_clicks(interleaving_column[idx]))
        if idx % 100000 == 0:
            print(str(idx)+' clicks column size: ' + str(asizeof(clicks_column)))
    print('final clicks column size: ' + str(asizeof(clicks_column)))
    return clicks_column

def iterate_winner_calculation(clicked_interleaved):
    aggregated_clicks_column = np.empty([len(clicked_interleaved), 3], dtype="uint16")
    for idx in range(0, len(clicked_interleaved)):
        aggregated_clicks_column[idx] = utils.compute_winning_model(clicked_interleaved[idx])
    print('final clicks column size: ' + str(asizeof(aggregated_clicks_column)))
    return aggregated_clicks_column
