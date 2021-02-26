import utils
import pandas as pd
import numpy as np
import time
from datetime import datetime
# from guppy import hpy


def start_experiment(dataset_path, seed, query_set=1000, max_range_pair=138, experiment_one_bis=False):
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    # h = hpy()
    # print('Starting memory usage:')
    # print(h.heap())
    # print()

    # load dataframe
    print('Loading dataframe')
    start_loading = time.time()
    training_dataset = utils.load_dataframe(dataset_path)
    end_loading = time.time()
    time_for_loading = end_loading - start_loading
    print('Time for loading dataframe: ' + str(time_for_loading))

    # Fixed subset of 1000 queries
    print('Selecting queries')
    if not experiment_one_bis:
        set_of_queries = training_dataset.queryId.unique()[:query_set]
    else:
        set_of_queries = utils.generate_set_with_search_demand_curve(training_dataset)

    # h = hpy()
    # print('After load memory usage:')
    # print(h.heap())
    # print()

    # Set up the experiment dataframe, each row is a triple <rankerA,rankerB,queryId>
    # Rankers goes from 1 to 136 (therefore our range goes from 1 to 137)
    print('\nComputing experiment dataframe')
    start_computing_experiment_df = time.time()
    experiment_data = []
    for ranker_a in range(1, max_range_pair):
        for ranker_b in range(ranker_a + 1, max_range_pair):
            for query_index in range(0, len(set_of_queries)):
                query_id = set_of_queries[query_index]
                experiment_data.append([ranker_a, ranker_b, query_id])
    experiment_dataframe = pd.DataFrame(experiment_data, columns=['rankerA_id', 'rankerB_id', 'query_id'])
    experiment_dataframe['rankerA_list'] = np.nan
    experiment_dataframe['rankerA_list'] = experiment_dataframe['rankerA_list'].astype(object)
    experiment_dataframe['rankerB_list'] = np.nan
    experiment_dataframe['rankerB_list'] = experiment_dataframe['rankerB_list'].astype(object)

    experiment_dataframe['rankerA_ratings'] = np.nan
    experiment_dataframe['rankerA_ratings'] = experiment_dataframe['rankerA_ratings'].astype(object)
    experiment_dataframe['rankerB_ratings'] = np.nan
    experiment_dataframe['rankerB_ratings'] = experiment_dataframe['rankerB_ratings'].astype(object)

    experiment_dataframe['rankerA_NDCG'] = np.nan
    experiment_dataframe['rankerB_NDCG'] = np.nan

    experiment_dataframe['rankerA_avg_NDCG'] = np.nan
    experiment_dataframe['rankerB_avg_NDCG'] = np.nan

    experiment_dataframe['avg_NDCG_winning_ranker'] = np.nan

    end_computing_experiment_df = time.time()
    time_computing_experiment_df = end_computing_experiment_df - start_computing_experiment_df
    print('Time for computing experiment df: ' + str(time_computing_experiment_df))

    # Let's add to each row :
    # ranked list for rankerA, ranker list for rankerB, ratings for rankerA, ratings for rankerB and interleaved list
    print('\nComputing Interleaving')
    start_interleaving = time.time()
    for ranker in range(1, max_range_pair):
        for query_index in range(0, len(set_of_queries)):
            chosen_query_id = set_of_queries[query_index]
            query_selected_documents = training_dataset[training_dataset['queryId'] == chosen_query_id]
            ranked_list = query_selected_documents.sort_values(by=[ranker], ascending=False)
            ranked_list_ids = ranked_list.index.values
            ranked_list_ratings = ranked_list['relevance'].values
            ndcg_per_query_ranker = utils.compute_ndcg(ranked_list)

            indexes_to_change = experiment_dataframe.loc[
                (experiment_dataframe['rankerA_id'] == ranker) &
                (experiment_dataframe['query_id'] == chosen_query_id)].index.values
            experiment_dataframe.loc[indexes_to_change, 'rankerA_list'] = pd.Series(
                [ranked_list_ids] * len(indexes_to_change), index=indexes_to_change)
            experiment_dataframe.loc[indexes_to_change, 'rankerA_ratings'] = pd.Series(
                [ranked_list_ratings] * len(indexes_to_change), index=indexes_to_change)

            indexes_to_change = experiment_dataframe.loc[
                (experiment_dataframe['rankerB_id'] == ranker) &
                (experiment_dataframe['query_id'] == chosen_query_id)].index.values
            experiment_dataframe.loc[indexes_to_change, 'rankerB_list'] = pd.Series(
                [ranked_list_ids] * len(indexes_to_change), index=indexes_to_change)
            experiment_dataframe.loc[indexes_to_change, 'rankerB_ratings'] = pd.Series(
                [ranked_list_ratings] * len(indexes_to_change), index=indexes_to_change)

            experiment_dataframe.loc[
                (experiment_dataframe['rankerA_id'] == ranker) &
                (experiment_dataframe['query_id'] == chosen_query_id), 'rankerA_NDCG'] = ndcg_per_query_ranker
            experiment_dataframe.loc[
                (experiment_dataframe['rankerB_id'] == ranker) &
                (experiment_dataframe['query_id'] == chosen_query_id), 'rankerB_NDCG'] = ndcg_per_query_ranker
        if ranker < max_range_pair - 1:
            single_avg_ndcg = experiment_dataframe.loc[
                (experiment_dataframe['rankerA_id'] == ranker)].head(query_set)['rankerA_NDCG']
            avg_ndcg = sum(single_avg_ndcg) / len(single_avg_ndcg)
            experiment_dataframe.loc[
                (experiment_dataframe['rankerA_id'] == ranker), 'rankerA_avg_NDCG'] = avg_ndcg
            experiment_dataframe.loc[
                (experiment_dataframe['rankerB_id'] == ranker), 'rankerB_avg_NDCG'] = avg_ndcg
        else:
            single_avg_ndcg = experiment_dataframe.loc[
                (experiment_dataframe['rankerB_id'] == ranker)].head(query_set)['rankerB_NDCG']
            avg_ndcg = sum(single_avg_ndcg) / len(single_avg_ndcg)
            experiment_dataframe.loc[
                (experiment_dataframe['rankerA_id'] == ranker), 'rankerA_avg_NDCG'] = avg_ndcg
            experiment_dataframe.loc[
                (experiment_dataframe['rankerB_id'] == ranker), 'rankerB_avg_NDCG'] = avg_ndcg

    experiment_dataframe.drop(columns=['rankerA_NDCG', 'rankerB_NDCG'], inplace=True)
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] > experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 'a'
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] == experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 't'
    indexes_to_change = experiment_dataframe.loc[experiment_dataframe['rankerA_avg_NDCG'] < experiment_dataframe[
        'rankerB_avg_NDCG']].index.values
    experiment_dataframe.loc[indexes_to_change, 'avg_NDCG_winning_ranker'] = 'b'
    experiment_dataframe.drop(columns=['rankerA_avg_NDCG', 'rankerB_avg_NDCG'], inplace=True)

    experiment_dataframe['interleaved_list'] = np.vectorize(utils.execute_tdi_interleaving)(
        experiment_dataframe['rankerA_list'], experiment_dataframe['rankerA_ratings'],
        experiment_dataframe['rankerB_list'], experiment_dataframe['rankerB_ratings'], seed)
    experiment_dataframe.drop(columns=['rankerA_list', 'rankerA_ratings', 'rankerB_list', 'rankerB_ratings'], axis=1,
                              inplace=True)
    end_interleaving = time.time()
    time_for_interleaving = end_interleaving - start_interleaving
    print('Time for interleaving: ' + str(time_for_interleaving))

    # At this point we have the interleaved list in a column, we should calculate the clicks
    print('\nGenerating Clicks')
    start_generating_clicks = time.time()
    experiment_dataframe['clicks'] = np.vectorize(utils.simulate_clicks)(experiment_dataframe['interleaved_list'], seed)
    experiment_dataframe.drop(columns=['interleaved_list'], inplace=True)
    experiment_dataframe.rename(columns={'clicks': 'clicked_interleaved_list'}, inplace=True)
    end_generating_clicks = time.time()
    time_generating_clicks = end_generating_clicks - start_generating_clicks
    print('Time for generating clicks: ' + str(time_generating_clicks))

    # Computing the per query winning model/ranker
    print('\nComputing per query winning model')
    start_computing_per_query_winner = time.time()
    experiment_dataframe['clicks_per_ranker'], experiment_dataframe['total_clicks'], experiment_dataframe[
        'per_query_TDI_winning_ranker'] = np.vectorize(utils.compute_winning_model)(experiment_dataframe[
                                                                                  'clicked_interleaved_list'])
    experiment_dataframe.drop(columns=['clicked_interleaved_list'], inplace=True)
    end_computing_per_query_winner = time.time()
    time_computing_per_query_winner = end_computing_per_query_winner - start_computing_per_query_winner
    print('Time for computing per query winning model: ' + str(time_computing_per_query_winner))

    # Pruning
    print('\nPruning')
    start_pruning = time.time()
    experiment_dataframe_pruned = utils.pruning(experiment_dataframe)
    experiment_dataframe.drop(columns=['clicks_per_ranker', 'total_clicks'], inplace=True)
    end_pruning = time.time()
    time_pruning = end_pruning - start_pruning
    print('Time for pruning: ' + str(time_pruning))

    # Computing standard ab_score
    print('\nComputing standard AB score')
    start_ab = time.time()
    experiment_dataframe = utils.computing_winning_model_ab_score(experiment_dataframe)
    experiment_dataframe.drop(columns=['query_id', 'per_query_TDI_winning_ranker'], inplace=True)
    experiment_dataframe = experiment_dataframe.drop_duplicates()
    end_ab = time.time()
    time_ab = end_ab - start_ab
    print('Time for ab score: ' + str(time_ab))

    # Check if ndcg agree with ab_score
    ranker_pair_agree = len(experiment_dataframe[
        experiment_dataframe['avg_NDCG_winning_ranker'] == experiment_dataframe['TDI_winning_ranker']])

    # Computing pruning ab_score
    if not experiment_dataframe_pruned.empty:
        print('\nComputing standard AB score on pruned dataset')
        start_ab_pruning = time.time()
        experiment_dataframe_pruned.drop(columns=['clicks_per_ranker', 'total_clicks'], inplace=True)
        experiment_dataframe_pruned = utils.computing_winning_model_ab_score(experiment_dataframe_pruned)
        experiment_dataframe_pruned.drop(columns=['query_id', 'per_query_TDI_winning_ranker'], inplace=True)
        experiment_dataframe_pruned = experiment_dataframe_pruned.drop_duplicates()
        end_ab_pruning = time.time()
        time_ab_pruning = end_ab_pruning - start_ab_pruning
        print('Time for ab score pruning: ' + str(time_ab_pruning))

        # Check if ndcg agree with pruning ab_score
        ranker_pair_pruning_agree = len(
            experiment_dataframe_pruned[
                experiment_dataframe_pruned['avg_NDCG_winning_ranker'] == experiment_dataframe_pruned[
                    'TDI_winning_ranker']])

        accuracy_pruning_tdi = ranker_pair_pruning_agree / len(experiment_dataframe)
        print('\nAccuracy of pruning tdi on all pairs of rankers: ' + str(accuracy_pruning_tdi))
    else:
        print('\n!!!!!!!!! The pruning removes all the queries for all the rankers !!!!!!!!!!')

    accuracy_standard_tdi = ranker_pair_agree / len(experiment_dataframe)
    print('\nAccuracy of tdi on all pairs of rankers: ' + str(accuracy_standard_tdi))

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))
