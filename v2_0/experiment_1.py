import utils
import pandas as pd
import numpy as np
import time
from datetime import datetime
# from guppy import hpy


def start_experiment(dataset_path, seed, query_set=1000, max_range_pair=137, experiment_one_bis=False):
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    # h = hpy()
    # print('Starting memory usage:')
    # print(h.heap())
    # print()

    # load dataframe
    print('Loading dataframe')
    training_dataset = utils.load_dataframe(dataset_path)

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
    # Rankers goes from 1 to 137
    print('Computing experiment dataframe')
    experiment_data = []
    for ranker_a in range(1, max_range_pair):
        for ranker_b in range(ranker_a + 1, max_range_pair):
            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')
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

    # Let's add to each row :
    # ranked list for rankerA, ranker list for rankerB, ratings for rankerA, ratings for rankerB and interleaved list
    print('Computing Interleaving')
    rankers_avg_ndcg = []
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

    experiment_dataframe['avg_NDCG_winning_ranker'] = np.where(
        experiment_dataframe['rankerA_avg_NDCG'] > experiment_dataframe['rankerB_avg_NDCG'], 'a', 'n')
    experiment_dataframe['avg_NDCG_winning_ranker'] = np.where(
        experiment_dataframe['rankerA_avg_NDCG'] == experiment_dataframe['rankerB_avg_NDCG'], 't', 'n')
    experiment_dataframe['avg_NDCG_winning_ranker'] = np.where(
        experiment_dataframe['rankerA_avg_NDCG'] < experiment_dataframe['rankerB_avg_NDCG'], 'b', 'n')
    experiment_dataframe.drop(columns=['rankerA_avg_NDCG', 'rankerB_avg_NDCG', 'rankerA_NDCG', 'rankerB_NDCG'],
                              inplace=True)

    experiment_dataframe['interleaved_list'] = np.vectorize(utils.execute_tdi_interleaving)(
        experiment_dataframe['rankerA_list'], experiment_dataframe['rankerA_ratings'],
        experiment_dataframe['rankerB_list'], experiment_dataframe['rankerB_ratings'], seed)
    experiment_dataframe.drop(columns=['rankerA_list', 'rankerA_ratings', 'rankerB_list', 'rankerB_ratings'], axis=1,
                              inplace=True)
    end_interleaving = time.time()
    time_for_interleaving = end_interleaving - start_total
    print(time_for_interleaving)

    # At this point we have the interleaved list in a column, we should calculate the clicks
    print('Generating Clicks')
    experiment_dataframe['clicks'] = np.vectorize(utils.simulate_clicks)(experiment_dataframe['interleaved_list'], seed)
    experiment_dataframe.drop(columns=['interleaved_list'], inplace=True)
    experiment_dataframe.rename(columns={'clicks': 'clicked_interleaved_list'}, inplace=True)

    # Computing the per query winning model/ranker
    print('Computing per query winning model')
    experiment_dataframe['clicks_per_ranker'], experiment_dataframe['total_clicks'], experiment_dataframe[
        'TDI_winning_ranker'] = np.vectorize(utils.compute_winning_model)(experiment_dataframe[
                                                                                  'clicked_interleaved_list'])
    experiment_dataframe.drop(columns=['clicked_interleaved_list'], inplace=True)

    # Pruning
    experiment_dataframe_pruned = utils.pruning(experiment_dataframe)

    # TO CONTINUE

    # Computing standard ab_score
    ab_score_winning_model = utils.computing_winning_model_ab_score(all_queries_winning_model)

    # Check if ndcg agree with ab_score
    if ndcg_winning_model == ab_score_winning_model:
        ranker_pair_agree.append(1)
    else:
        ranker_pair_agree.append(0)

    # Computing pruning ab_score
    if not all_queries_winning_model_pruned.empty:
        ab_score_pruning_winning_model = utils.computing_winning_model_ab_score(
            all_queries_winning_model_pruned)

        # Check if ndcg agree with pruning ab_score
        if ndcg_winning_model == ab_score_pruning_winning_model:
            ranker_pair_pruning_agree.append(1)
        else:
            ranker_pair_pruning_agree.append(0)
    else:
        ranker_pair_pruning_agree.append(0)
        print('The pruning removes all the queries')

    end_each_pair = time.time()
    each_pair_time.append(end_each_pair - start_each_pair)

    accuracy_standard_tdi = sum(ranker_pair_agree) / len(ranker_pair_agree)
    print('\nAccuracy of tdi on all pairs of rankers:')
    print(accuracy_standard_tdi)

    if len(ranker_pair_pruning_agree) > 0:
        accuracy_pruning_tdi = sum(ranker_pair_pruning_agree) / len(
            ranker_pair_pruning_agree)
        print('Accuracy of pruning tdi on all pairs of rankers:')
        print(accuracy_pruning_tdi)
    else:
        print('Pruning removes all queries for all pairs')


    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))
