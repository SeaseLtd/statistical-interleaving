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
    training_dataset = utils.load_dataframe(dataset_path)

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = training_dataset.queryId.unique()[:query_set]
    else:
        set_of_queries = utils.generate_set_with_search_demand_curve(training_dataset)

    # h = hpy()
    # print('After load memory usage:')
    # print(h.heap())
    # print()

    #Set up the experiment dataframe, each row is a triple <ranker1,ranker2,queryId>
    experiment_data = []
    for ranker_a in range(1, max_range_pair):
        for ranker_b in range(ranker_a + 1, max_range_pair):
            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')
            for query_index in range(0, len(set_of_queries)):
                query_id = set_of_queries[query_index]
                experiment_data.append([ranker_a, ranker_b, query_id])
    experiment_dataframe = pd.DataFrame(experiment_data, columns=[ 'ranker1_id', 'ranker2_id','query_id'])
    experiment_dataframe['ranker1_list'] = np.nan
    experiment_dataframe['ranker1_list'] = experiment_dataframe['ranker1_list'].astype(object)
    experiment_dataframe['ranker2_list'] = np.nan
    experiment_dataframe['ranker2_list'] = experiment_dataframe['ranker2_list'].astype(object)

    experiment_dataframe['ranker1_ratings'] = np.nan
    experiment_dataframe['ranker1_ratings'] = experiment_dataframe['ranker1_ratings'].astype(object)
    experiment_dataframe['ranker2_ratings'] = np.nan
    experiment_dataframe['ranker2_ratings'] = experiment_dataframe['ranker2_ratings'].astype(object)

    experiment_dataframe['ranker1_NDCG'] = np.nan
    experiment_dataframe['ranker2_NDCG'] = np.nan

    #let's add to each row : ranked list for ranker 1, ranker list for ranker 2, ratings for ranker 1, ratings for ranker 2 and the interleaved list
    for ranker in range(1, max_range_pair):
        for query_index in range(0, len(set_of_queries)):
            chosen_query_id = set_of_queries[query_index]
            query_selected_documents = training_dataset[training_dataset['queryId'] == chosen_query_id]
            ranked_list = query_selected_documents.sort_values(by=[ranker], ascending=False)
            ranked_list_ids = ranked_list.index.values
            ranked_list_ratings = ranked_list['relevance'].values
            ndcg_per_query_ranker = utils.compute_ndcg(ranked_list)

            indexes_to_change = experiment_dataframe.loc[(experiment_dataframe['ranker1_id'] == ranker) & (experiment_dataframe['query_id'] == chosen_query_id)].index.values
            experiment_dataframe.loc[indexes_to_change, 'ranker1_list'] = pd.Series([ranked_list_ids] * len(indexes_to_change), index=indexes_to_change)
            experiment_dataframe.loc[indexes_to_change, 'ranker1_ratings'] = pd.Series([ranked_list_ratings] * len(indexes_to_change), index=indexes_to_change)

            indexes_to_change = experiment_dataframe.loc[(experiment_dataframe['ranker2_id'] == ranker) & (experiment_dataframe['query_id'] == chosen_query_id)].index.values
            experiment_dataframe.loc[indexes_to_change, 'ranker2_list'] = pd.Series([ranked_list_ids] * len(indexes_to_change), index=indexes_to_change)
            experiment_dataframe.loc[indexes_to_change, 'ranker2_ratings'] = pd.Series([ranked_list_ratings] * len(indexes_to_change), index=indexes_to_change)

            experiment_dataframe.loc[(experiment_dataframe['ranker1_id'] == ranker) & (experiment_dataframe['query_id'] == chosen_query_id), 'ranker1_NDCG'] = ndcg_per_query_ranker
            experiment_dataframe.loc[(experiment_dataframe['ranker2_id'] == ranker) & (experiment_dataframe['query_id'] == chosen_query_id), 'ranker2_NDCG'] = ndcg_per_query_ranker

    experiment_dataframe['interleaved_list'] = np.vectorize(utils.execute_tdi_interleaving)(experiment_dataframe['ranker1_list'],experiment_dataframe['ranker1_ratings'], experiment_dataframe['ranker2_list'],experiment_dataframe['ranker2_ratings'], seed)
    experiment_dataframe.drop(columns=['ranker1_list','ranker1_ratings','ranker2_list','ranker2_ratings'], axis=1, inplace=True)
    end_interleaving = time.time()
    timeForInterleaving = end_interleaving - start_total
    print(timeForInterleaving)

    #at this point we have the interleaved list in a column, we should calculate the clicks then

    # Iterate on all possible pairs of rankers/models (from 1 to 137)
    for ranker_a in range(1, max_range_pair):
        for ranker_b in range(ranker_a + 1, max_range_pair):
            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')

            # Iterate on all 1000 queries (from 0 to 1000)
            for query_index in range(0, len(set_of_queries)):
                query_id = set_of_queries[query_index]
                # Simulate clicks
                interleaved_list = utils.simulate_clicks(interleaved_list, seed)

                # Computing the per query winning model/ranker
                all_queries_winning_model.append(utils.compute_winning_model(interleaved_list, chosen_query_id))

            # Computing average ndcg to find winning model/ranker
            ndcg_winning_model = utils.compute_ndcg_winning_model(list_ndcg_model_a, list_ndcg_model_b)

            # Pruning
            all_queries_winning_model = pd.DataFrame.from_records(all_queries_winning_model)
            all_queries_winning_model.rename(columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
                                                      3: 'winning_model'}, inplace=True)
            all_queries_winning_model_pruned = utils.pruning(all_queries_winning_model)

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

    # Computing avg times per step
    avg_time_per_pair = sum(each_pair_time) / len(each_pair_time)
    print('\nAverage times per step:')
    print(avg_time_per_pair)
    print('Compared ' + str(len(each_pair_time)) + ' pairs')

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))
