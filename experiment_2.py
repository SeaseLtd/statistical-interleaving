import utils
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.stats.proportion import proportion_confint


def start_experiment(dataset_path, seed, realistic_model=False):
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    print("Experiment started at:", start_time)
    print()

    subset_of_rankers = [(64, 14), (108, 84), (134, 96), (97, 106), (77, 1)]

    dataset = utils.load_dataframe(dataset_path)
    final_accuracy_standard_tdi = {}
    final_accuracy_pruning_tdi = {}

    # Iterate on pair of rankers:
    for ranker_pair in subset_of_rankers:
        start_each_pair = datetime.now()

        ranker_a = ranker_pair[0]
        ranker_b = ranker_pair[1]

        accuracy_standard_tdi, accuracy_pruning_tdi, interval_standard_tdi, interval_pruning_tdi = \
            per_query_size_evaluation(dataset, ranker_a, ranker_b, seed, realistic_model)

        final_accuracy_standard_tdi[ranker_pair] = accuracy_standard_tdi
        final_accuracy_pruning_tdi[ranker_pair] = accuracy_pruning_tdi

        print('Standard tdi: Confidence interval for pair (' + str(ranker_a) + ',' + str(ranker_b) + ') is ' +
              str(interval_standard_tdi))
        if len(interval_pruning_tdi) > 0:
            print('Pruning tdi: Confidence interval for pair (' + str(ranker_a) + ',' + str(ranker_b) + ') is ' +
                  str(interval_pruning_tdi))
        else:
            print('Pruning removes all queries\n')

        end_each_pair = datetime.now()
        print('Execution time each pair: ' + str(end_each_pair - start_each_pair) + '\n')

    print('Accuracy of tdi for all sizes and rankers:')
    print(final_accuracy_standard_tdi)
    print('Accuracy of pruning tdi for all sizes and rankers:')
    print(final_accuracy_pruning_tdi)

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    print("\nExperiment ended at:", end_time)
    print('Total experiment time: ' + str(end - start))


def per_query_size_evaluation(dataset, ranker_a, ranker_b, seed, realistic_model):
    accuracy_standard_tdi = {}
    accuracy_pruning_tdi = {}

    # Iterate on all possible query set sizes (from 1 to 10001)
    for query_set_size in range(1, 10001):

        ranker_pair_agree, ranker_pair_pruning_agree, interval_standard_tdi, interval_pruning_tdi = \
            repetition_1000_times(dataset, query_set_size, ranker_a, ranker_b, seed, realistic_model)

        accuracy_standard_tdi[query_set_size] = sum(ranker_pair_agree) / len(ranker_pair_agree)
        if len(ranker_pair_pruning_agree) > 0:
            accuracy_pruning_tdi[query_set_size] = sum(ranker_pair_pruning_agree) / len(ranker_pair_pruning_agree)

    return accuracy_standard_tdi, accuracy_pruning_tdi, interval_standard_tdi, interval_pruning_tdi


def repetition_1000_times(dataset, query_set_size, ranker_a, ranker_b, seed, realistic_model):
    ranker_pair_agree = []
    ranker_pair_pruning_agree = []
    interval_standard_tdi = {}
    interval_pruning_tdi = {}

    # For each query set size we repeat the experiment 1000 times
    for repetition in range(0, 1000):
        all_queries_winning_model = []
        list_ndcg_model_a = []
        list_ndcg_model_b = []

        if repetition == 50 or repetition == 100 or repetition == 500:
            print('round ' + str(repetition) + ' for same query set size: ' + str(query_set_size))

        np.random.seed(repetition)
        set_of_queries = np.random.choice(dataset.queryId.unique(), size=query_set_size, replace=False)

        for chosen_query_id in set_of_queries:
            # Reduce the dataset to the documents for the selected query
            query_selected_documents_1 = dataset[dataset['queryId'] == chosen_query_id]

            # Creating the models' ranked lists
            ranked_list_model_a = query_selected_documents_1.sort_values(by=[ranker_a], ascending=False)
            ranked_list_model_b = query_selected_documents_1.sort_values(by=[ranker_b], ascending=False)

            # Computing ndcg
            list_ndcg_model_a.append(utils.compute_ndcg(ranked_list_model_a))
            list_ndcg_model_b.append(utils.compute_ndcg(ranked_list_model_b))

            # Creating interleaved list
            interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b, seed)

            # Simulate clicks
            interleaved_list = utils.simulate_clicks(interleaved_list, seed, realistic_model)

            # Computing the per query winning model/ranker
            all_queries_winning_model.append(utils.compute_winning_model(interleaved_list, chosen_query_id))

        # Computing average ndcg to find winning model/ranker
        ndcg_winning_model = utils.compute_ndcg_winning_model(list_ndcg_model_a, list_ndcg_model_b)

        # Pruning
        all_queries_winning_model = pd.DataFrame.from_records(all_queries_winning_model)
        all_queries_winning_model.rename(
            columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
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
            ab_score_pruning_winning_model = utils.computing_winning_model_ab_score(all_queries_winning_model_pruned)

            # Check if ndcg agree with pruning ab_score
            if ndcg_winning_model == ab_score_pruning_winning_model:
                ranker_pair_pruning_agree.append(1)
            else:
                ranker_pair_pruning_agree.append(0)

    interval_standard_tdi[query_set_size] = proportion_confint(sum(ranker_pair_agree), len(ranker_pair_agree),
                                                               method='wilson')
    if len(ranker_pair_pruning_agree) > 0:
        interval_pruning_tdi[query_set_size] = proportion_confint(sum(ranker_pair_pruning_agree),
                                                                  len(ranker_pair_pruning_agree), method='wilson')

    return ranker_pair_agree, ranker_pair_pruning_agree, interval_standard_tdi, interval_pruning_tdi
