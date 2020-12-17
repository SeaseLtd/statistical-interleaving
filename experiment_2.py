import utils
import numpy as np
import pandas as pd


def start_experiment(dataset_path, seed, realistic_model=False):
    subset_of_rankers = [(64, 14), (108, 84), (134, 96), (97, 106), (77, 1)]

    dataset = utils.load_dataframe(dataset_path)
    final_accuracy_standard_tdi = {}
    final_accuracy_pruning_tdi = {}

    # Iterate on pair of rankers:
    for ranker_pair in subset_of_rankers:
        ranker_a = ranker_pair[0]
        ranker_b = ranker_pair[1]
        ranker_pair_agree = []
        ranker_pair_pruning_agree = []
        accuracy_standard_tdi = {}
        accuracy_pruning_tdi = {}

        # Iterate on all possible query set sizes (from 1 to 10001)
        for query_set_size in range(1, 3):
            # For each query set size we repeat the experiment 1000 times
            for repetition in range(0, 2):
                per_query_winning_model = []
                list_ndcg_model_a = []
                list_ndcg_model_b = []

                if repetition == 50 or repetition == 100 or repetition == 500:
                    print('round ' + str(repetition) + ' for same query set size: ' + str(query_set_size))
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
                    interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b)

                    # Simulate clicks
                    interleaved_list = utils.simulate_clicks(interleaved_list, seed, realistic_model)

                    # Computing the per query winning model/ranker
                    per_query_winning_model.append(utils.compute_winning_model(interleaved_list, chosen_query_id))

                # Computing average ndcg to find winning model/ranker
                avg_ndcg_model_a = sum(list_ndcg_model_a) / len(list_ndcg_model_a)
                avg_ndcg_model_b = sum(list_ndcg_model_b) / len(list_ndcg_model_b)
                if avg_ndcg_model_a > avg_ndcg_model_b:
                    ndcg_winning_model = 'a'
                elif avg_ndcg_model_a < avg_ndcg_model_b:
                    ndcg_winning_model = 'b'
                else:
                    ndcg_winning_model = 't'

                # Pruning
                per_query_winning_model = pd.DataFrame.from_records(per_query_winning_model)
                per_query_winning_model.rename(
                    columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
                             3: 'winning_model'}, inplace=True)
                per_query_winning_model_pruned = utils.pruning(per_query_winning_model)

                # Computing standard ab_score
                ab_score = utils.computing_ab_score(per_query_winning_model)

                # Computing winning model for ab_score
                if ab_score > 0:
                    ab_score_winning_model = 'a'
                elif ab_score < 0:
                    ab_score_winning_model = 'b'
                else:
                    ab_score_winning_model = 't'

                # Check if ndcg agree with ab_score
                if ndcg_winning_model == ab_score_winning_model:
                    ranker_pair_agree.append(1)
                else:
                    ranker_pair_agree.append(0)

                # Computing pruning ab_score
                if not per_query_winning_model_pruned.empty:
                    ab_score_pruning = utils.computing_ab_score(per_query_winning_model_pruned)

                    # Computing winning model for pruning ab_score
                    if ab_score_pruning > 0:
                        ab_score_pruning_winning_model = 'a'
                    elif ab_score_pruning < 0:
                        ab_score_pruning_winning_model = 'b'
                    else:
                        ab_score_pruning_winning_model = 't'

                    # Check if ndcg agree with pruning ab_score
                    if ndcg_winning_model == ab_score_pruning_winning_model:
                        ranker_pair_pruning_agree.append(1)
                    else:
                        ranker_pair_pruning_agree.append(0)
                else:
                    print('The pruning removes all the queries\n')

            accuracy_standard_tdi[query_set_size] = sum(ranker_pair_agree) / len(ranker_pair_agree)
            if len(ranker_pair_pruning_agree) > 0:
                accuracy_pruning_tdi[query_set_size] = sum(ranker_pair_pruning_agree) / len(ranker_pair_pruning_agree)
            else:
                print('Pruning removes all queries for all pairs\n')
        final_accuracy_standard_tdi[ranker_pair] = accuracy_standard_tdi
        final_accuracy_pruning_tdi[ranker_pair] = accuracy_pruning_tdi

    print('Accuracy of tdi for all sizes and rankers:')
    print(final_accuracy_standard_tdi)
    print('Accuracy of pruning tdi for all sizes and rankers:')
    print(final_accuracy_pruning_tdi)
