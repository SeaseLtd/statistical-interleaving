import utils
import pandas as pd


def start_experiment(dataset_path, seed, experiment_one_bis=False):
    dataset = utils.load_dataframe(dataset_path)
    ranker_pair_agree = []
    ranker_pair_pruning_agree = []
    accuracy_standard_tdi = {}
    accuracy_pruning_tdi = {}

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = dataset.queryId.unique()[:1000]
    else:
        set_of_queries = utils.generate_set_with_search_demand_curve(dataset)

    # Iterate on all possible pairs of rankers/models (from 0 to 136)
    for ranker_a in range(0, 3):
        for ranker_b in range(ranker_a + 1, 3):
            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')
            per_query_winning_model = []
            list_ndcg_model_a = []
            list_ndcg_model_b = []

            for query_index in range(0, 2):
                if query_index == 50 or query_index == 100 or query_index == 500:
                    print('round ' + str(query_index) + ' for same pair of rankers')
                print('round ' + str(query_index) + ' for same pair of rankers')
                chosen_query_id = set_of_queries[query_index]

                # Reduce the dataset to the documents for the selected query
                query_selected_documents = dataset[dataset['queryId'] == chosen_query_id]

                # Creating the models' ranked lists
                ranked_list_model_a = query_selected_documents.sort_values(by=[ranker_a], ascending=False)
                ranked_list_model_b = query_selected_documents.sort_values(by=[ranker_b], ascending=False)

                # Computing ndcg
                list_ndcg_model_a.append(utils.compute_ndcg(ranked_list_model_a))
                list_ndcg_model_b.append(utils.compute_ndcg(ranked_list_model_b))

                # Creating interleaved list
                interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b)

                # Simulate clicks
                interleaved_list = utils.simulate_clicks(interleaved_list, seed)

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
            per_query_winning_model.rename(columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
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
                print('The pruning removes all the queries')
            print('\n')

            accuracy_standard_tdi[(ranker_a, ranker_b)] = sum(ranker_pair_agree) / len(ranker_pair_agree)
            if len(ranker_pair_pruning_agree) > 0:
                accuracy_pruning_tdi[(ranker_a, ranker_b)] = sum(ranker_pair_pruning_agree) / len(
                    ranker_pair_pruning_agree)
            else:
                print('Pruning removes all queries for all pairs\n')

    print('Accuracy of tdi for all pairs of rankers:')
    print(accuracy_standard_tdi)
    print('Accuracy of pruning tdi for all pairs of rankers:')
    print(accuracy_pruning_tdi)
