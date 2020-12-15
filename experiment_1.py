import utils
import pandas as pd


def start_experiment(dataset_path, seed, experiment_one_bis=False):
    dataset = utils.load_dataframe(dataset_path)
    list_ndcg_model_a = []
    list_ndcg_model_b = []
    per_query_winning_model = []
    ranker_pair_agree = []
    ranker_pair_pruning_agree = []

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = dataset.queryId.unique()[:1000]
    else:
        set_of_queries = utils.generate_set_with_search_demand_curve(dataset)

    # Iterate on all possible pairs of rankers/models (from 1 to 137)
    for i in range(1, 137):
        for j in range(i + 1, 137):
            print('-------- Pair of rankers: (' + str(i) + ', ' + str(j) + ') --------')
            for k in range(0, 1000):
                if k == 50 or k == 100 or k == 500:
                    print('round ' + str(k) + ' for same pair of rankers')
                chosen_query_id = set_of_queries[k]

                # Reduce the dataset to the documents for the selected query
                query_selected_documents = dataset[dataset['queryId'] == chosen_query_id]

                # Creating the models' ranked lists
                ranked_list_model_a = query_selected_documents.sort_values(by=[i], ascending=False)
                ranked_list_model_b = query_selected_documents.sort_values(by=[j], ascending=False)

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
                ranker_pair_pruning_agree.append(0)

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
                if ndcg_winning_model == ab_score_winning_model:
                    ranker_pair_pruning_agree.append(1)
                else:
                    ranker_pair_pruning_agree.append(0)
            else:
                print('The pruning removes all the queries')
            print('\n')

    accuracy_standard_tdi = sum(ranker_pair_agree) / len(ranker_pair_agree)
    print('Accuracy for standard tdi: ' + str(accuracy_standard_tdi))
    if len(ranker_pair_pruning_agree) > 0:
        accuracy_pruning_tdi = sum(ranker_pair_pruning_agree) / len(ranker_pair_pruning_agree)
        print('Accuracy for pruning tdi: ' + str(accuracy_pruning_tdi))
    else:
        print('Pruning removes all queries for all pairs')
