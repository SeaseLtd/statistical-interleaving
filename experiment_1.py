import utils
import pandas as pd
from datetime import datetime


def start_experiment(dataset_path, seed, experiment_one_bis=False):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Experiment started at:", current_time)
    print()

    dataset = utils.load_dataframe(dataset_path)
    ranker_pair_agree = []
    ranker_pair_pruning_agree = []

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = dataset.queryId.unique()[:1000]
    else:
        set_of_queries = utils.generate_set_with_search_demand_curve(dataset)

    # Iterate on all possible pairs of rankers/models (from 1 to 137)
    for ranker_a in range(1, 137):
        for ranker_b in range(ranker_a + 1, 137):
            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')
            all_queries_winning_model = []
            list_ndcg_model_a = []
            list_ndcg_model_b = []

            # Iterate on all 1000 queries (from 0 to 1000)
            for query_index in range(0, 1000):
                if query_index == 50 or query_index == 100 or query_index == 500:
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
                interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b, seed)

                # Simulate clicks
                interleaved_list = utils.simulate_clicks(interleaved_list, seed)

                # Computing the per query winning model/ranker
                all_queries_winning_model.append(utils.compute_winning_model(interleaved_list, chosen_query_id))

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
            all_queries_winning_model = pd.DataFrame.from_records(all_queries_winning_model)
            all_queries_winning_model.rename(columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
                                                      3: 'winning_model'}, inplace=True)
            all_queries_winning_model_pruned = utils.pruning(all_queries_winning_model)

            # Computing standard ab_score
            ab_score = utils.computing_ab_score(all_queries_winning_model)

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
            if not all_queries_winning_model_pruned.empty:
                ab_score_pruning = utils.computing_ab_score(all_queries_winning_model_pruned)

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

    accuracy_standard_tdi = sum(ranker_pair_agree) / len(ranker_pair_agree)
    print('Accuracy of tdi on all pairs of rankers:')
    print(accuracy_standard_tdi)

    if len(ranker_pair_pruning_agree) > 0:
        accuracy_pruning_tdi = sum(ranker_pair_pruning_agree) / len(
            ranker_pair_pruning_agree)
        print('Accuracy of pruning tdi on all pairs of rankers:')
        print(accuracy_pruning_tdi)
    else:
        print('Pruning removes all queries for all pairs')
