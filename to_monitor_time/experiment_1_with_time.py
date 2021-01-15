import utils_with_time
import pandas as pd
import time
from datetime import datetime


def start_experiment(dataset_path, seed, monitoring_time=False, experiment_one_bis=False):
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    if monitoring_time:
        avg_times_per_step = []
        start_load = time.time()
        print("Loading the dataframe")
    dataset = utils_with_time.load_dataframe(dataset_path)
    if monitoring_time:
        end_load = time.time()

    ranker_pair_agree = []
    ranker_pair_pruning_agree = []

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = dataset.queryId.unique()[:1000]
    else:
        if monitoring_time:
            start_query = datetime.now()
            print("Selecting the queries")
        set_of_queries = utils_with_time.generate_set_with_search_demand_curve(dataset)
        if monitoring_time:
            end_query = datetime.now()
            print("Queries selected in:", start_query - end_query)

    # Iterate on all possible pairs of rankers/models (from 1 to 137)
    for ranker_a in range(1, 137):
        for ranker_b in range(ranker_a + 1, 137):
            start_each_pair = datetime.now()

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
                if monitoring_time:
                    start_documents = datetime.now()
                    print("Selecting the documents")
                query_selected_documents = dataset[dataset['queryId'] == chosen_query_id]
                if monitoring_time:
                    end_documents = datetime.now()
                    print("Documents selected in:", start_documents - end_documents)

                # Creating the models' ranked lists
                if monitoring_time:
                    start_rank_a = datetime.now()
                    print("Creating ranked list A")
                ranked_list_model_a = query_selected_documents.sort_values(by=[ranker_a], ascending=False)
                if monitoring_time:
                    end_rank_a = datetime.now()
                    print("Ranked list A created in:", start_rank_a - end_rank_a)
                    start_rank_b = datetime.now()
                    print("Creating ranked list B")
                ranked_list_model_b = query_selected_documents.sort_values(by=[ranker_b], ascending=False)
                if monitoring_time:
                    end_rank_b = datetime.now()
                    print("Ranked list B created in:", start_rank_b - end_rank_b)

                # Computing ndcg
                if monitoring_time:
                    start_ndcg_a = datetime.now()
                    print("Computing ndcg A")
                list_ndcg_model_a.append(utils_with_time.compute_ndcg(ranked_list_model_a))
                if monitoring_time:
                    end_ndcg_a = datetime.now()
                    print("Ndcg A computed in:", start_ndcg_a - end_ndcg_a)
                    start_ndcg_b = datetime.now()
                    print("Computing ndcg B")
                list_ndcg_model_b.append(utils_with_time.compute_ndcg(ranked_list_model_b))
                if monitoring_time:
                    end_ndcg_b = datetime.now()
                    print("Ndcg B computed in:", start_ndcg_b - end_ndcg_b)

                # Creating interleaved list
                if monitoring_time:
                    start_interleaving = datetime.now()
                    print("Computing interleaved list")
                interleaved_list = utils_with_time.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b, seed)
                if monitoring_time:
                    end_interleaving = datetime.now()
                    print("Interleaving list computed in:", start_interleaving - end_interleaving)

                # Simulate clicks
                if monitoring_time:
                    start_clicks = datetime.now()
                    print("Computing simulated clicks")
                interleaved_list = utils_with_time.simulate_clicks(interleaved_list, seed)
                if monitoring_time:
                    end_clicks = datetime.now()
                    print("Simulated clicks computed in:", start_clicks - end_clicks)

                # Computing the per query winning model/ranker
                if monitoring_time:
                    start_winning = datetime.now()
                    print("Computing per query winning model")
                all_queries_winning_model.append(utils_with_time.compute_winning_model(interleaved_list, chosen_query_id))
                if monitoring_time:
                    end_winning = datetime.now()
                    print("Per query winning model computed in:", start_winning - end_winning)

            # Computing average ndcg to find winning model/ranker
            if monitoring_time:
                start_winning_ncdg = datetime.now()
                print("Computing ndcg winning model")
            ndcg_winning_model = utils_with_time.compute_ndcg_winning_model(list_ndcg_model_a, list_ndcg_model_b)
            if monitoring_time:
                end_winning_ndcg = datetime.now()
                print("Ndcg winning model computed in:", start_winning_ncdg - end_winning_ndcg)

            # Pruning
            if monitoring_time:
                start_pruning = datetime.now()
                print("Pruning")
            all_queries_winning_model = pd.DataFrame.from_records(all_queries_winning_model)
            all_queries_winning_model.rename(columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
                                                      3: 'winning_model'}, inplace=True)
            all_queries_winning_model_pruned = utils_with_time.pruning(all_queries_winning_model)
            if monitoring_time:
                end_pruning = datetime.now()
                print("Ndcg winning model computed in:", start_pruning - end_pruning)

            # Computing standard ab_score
            if monitoring_time:
                start_ab = datetime.now()
                print("Computing AB score")
            ab_score_winning_model = utils_with_time.computing_winning_model_ab_score(all_queries_winning_model)
            if monitoring_time:
                end_ab = datetime.now()
                print("AB score computed in:", start_ab - end_ab)

            # Check if ndcg agree with ab_score
            if ndcg_winning_model == ab_score_winning_model:
                ranker_pair_agree.append(1)
            else:
                ranker_pair_agree.append(0)

            # Computing pruning ab_score
            if monitoring_time:
                start_pruning_ab = datetime.now()
                print("Computing pruning AB score")
            if not all_queries_winning_model_pruned.empty:
                ab_score_pruning_winning_model = utils_with_time.computing_winning_model_ab_score(
                    all_queries_winning_model_pruned)
                if monitoring_time:
                    end_pruning_ab = datetime.now()
                    print("AB score computed in:", start_pruning_ab - end_pruning_ab)

                # Check if ndcg agree with pruning ab_score
                if ndcg_winning_model == ab_score_pruning_winning_model:
                    ranker_pair_pruning_agree.append(1)
                else:
                    ranker_pair_pruning_agree.append(0)
            else:
                ranker_pair_pruning_agree.append(0)
                print('The pruning removes all the queries\n')

            end_each_pair = datetime.now()
            print('Execution time each pair: ' + str(end_each_pair - start_each_pair) + '\n')

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

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))
