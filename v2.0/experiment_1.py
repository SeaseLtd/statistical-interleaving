import utils
import pandas as pd
import time
from datetime import datetime
import pprint


def start_experiment(dataset_path, seed, query_set=1000, max_range_pair=137, experiment_one_bis=False):
    start_total = time.time()
    print("Experiment started at:", datetime.now().strftime("%H:%M:%S"))
    print()

    avg_times_per_step, total_repetitions_per_step = utils.initialize_dictionaries()

    # load dataframe
    start_step = time.time()
    dataset = utils.load_dataframe(dataset_path)
    end_step = time.time()
    avg_times_per_step['load'] = end_step - start_step
    total_repetitions_per_step['load'] += 1

    ranker_pair_agree = []
    ranker_pair_pruning_agree = []

    # Fixed subset of 1000 queries
    if not experiment_one_bis:
        set_of_queries = dataset.queryId.unique()[:query_set]
    else:
        start_step = time.time()
        set_of_queries = utils.generate_set_with_search_demand_curve(dataset)
        end_step = time.time()
        avg_times_per_step['query'] = end_step - start_step
        total_repetitions_per_step['query'] += 1

    # Precompute ranked lists and ndcg per ranker-query
    ranked_table, ndcg_ranked_table = utils.precompute_ranked_table(dataset, max_range_pair, set_of_queries)

    # Iterate on all possible pairs of rankers/models (from 1 to 137)
    for ranker_a in range(1, max_range_pair):
        for ranker_b in range(ranker_a + 1, max_range_pair):
            start_each_pair = datetime.now()

            print('-------- Pair of rankers: (' + str(ranker_a) + ', ' + str(ranker_b) + ') --------')
            all_queries_winning_model = []
            list_ndcg_model_a = []
            list_ndcg_model_b = []

            # Iterate on all 1000 queries (from 0 to 1000)
            for query_index in range(0, len(set_of_queries)):
                if query_index == 50 or query_index == 100 or query_index == 500:
                    print('round ' + str(query_index) + ' for same pair of rankers')
                chosen_query_id = set_of_queries[query_index]

                # Reduce the dataset to the documents for the selected query
                start_step = time.time()
                query_selected_documents = ranked_table[ranked_table['queryId'] == chosen_query_id]
                end_step = time.time()
                avg_times_per_step['select_documents'].append(end_step - start_step)
                total_repetitions_per_step['select_documents'] += 1

                # Selecting the models' ranked lists
                start_step = time.time()
                ranked_list_model_a = query_selected_documents.query("ranker == {}".format(ranker_a))
                end_step = time.time()
                avg_times_per_step['ranked_list_a'].append(end_step - start_step)
                total_repetitions_per_step['ranked_list_a'] += 1
                start_step = time.time()
                ranked_list_model_b = query_selected_documents.query("ranker == {}".format(ranker_b))
                end_step = time.time()
                avg_times_per_step['ranked_list_b'].append(end_step - start_step)
                total_repetitions_per_step['ranked_list_b'] += 1

                # Computing ndcg
                start_step = time.time()
                list_ndcg_model_a.append(ndcg_ranked_table[(ndcg_ranked_table['queryId'] == chosen_query_id) &
                                                           (ndcg_ranked_table['ranker'] == ranker_a)].ndcg)
                end_step = time.time()
                avg_times_per_step['ndcg_a'].append(end_step - start_step)
                total_repetitions_per_step['ndcg_a'] += 1
                start_step = time.time()
                list_ndcg_model_b.append(ndcg_ranked_table[(ndcg_ranked_table['queryId'] == chosen_query_id) &
                                                           (ndcg_ranked_table['ranker'] == ranker_b)].ndcg)
                end_step = time.time()
                avg_times_per_step['ndcg_b'].append(end_step - start_step)
                total_repetitions_per_step['ndcg_b'] += 1

                # Creating interleaved list
                start_step = time.time()
                interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b, seed)
                end_step = time.time()
                avg_times_per_step['interleaving'].append(end_step - start_step)
                total_repetitions_per_step['interleaving'] += 1

                # Simulate clicks
                start_step = time.time()
                interleaved_list = utils.simulate_clicks(interleaved_list, seed)
                end_step = time.time()
                avg_times_per_step['clicks'].append(end_step - start_step)
                total_repetitions_per_step['clicks'] += 1

                # Computing the per query winning model/ranker
                start_step = time.time()
                all_queries_winning_model.append(utils.compute_winning_model(interleaved_list, chosen_query_id))
                end_step = time.time()
                avg_times_per_step['query_winning_model'].append(end_step - start_step)
                total_repetitions_per_step['query_winning_model'] += 1

            # Computing average ndcg to find winning model/ranker
            start_step = time.time()
            ndcg_winning_model = utils.compute_ndcg_winning_model(list_ndcg_model_a, list_ndcg_model_b)
            end_step = time.time()
            avg_times_per_step['ndcg_winning_model'].append(end_step - start_step)
            total_repetitions_per_step['ndcg_winning_model'] += 1

            # Pruning
            start_step = time.time()
            all_queries_winning_model = pd.DataFrame.from_records(all_queries_winning_model)
            all_queries_winning_model.rename(columns={0: 'queryId', 1: 'click_per_winning_model', 2: 'click_per_query',
                                                      3: 'winning_model'}, inplace=True)
            all_queries_winning_model_pruned = utils.pruning(all_queries_winning_model)
            end_step = time.time()
            avg_times_per_step['pruning'].append(end_step - start_step)
            total_repetitions_per_step['pruning'] += 1

            # Computing standard ab_score
            start_step = time.time()
            ab_score_winning_model = utils.computing_winning_model_ab_score(all_queries_winning_model)
            end_step = time.time()
            avg_times_per_step['ab_score'].append(end_step - start_step)
            total_repetitions_per_step['ab_score'] += 1

            # Check if ndcg agree with ab_score
            if ndcg_winning_model == ab_score_winning_model:
                ranker_pair_agree.append(1)
            else:
                ranker_pair_agree.append(0)

            # Computing pruning ab_score
            start_step = time.time()
            if not all_queries_winning_model_pruned.empty:
                ab_score_pruning_winning_model = utils.computing_winning_model_ab_score(
                    all_queries_winning_model_pruned)
                end_step = time.time()
                avg_times_per_step['pruning_ab_score'].append(end_step - start_step)
                total_repetitions_per_step['pruning_ab_score'] += 1

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

    # Computing avg times per step
    for key in avg_times_per_step:
        if not isinstance(avg_times_per_step[key], float):
            avg_times_per_step[key] = sum(avg_times_per_step[key])
    for key, value in list(total_repetitions_per_step.items()):
        if value == 0.0:
            del total_repetitions_per_step[key]
            del avg_times_per_step[key]
    for key, dividend in avg_times_per_step.items():
        avg_times_per_step[key] = dividend / total_repetitions_per_step.get(key, 1)
    print('\nAverage times per step:')
    pprint.pprint(avg_times_per_step)

    end_total = time.time()
    print("\nExperiment ended at:", datetime.now().strftime("%H:%M:%S"))
    print('Total experiment time: ' + str(end_total - start_total))
