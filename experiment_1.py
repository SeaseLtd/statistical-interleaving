import utils


def start_experiment(dataset_path, seed):
    dataset = utils.load_dataframe(dataset_path)
    list_ndcg_model_a = []
    list_ndcg_model_b = []
    per_query_winning_model = []

    # Fixed subset of 1000 queries
    set_of_queries = dataset.queryId.unique()[:1000]

    # Iterate on all possible pairs of rankers
    for i in range(1, 137):
        for j in range(i + 1, 137):
            for k in range(0, 1000):
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

                # Computing the per query winning model
                per_query_winning_model.append(utils.compute_winning_model())



            # Pruning
            interleaved_list_pruned = utils.pruning(interleaved_list)

            # Computing ab_score
            ab_score = utils.computing_ab_score(interleaved_list)
            ab_score_pruning = utils.computing_ab_score(interleaved_list_pruned)
            print()
