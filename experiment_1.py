import utils
import numpy as np


def start_experiment(dataset_path, seed, not_uniform_probability=False):
    dataset = utils.load_dataframe(dataset_path)

    # Iterate on all possible pairs of rankers
    for i in range(1, 137):
        for j in range(i + 1, 137):
            # Random selecting the query
            np.random.seed(seed)
            chosen_query_id = int(np.random.choice(dataset.queryId.unique(), 1))

            # Reduce the dataset to the documents for the selected query
            query_selected_documents = dataset[dataset['queryId'] == chosen_query_id]

            # Creating the models' ranked lists
            ranked_list_model_a = query_selected_documents.sort_values(by=[i], ascending=False)
            ranked_list_model_b = query_selected_documents.sort_values(by=[j], ascending=False)
            ndcg_model_a = utils.compute_ndcg(ranked_list_model_a)
            ndcg_model_b = utils.compute_ndcg(ranked_list_model_b)

            # Identify the winning model by ndcg
            if ndcg_model_a > ndcg_model_b:
                winning_model = 'a'
            elif ndcg_model_a < ndcg_model_b:
                winning_model = 'b'
            else:
                winning_model = 't'

            # Creating interleaved list
            interleaved_list = utils.execute_tdi_interleaving(ranked_list_model_a, ranked_list_model_b)

            # Simulate clicks

            # Pruning
            interleaved_list_pruned = utils.pruning(interleaved_list)

            # Computing ab_score
            ab_score = utils.computing_ab_score(interleaved_list)
            ab_score_pruning = utils.computing_ab_score(interleaved_list_pruned)
            print()
