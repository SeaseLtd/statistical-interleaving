import utils
import numpy as np


def start_experiment(dataset_path, seed, not_uniform_probability=False):
    dataset = utils.load_dataframe(dataset_path)

    # Iterate on all possible pairs of rankers
    for i in range(1, 137):
        for j in range(i + 1, 137):
            np.random.seed(seed)
            chosen_queryId = np.random.choice(dataset.queryId.unique(), 1)
            
