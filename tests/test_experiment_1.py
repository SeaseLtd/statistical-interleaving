import os
import numpy as np
import pandas as pd
import experiment_1 as ex
from unittest import TestCase
from pandas.util.testing import assert_frame_equal, assert_numpy_array_equal

EXEC_DIR = os.path.dirname(os.path.abspath(__file__))


class Experiment1Test(TestCase):
    def test_interleave_iteration(self):
        np.random.seed(10)

        # Input dataframe
        input_array = np.array([
            [1, 2, 1, 1],
            [1, 2, 2, 1],
            [1, 3, 1, 1],
            [1, 3, 2, 1],
            [2, 3, 1, 1],
            [2, 3, 2, 1]])

        # Ranked list cache
        ranked_list_cache = dict({
            '1_1': [np.array([2, 0, 1]), np.array([2, 2, 3])],
            '1_2': [np.array([3, 4, 2]), np.array([0, 1, 0])],
            '2_1': [np.array([2, 2, 0]), np.array([1, 2, 3])],
            '2_2': [np.array([4, 3, 1]), np.array([1, 0, 3])],
            '3_1': [np.array([1, 2, 0]), np.array([3, 2, 2])],
            '3_2': [np.array([3, 4, 0]), np.array([1, 1, 1])]})

        returned_interleaved_ranked_lists = ex.interleave_iteration(input_array, ranked_list_cache)

        # Expected list
        expected_list = [
            np.array([[1, 2, 3], [2, 0, 0]], dtype='uint8'),
            np.array([[1, 0, 3], [1, 0, 1]], dtype='uint8'),
            np.array([[3, 2, 2], [1, 0, 1]], dtype='uint8'),
            np.array([[1, 1, 1], [2, 2, 1]], dtype='uint8'),
            np.array([[3, 1, 3], [1, 0, 2]], dtype='uint8'),
            np.array([[1, 1, 3], [1, 0, 0]], dtype='uint8')]

        # Asserting
        for i in range(0, len(returned_interleaved_ranked_lists)):
            assert_frame_equal(pd.DataFrame(returned_interleaved_ranked_lists[i]), pd.DataFrame(expected_list[i]))

    def test_clicks_generation_iteration(self):
        np.random.seed(10)

        # Interleaved ranked list
        interleaved_ranked_list = [
            np.array([[1, 2, 3], [2, 0, 0]], dtype='uint8'),
            np.array([[1, 0, 3], [1, 0, 1]], dtype='uint8'),
            np.array([[3, 2, 2], [1, 0, 1]], dtype='uint8'),
            np.array([[1, 1, 1], [2, 2, 1]], dtype='uint8'),
            np.array([[3, 1, 3], [1, 0, 2]], dtype='uint8'),
            np.array([[1, 1, 3], [1, 0, 0]], dtype='uint8')]

        # Expected clicks result lists
        expected_clicks_results_lists = [
            [np.array([2, 0, 0], dtype='uint8'), np.array([0, 0, 1], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([0, 0, 1], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([1, 0, 1], dtype='uint8')],
            [np.array([2, 2, 1], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 2], dtype='uint8'), np.array([1, 0, 1], dtype='uint8')],
            [np.array([1, 0, 0], dtype='uint8'), np.array([1, 0, 1], dtype='uint8')]]

        returned_clicks_results_lists = ex.clicks_generation_iteration(interleaved_ranked_list,
                                                                       click_generation_top_k=0,
                                                                       click_generation_realistic=False)

        # Asserting
        for i in range(0, len(returned_clicks_results_lists)):
            assert_frame_equal(pd.DataFrame(returned_clicks_results_lists[i]),
                               pd.DataFrame(expected_clicks_results_lists[i], dtype='uint8'))

        # ---------------------------------------- REALISTIC CLICK GENERATION ------------------------------

        # Interleaved ranked list
        interleaved_ranked_list = [
            np.array([[1, 2, 3], [2, 0, 0]], dtype='uint8'),
            np.array([[1, 0, 3], [1, 0, 1]], dtype='uint8'),
            np.array([[3, 2, 2], [1, 0, 1]], dtype='uint8'),
            np.array([[1, 1, 1], [2, 2, 1]], dtype='uint8'),
            np.array([[3, 1, 3], [1, 0, 2]], dtype='uint8'),
            np.array([[1, 1, 3], [1, 0, 0]], dtype='uint8')]

        # Expected clicks result lists
        expected_clicks_results_lists = [
            [np.array([2, 0, 0], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([0, 1, 0], dtype='uint8')],
            [np.array([2, 2, 1], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 2], dtype='uint8'), np.array([1, 0, 0], dtype='uint8')],
            [np.array([1, 0, 0], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')]]

        returned_clicks_results_lists = ex.clicks_generation_iteration(interleaved_ranked_list,
                                                                       click_generation_top_k=0,
                                                                       click_generation_realistic=True)

        # Asserting
        for i in range(0, len(returned_clicks_results_lists)):
            assert_frame_equal(pd.DataFrame(returned_clicks_results_lists[i]),
                               pd.DataFrame(expected_clicks_results_lists[i], dtype='uint8'))

    def test_aggregate_interleaving_clicks_per_model_iteration(self):
        np.random.seed(10)

        # Interleaved ranked list clicks
        interleaved_ranked_lists_clicks = [
            [np.array([2, 0, 0], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 1], dtype='uint8'), np.array([0, 1, 0], dtype='uint8')],
            [np.array([2, 2, 1], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')],
            [np.array([1, 0, 2], dtype='uint8'), np.array([1, 0, 0], dtype='uint8')],
            [np.array([1, 0, 0], dtype='uint8'), np.array([0, 0, 0], dtype='uint8')]]

        # Expected aggregated clicks column
        expected_aggregated_clicks_column = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0]], dtype='uint16')

        returned_aggregated_clicks_column = ex.aggregate_interleaving_clicks_per_model_iteration(
            interleaved_ranked_lists_clicks)

        # Asserting
        assert_numpy_array_equal(returned_aggregated_clicks_column, expected_aggregated_clicks_column)
