import unittest
import pandas as pd
from our_implementation.utils import elaborate_dataset_for_score, generate_new_data, statistical_significance_computation, pruning, \
    computing_ab_score, same_score


class TestUtils(unittest.TestCase):

    def test_elaborate_dataset_for_score(self):
        input_data_frame = pd.DataFrame({'userId': [1, 2, 1, 1, 1, 2, 2, 2, 3],
                                         'click_per_userId': [1, 3, 1, 2, 1, 3, 2, 1, 2],
                                         'queryId': [5968, 5968, 5971, 5989, 5992, 5971, 5989, 5992, 5992],
                                         'click_per_query': [4, 4, 4, 4, 4, 4, 4, 4, 4],
                                         'click_per_model_A': [0, 2, 1, 0, 1, 3, 1, 1, 1],
                                         })
        expected_data_frame = pd.DataFrame({'queryId': [5968, 5971, 5989, 5992],
                                            'click_per_query': [4, 4, 4, 4],
                                            'click_per_winning_model': [2, 4, 3, 3],
                                            'winning_model': [2, 0, 1, 0],
                                            })

        result = elaborate_dataset_for_score(input_data_frame).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_generate_new_data(self):
        input_data_frame = pd.DataFrame({'queryId': [5971, 5989, 5992],
                                         'new_interactions_to_add': [5, 4, 3],
                                         'click_per_query': [1, 2, 3],
                                         })
        expected_data_frame_a = pd.DataFrame({'userId': [1, 1, 1, 2],
                                              'click_per_userId': [5, 4, 1, 2],
                                              'queryId': [5971, 5989, 5992, 5992],
                                              'click_per_query': [6, 6, 6, 6],
                                              'click_per_model_A': [3, 3, 1, 1],
                                              })
        expected_data_frame_b = pd.DataFrame({'userId': [1, 1, 1, 2],
                                              'click_per_userId': [5, 4, 1, 2],
                                              'queryId': [5971, 5989, 5992, 5992],
                                              'click_per_query': [6, 6, 6, 6],
                                              'click_per_model_A': [3, 3, 0, 1],
                                              })

        result_a = generate_new_data(input_data_frame, 6, 0.0566, 1, 15, 0)
        result_b = generate_new_data(input_data_frame, 6, 0, 0.943, 15, 0)

        pd.testing.assert_frame_equal(result_a, expected_data_frame_a)
        pd.testing.assert_frame_equal(result_b, expected_data_frame_b)

    def test_statistical_significance_computation(self):
        input_data_frame = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [4, 4, 4, 4],
             'click_per_winning_model': [2, 4, 3, 3],
             'winning_model': [2, 0, 1, 0],
             })
        expected_data_frame = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [4, 4, 4, 4],
             'click_per_winning_model': [2, 4, 3, 3],
             'winning_model': [2, 0, 1, 0],
             'statistical_significance': [1.04960, 0.05120, 0.35840, 0.35840],
             })

        result_data_frame = statistical_significance_computation(input_data_frame, 0.4)
        pd.testing.assert_frame_equal(result_data_frame, expected_data_frame)

    def test_pruning(self):
        input_data_frame = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [10, 10, 10, 10],
             'click_per_winning_model': [1, 9, 3, 3],
             'winning_model': [2, 0, 1, 0],
             })
        expected_data_frame = pd.DataFrame(
            {'queryId': [5968, 5971],
             'click_per_query': [10, 10],
             'click_per_winning_model': [1, 9],
             'winning_model': [2, 0],
             })

        result = pruning(input_data_frame, list([0.8]))
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_computing_ab_score(self):
        input_data_frame = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [10, 10, 10, 10],
             'click_per_winning_model': [1, 9, 3, 3],
             'winning_model': [2, 0, 1, 0],
             })
        expected_result = 0.125

        input_data_frame_2 = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [10, 10, 10, 10],
             'click_per_winning_model': [1, 9, 3, 3],
             'winning_model': [2, 0, 1, 1],
             })
        expected_result_2 = -0.125

        input_data_frame_3 = pd.DataFrame(
            {'queryId': [5968, 5971, 5989, 5992],
             'click_per_query': [10, 10, 10, 10],
             'click_per_winning_model': [1, 9, 3, 3],
             'winning_model': [2, 0, 1, 2],
             })
        expected_result_3 = 0.00

        result = computing_ab_score(input_data_frame)
        result_2 = computing_ab_score(input_data_frame_2)
        result_3 = computing_ab_score(input_data_frame_3)

        self.assertEqual(result, expected_result)
        self.assertEqual(result_2, expected_result_2)
        self.assertEqual(result_3, expected_result_3)

    def test_same_score(self):
        ab_first_same = 0.43
        ab_second_same = 0.12
        expected_result = True

        ab_first_different = 0.43
        ab_second_different = -0.12
        expected_result_different = False

        ab_first_different_2 = 0.43
        ab_second_different_2 = 0.00
        expected_result_different_2 = False

        ab_first_zero = 0.00
        ab_second_zero = 0.00
        expected_result_zero = True

        result_same = same_score(ab_first_same, ab_second_same)
        result_different = same_score(ab_first_different, ab_second_different)
        result_different_2 = same_score(ab_first_different_2, ab_second_different_2)
        result_zero = same_score(ab_first_zero, ab_second_zero)

        self.assertEqual(result_same, expected_result)
        self.assertEqual(result_different, expected_result_different)
        self.assertEqual(result_different_2, expected_result_different_2)
        self.assertEqual(result_zero, expected_result_zero)
