import unittest
import pandas as pd
from our_implementation.dataset_creation import create_adore_dataset, create_primary_dataset, create_variation_dataset


class TestCreateDataset(unittest.TestCase):

    def test_create_adore_dataset(self):
        dataset_path = 'resources/query_click_user.json'

        expected_data_frame_a = pd.DataFrame({'userId': [1, 2, 1, 1, 1],
                                              'click_per_userId': [1, 3, 1, 2, 1],
                                              'queryId': [5968, 5968, 5971, 5989, 5992],
                                              'click_per_query': [4, 4, 1, 2, 1],
                                              'click_per_model_A': [0, 3, 1, 0, 1],
                                              })
        expected_data_frame_b = pd.DataFrame({'userId': [1, 2, 1, 1, 1],
                                              'click_per_userId': [1, 3, 1, 2, 1],
                                              'queryId': [5968, 5968, 5971, 5989, 5992],
                                              'click_per_query': [4, 4, 1, 2, 1],
                                              'click_per_model_A': [0, 0, 0, 1, 0],
                                              })

        result_a = create_adore_dataset(dataset_path, 0.0566, 1, 0, False)
        result_b = create_adore_dataset(dataset_path, 0, 0.943, 0, False)

        pd.testing.assert_frame_equal(result_a, expected_data_frame_a)
        pd.testing.assert_frame_equal(result_b, expected_data_frame_b)

    def test_create_primary_dataset(self):
        input_data_frame = pd.DataFrame({'userId': [1, 2, 1, 1, 1],
                                         'click_per_userId': [1, 3, 1, 2, 1],
                                         'queryId': [5968, 5968, 5971, 5989, 5992],
                                         'click_per_query': [4, 4, 1, 2, 1],
                                         'click_per_model_A': [0, 3, 1, 0, 1],
                                         })
        expected_data_frame_a = pd.DataFrame({'userId': [1, 2, 1, 1, 1, 2, 2, 2, 3],
                                              'click_per_userId': [1, 3, 1, 2, 1, 3, 2, 1, 2],
                                              'queryId': [5968, 5968, 5971, 5989, 5992, 5971, 5989, 5992, 5992],
                                              'click_per_query': [4, 4, 4, 4, 4, 4, 4, 4, 4],
                                              'click_per_model_A': [0, 3, 1, 0, 1, 3, 1, 1, 1],
                                              })
        expected_data_frame_b = pd.DataFrame({'userId': [1, 2, 1, 1, 1, 2, 2, 2, 3],
                                              'click_per_userId': [1, 3, 1, 2, 1, 3, 2, 1, 2],
                                              'queryId': [5968, 5968, 5971, 5989, 5992, 5971, 5989, 5992, 5992],
                                              'click_per_query': [4, 4, 4, 4, 4, 4, 4, 4, 4],
                                              'click_per_model_A': [0, 3, 1, 0, 1, 1, 1, 0, 1],
                                              })

        result_a = create_primary_dataset(input_data_frame, 0.0566, 1, 15, 0)
        result_b = create_primary_dataset(input_data_frame, 0, 0.943, 15, 0)

        pd.testing.assert_frame_equal(result_a, expected_data_frame_a)
        pd.testing.assert_frame_equal(result_b, expected_data_frame_b)

    def test_create_variation_dataset(self):
        input_data_frame = pd.DataFrame({'userId': [1, 2, 1, 1, 1, 2, 2, 2, 3],
                                         'click_per_userId': [1, 3, 1, 2, 1, 3, 2, 1, 2],
                                         'queryId': [5968, 5968, 5971, 5989, 5992, 5971, 5989, 5992, 5992],
                                         'click_per_query': [4, 4, 4, 4, 4, 4, 4, 4, 4],
                                         'click_per_model_A': [0, 3, 1, 0, 1, 3, 1, 1, 1],
                                         })
        click_per_query = pd.DataFrame({'queryId': [5968, 5968, 5971, 5989, 5992],
                                        'click_per_query': [4, 4, 1, 2, 1],
                                        })
        click_per_query = click_per_query.drop_duplicates(subset=['queryId', 'click_per_query'], keep='last')[[
            'queryId', 'click_per_query']]
        click_per_query.set_index('queryId', inplace=True)
        expected_data_frame = pd.DataFrame({'userId': [2, 1, 1, 1, 3],
                                            'click_per_userId': [3, 1, 1, 2, 1],
                                            'queryId': [5968, 5968, 5971, 5989, 5992],
                                            'click_per_query': [4, 4, 1, 2, 1],
                                            'click_per_model_A': [3, 0, 1, 0, 0],
                                            })

        result = create_variation_dataset(input_data_frame, click_per_query, 0)

        pd.testing.assert_frame_equal(result, expected_data_frame)
