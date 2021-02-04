import unittest
import pandas as pd
from v2_0 import utils


class TestUtils(unittest.TestCase):

    def test_load_dataframe(self):
        input_data_frame_path = '../resources/train.txt'
        expected_data_frame = pd.DataFrame({1: pd.Series([3.00, 3.00, 3.00, 3.00, 3.00, 3.00], dtype='float32'),
                                            2: pd.Series([3.00, 0.00, 0.11, 0.00, 0.00, 0.00], dtype='float32'),
                                            3: pd.Series([0.00, 3.00, 2.00, 3.00, 3.00, 3.00], dtype='float32'),
                                            4: pd.Series([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype='float32'),
                                            5: pd.Series([3.45, 3.00, 3.00, 3.00, 3.00, 3.00], dtype='float32'),
                                            'relevance': pd.Series([2.00, 2.00, 0.00, 2.00, 1.00, 1.00],
                                                                   dtype='float32'),
                                            'queryId': pd.Series([1, 1, 1, 1, 1, 1], dtype='int32')
                                            })

        result = utils.load_dataframe(input_data_frame_path)
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_compute_ndcg(self):
        # Ordered by feature 5
        input_data_frame = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                         2: [3.00, 0.00, 0.11, 0.00, 0.00, 0.00],
                                         3: [0.00, 3.00, 2.00, 3.00, 3.00, 3.00],
                                         4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                         5: [3.45, 3.00, 3.00, 3.00, 3.00, 3.00],
                                         'relevance': [2.00, 2.00, 0.00, 2.00, 1.00, 1.00],
                                         'queryId': [1, 2, 2, 2, 2, 1]
                                         })
        expected_ndcg = 0.961

        result = utils.compute_ndcg(input_data_frame)
        self.assertEqual(result, expected_ndcg)

    def test_execute_tdi_interleaving(self):
        # Ordered by feature 5
        input_data_frame_a = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           2: [3.00, 0.00, 0.11, 0.00, 0.00, 0.00],
                                           3: [0.00, 3.00, 2.00, 3.00, 3.00, 3.00],
                                           4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                           5: [3.45, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           'relevance': [2.00, 2.00, 2.00, 1.00, 1.00, 0.00],
                                           'queryId': [2, 2, 2, 2, 2, 2],
                                           'ranker': [1, 1, 1, 1, 1, 1],
                                           'query_doc_id': [0, 1, 2, 3, 4, 5]
                                           })
        input_data_frame_a.set_index(['ranker', 'query_doc_id'], inplace=True, verify_integrity=True)
        # Ordered by feature 3
        input_data_frame_b = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           2: [0.00, 0.00, 0.00, 0.00, 0.11, 3.00],
                                           3: [3.00, 3.00, 3.00, 3.00, 2.00, 0.00],
                                           4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                           5: [3.00, 3.00, 3.00, 3.00, 3.00, 3.45],
                                           'relevance': [2.00, 1.00, 1.00, 0.00, 2.00, 2.00],
                                           'queryId': [2, 2, 2, 2, 2, 2],
                                           'ranker': [2, 2, 2, 2, 2, 2],
                                           'query_doc_id': [1, 3, 4, 5, 2, 0]
                                           })
        input_data_frame_b.set_index(['ranker', 'query_doc_id'], inplace=True, verify_integrity=True)
        expected_data_frame = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                            2: [0.00, 3.00, 0.11, 0.00, 0.00, 0.00],
                                            3: [3.00, 0.00, 2.00, 3.00, 3.00, 3.00],
                                            4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                            5: [3.00, 3.45, 3.00, 3.00, 3.00, 3.00],
                                            'relevance': [2.00, 2.00, 2.00, 1.00, 1.00, 0.00],
                                            'queryId': [2, 2, 2, 2, 2, 2],
                                            'model': ['b', 'a', 'a', 'b', 'a', 'b']
                                            })
        expected_data_frame.set_index([pd.Int64Index([1, 0, 2, 3, 4, 5], dtype='int64', name='query_doc_id')],
                                      inplace=True)

        result = utils.execute_tdi_interleaving(input_data_frame_a.loc[1], input_data_frame_b.loc[2], 0)
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_simulate_clicks(self):
        input_data_frame = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                         2: [0.00, 3.00, 0.11, 0.00, 0.00, 0.00],
                                         3: [3.00, 0.00, 2.00, 3.00, 3.00, 3.00],
                                         4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                         5: [3.00, 3.45, 3.00, 3.00, 3.00, 3.00],
                                         'relevance': [2.00, 3.00, 4.00, 1.00, 1.00, 0.00],
                                         'queryId': [2, 2, 2, 1, 3, 1],
                                         'model': ['b', 'a', 'a', 'b', 'a', 'b']
                                         })
        expected_data_frame = pd.DataFrame({1: [3.00, 3.00, 3.00],
                                            2: [0.00, 3.00, 0.11],
                                            3: [3.00, 0.00, 2.00],
                                            4: [0.00, 0.00, 0.00],
                                            5: [3.00, 3.45, 3.00],
                                            'relevance': [2.00, 3.00, 4.00],
                                            'queryId': [2, 2, 2],
                                            'model': ['b', 'a', 'a']
                                            })

        result = utils.simulate_clicks(input_data_frame, 0)
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_compute_winning_model(self):
        input_data_frame_a = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           2: [0.00, 3.00, 0.11, 0.00, 0.00, 0.00],
                                           3: [3.00, 0.00, 2.00, 3.00, 3.00, 3.00],
                                           4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                           5: [3.00, 3.45, 3.00, 3.00, 3.00, 3.00],
                                           'relevance': [2.00, 3.00, 4.00, 1.00, 1.00, 0.00],
                                           'queryId': [2, 2, 2, 2, 2, 2],
                                           'model': ['b', 'a', 'a', 'b', 'a', 'a']
                                           })
        input_data_frame_b = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           2: [0.00, 3.00, 0.11, 0.00, 0.00, 0.00],
                                           3: [3.00, 0.00, 2.00, 3.00, 3.00, 3.00],
                                           4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                           5: [3.00, 3.45, 3.00, 3.00, 3.00, 3.00],
                                           'relevance': [2.00, 3.00, 4.00, 1.00, 1.00, 0.00],
                                           'queryId': [2, 2, 2, 2, 2, 2],
                                           'model': ['b', 'a', 'a', 'b', 'b', 'b']
                                           })
        input_data_frame_t = pd.DataFrame({1: [3.00, 3.00, 3.00, 3.00, 3.00, 3.00],
                                           2: [0.00, 3.00, 0.11, 0.00, 0.00, 0.00],
                                           3: [3.00, 0.00, 2.00, 3.00, 3.00, 3.00],
                                           4: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                           5: [3.00, 3.45, 3.00, 3.00, 3.00, 3.00],
                                           'relevance': [2.00, 3.00, 4.00, 1.00, 1.00, 0.00],
                                           'queryId': [2, 2, 2, 2, 2, 2],
                                           'model': ['b', 'a', 'a', 'b', 'a', 'b']
                                           })
        expected_result_a = [2, 4, 6, 'a']
        expected_result_b = [2, 4, 6, 'b']
        expected_result_t = [2, 3, 6, 't']

        result_a = utils.compute_winning_model(input_data_frame_a, 2)
        result_b = utils.compute_winning_model(input_data_frame_b, 2)
        result_t = utils.compute_winning_model(input_data_frame_t, 2)
        self.assertListEqual(result_a, expected_result_a)
        self.assertListEqual(result_b, expected_result_b)
        self.assertListEqual(result_t, expected_result_t)

    def test_statistical_significance_computation(self):
        input_data_frame = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 'a', 'b', 't'],
             })
        expected_data_frame = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 'a', 'b', 't'],
             'statistical_significance': [1.0000000000000002, 0.031250000000000236, 1.3125, 0.0390625],
             })

        result_data_frame = utils.statistical_significance_computation(input_data_frame, 0.5)
        pd.testing.assert_frame_equal(result_data_frame, expected_data_frame)

    def test_pruning(self):
        input_data_frame = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 'a', 'b', 't'],
             })
        expected_data_frame = pd.DataFrame(
            {'queryId': [1, 0],
             'click_per_winning_model': [6, 1],
             'click_per_query': [6, 9],
             'winning_model': ['a', 't'],
             })
        expected_data_frame.set_index([pd.Int64Index([1, 3], dtype='int64', name='index')], inplace=True)
        expected_data_frame.index.name = None

        result = utils.pruning(input_data_frame)
        pd.testing.assert_frame_equal(result, expected_data_frame)

    def test_computing_ab_score(self):
        input_data_frame_a = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 'a', 'b', 't'],
             })
        expected_result_a = 'a'

        input_data_frame_b = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 'b', 'b', 't'],
             })
        expected_result_b = 'b'

        input_data_frame_t = pd.DataFrame(
            {'queryId': [2, 1, 4, 0],
             'click_per_winning_model': [2, 6, 3, 1],
             'click_per_query': [3, 6, 6, 9],
             'winning_model': ['a', 't', 'b', 't'],
             })
        expected_result_t = 't'

        result_a = utils.computing_winning_model_ab_score(input_data_frame_a)
        result_b = utils.computing_winning_model_ab_score(input_data_frame_b)
        result_t = utils.computing_winning_model_ab_score(input_data_frame_t)

        self.assertEqual(result_a, expected_result_a)
        self.assertEqual(result_b, expected_result_b)
        self.assertEqual(result_t, expected_result_t)
