import os
import numpy as np
import pandas as pd
import utils
from io import StringIO
from unittest import TestCase
from unittest.mock import patch
from pandas.util.testing import assert_frame_equal, assert_numpy_array_equal

EXEC_DIR = os.path.dirname(os.path.abspath(__file__))


class UtilsTest(TestCase):
    def test_load_dataframe(self):
        input_dir = EXEC_DIR + '/resources/train.txt'
        processed_data_frame = utils.load_dataframe(input_dir)

        # Expected dataframe
        expected_dataframe = pd.DataFrame({1: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                           2: [3.0, 0.0, 0.11, 0.0, 0.0, 0.0, 3.0, 0.0, 0.11, 0.0, 0.0, 0.0],
                                           3: [0.0, 3.0, 2.0, 3.0, 3.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0, 3.0],
                                           4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           5: [3.45, 3.0, 3.0, 3.0, 3.0, 3.0, 3.45, 3.0, 3.0, 3.0, 3.0, 3.0],
                                           'relevance': [2, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1],
                                           'query_id': [1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 3]})

        for column_name in [1, 2, 3, 4, 5]:
            expected_dataframe[column_name] = expected_dataframe[column_name].astype('float32')
        for column_name in ['relevance', 'query_id']:
            expected_dataframe[column_name] = expected_dataframe[column_name].astype('int32')

        # Asserting
        assert_frame_equal(processed_data_frame, expected_dataframe)

    def test_parse_solr_json_facet_dataset(self):
        input_dir = EXEC_DIR + '/resources/query_click_user.json'
        processed_data_frame = utils.parse_solr_json_facet_dataset(input_dir)

        # Expected dataframe
        expected_dataframe = pd.DataFrame({
            'userId': ['0d7dca79487741cb22c0525d2c227a1d832cfeb4dcaa94f26fbb7ab2a61a719d',
                       '6ddaf4574436fe1b1cd31286e7aaff209cb0e17330c97f95c54032d7c8df11d4',
                       '8fbdcf068f7b35bc7f64d0d4594d056ae7127f294347c19bd4d67e23c84fe6a2',
                       '70d224c5794db00100d367973056d3e69184f99d61eab80b3bcf004b2cbaae4c',
                       'e45993176582c22eb0266561878ae18688e131a7ccdd4774b6e87d49d62c46b2',
                       'f195c9a0ffd8717db412082a8c6ee6c25bcf474a316d2857261120c9483bf2a7',
                       '3d468e5c64be793bf966ae8e34315de374ba51898a161d93383215fc3540dc3d',
                       '87822d68c93f22b3af89dd0e2d8152686b6212f6e9f11e279693e96dd0148afb',
                       'ffd883b352042b6ee7a15cc95a023175306612eb5d7a287150e02cb25af4ccba'],
            'click_per_userId': [151, 106, 68, 65, 85, 47, 36, 1, 1],
            'query_id': [4577, 4577, 4577, 4577, 4403, 4403, 4403, 5989, 5992],
            'click_per_query': [4, 4, 4, 4, 3, 3, 3, 1, 1]})
        expected_dataframe['query_id'] = expected_dataframe['query_id'].astype('object')
        expected_dataframe['click_per_query'] = expected_dataframe['click_per_query'].astype('object')

        # Asserting
        assert_frame_equal(processed_data_frame, expected_dataframe)

    def test_get_long_tail_query_set(self):
        # Query document pairs
        input_dataframe = pd.DataFrame({1: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                        2: [3.0, 0.0, 0.11, 0.0, 0.0, 0.0, 3.0, 0.0, 0.11, 0.0, 0.0, 0.0],
                                        3: [0.0, 3.0, 2.0, 3.0, 3.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0, 3.0],
                                        4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        5: [3.45, 3.0, 3.0, 3.0, 3.0, 3.0, 3.45, 3.0, 3.0, 3.0, 3.0, 3.0],
                                        'relevance': [2, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1],
                                        'query_id': [1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 3]})

        # Industrial dataset
        input_dir = EXEC_DIR + '/resources/query_click_user_big.json'
        industrial_dataframe = utils.parse_solr_json_facet_dataset(input_dir)

        # Expected dataframe
        expected_set_of_queries = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        result_set_of_queries = utils.get_long_tail_query_set(input_dataframe, industrial_dataframe, 1.0)

        # Asserting
        assert_numpy_array_equal(result_set_of_queries, expected_set_of_queries)

    def test_cache_ranked_lists_per_ranker(self):
        # Input dataframe
        input_dataframe = pd.DataFrame({'rankerA_id': [1, 1, 1, 1, 1, 2, 2],
                                        'rankerB_id': [2, 2, 2, 2, 2, 3, 3],
                                        'query_id': [1, 1, 1, 6, 6, 4, 5],
                                        'rankerA_avg_NDCG': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                        'rankerB_avg_NDCG': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                        'avg_NDCG_winning_ranker': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                                    np.nan]})

        # Query document pairs
        query_document_pairs = pd.DataFrame({1: [3, 3, 5, 1, 0, 2, 4.56],
                                             2: [3, 0, 23, 0.6, 1, -45, 4],
                                             3: [0, 3.4, 1, 2, -3, 0, 2],
                                             4: [0, 80, 2, 3, 1, 2, 34],
                                             'relevance': [2, 3, 2, 0, 1, 3, 2],
                                             'query_id': [1, 1, 1, 6, 6, 4, 5]})

        # Expected ranked list cache
        expected_ranked_list_cache = pd.DataFrame(dict({
            '1_1': [np.array([2, 0, 1]), np.array([2, 2, 3])],
            '1_6': [np.array([3, 4]), np.array([0, 1])],
            '1_4': [np.array([5]), np.array([3])],
            '1_5': [np.array([6]), np.array([2])],
            '2_1': [np.array([2, 0, 1]), np.array([2, 2, 3])],
            '2_6': [np.array([4, 3]), np.array([1, 0])],
            '2_4': [np.array([5]), np.array([3])],
            '2_5': [np.array([6]), np.array([2])],
            '3_1': [np.array([1, 2, 0]), np.array([3, 2, 2])],
            '3_6': [np.array([3, 4]), np.array([0, 1])],
            '3_4': [np.array([5]), np.array([3])],
            '3_5': [np.array([6]), np.array([2])]}))

        # Expected input dataframe
        expected_input_dataframe = pd.DataFrame({
            'rankerA_id': [1, 1, 1, 1, 1, 2, 2],
            'rankerB_id': [2, 2, 2, 2, 2, 3, 3],
            'query_id': [1, 1, 1, 6, 6, 4, 5],
            'rankerA_avg_NDCG': [0.81229, 0.81229, 0.81229, 0.81229, 0.81229, 0.91771,
                                 0.91771],
            'rankerB_avg_NDCG': [0.91771, 0.91771, 0.91771, 0.91771, 0.91771, 0.89457,
                                 0.89457],
            'avg_NDCG_winning_ranker': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                        np.nan]})

        # Asserting
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result_ranked_list = utils.cache_ranked_lists_per_ranker(input_dataframe, 0, query_document_pairs, 3,
                                                                     np.array([1, 1, 1, 6, 6, 4, 5]))
            self.assertEqual(fake_out.getvalue(),
                             '\nRanker[1] AVG NDCG:0.8122857142857144\n\nRanker[2] AVG NDCG:0.9177142857142858\n\nRanker[3] AVG NDCG:0.8945714285714287\n')

        result_ranked_list = pd.DataFrame(result_ranked_list)
        assert_frame_equal(result_ranked_list, expected_ranked_list_cache)
        assert_frame_equal(input_dataframe, expected_input_dataframe)

    def test_compute_ndcg(self):
        ratings_list = np.array([2, 0, 0, 1, 2, 3, 2])
        top_k = 0

        result_ndcg = utils.compute_ndcg(ratings_list, top_k)
        expected_ndcg = 0.670

        self.assertEqual(result_ndcg, expected_ndcg)

        top_k = 3

        result_ndcg = utils.compute_ndcg(ratings_list, top_k)
        expected_ndcg = 0.289

        self.assertEqual(result_ndcg, expected_ndcg)

    def test_execute_team_draft_interleaving(self):
        np.random.seed(0)

        # Input lists
        ranked_list_a = np.array([0, 40, 61, 10, 2, 35, 21])
        ranked_list_b = np.array([0, 20, 83, 11, 2, 3, 75])
        a_ratings = np.array([2, 1, 0, 0, 1, 0, 3])
        b_ratings = np.array([2, 1, 1, 0, 1, 0, 1])

        # Expected list
        expected_list = np.array([[2, 1, 1, 1, 0, 0, 1], [2, 1, 1, 0, 1, 0, 2]], dtype='uint8')

        returned_list = utils.execute_team_draft_interleaving(ranked_list_a, a_ratings, ranked_list_b, b_ratings)

        # Asserting
        assert_numpy_array_equal(returned_list, expected_list)

    def test_simulate_clicks(self):
        np.random.seed(100)

        # Interleaved list
        interleaved_list = np.array([[2, 1, 4, 0, 3, 0, 3], [2, 1, 1, 0, 1, 0, 2]], dtype='uint8')

        top_k = 0
        realistic_model = False

        # Click distribution per rating
        click_distribution_per_rating = np.zeros(5, dtype='int')

        # Expected clicks
        expected_clicks = np.array([[2, 1, 1, 0, 1, 0, 2], [0, 0, 1, 0, 1, 0, 1]], dtype='uint8')

        # Expected click distribution
        expected_clicks_distribution = np.array([0, 0, 0, 1, 1])

        result_clicks = utils.simulate_clicks(interleaved_list, top_k, realistic_model, click_distribution_per_rating)

        # Asserting
        assert_numpy_array_equal(result_clicks, expected_clicks)
        assert_numpy_array_equal(click_distribution_per_rating, expected_clicks_distribution)

        # --------------------------- REALISTIC MODEL TEST ---------------------
        realistic_model = True

        # Click distribution per rating
        click_distribution_per_rating = np.zeros(5, dtype='int')

        # Expected clicks
        expected_clicks = np.array([[2, 1, 1, 0, 1, 0, 2], [0, 0, 1, 0, 0, 0, 0]], dtype='uint8')

        # Expected click distribution
        expected_clicks_distribution = np.array([0, 0, 0, 0, 1])

        result_clicks = utils.simulate_clicks(interleaved_list, top_k, realistic_model, click_distribution_per_rating)

        # Asserting
        assert_numpy_array_equal(result_clicks, expected_clicks)
        assert_numpy_array_equal(click_distribution_per_rating, expected_clicks_distribution)

    def test_aggregate_clicks_per_ranker(self):
        # Interleaved rankers with clicks
        interleaved_rankers_with_clicks = np.array([[2, 0, 0, 1, 1, 0, 2], [1, 0, 0, 1, 1, 1, 0]])

        # Expected clicks count
        expected_clicks_count_per_ranker = np.array([1, 2, 3], dtype='uint16')

        result_clicks_count_per_ranker = utils.aggregate_clicks_per_ranker(interleaved_rankers_with_clicks)

        # Asserting
        assert_numpy_array_equal(result_clicks_count_per_ranker, expected_clicks_count_per_ranker)

    def test_statistical_significance_computation(self):
        # Input dataframe
        input_dataframe = pd.DataFrame({'rankerA_id': [1, 1, 1, 1, 1, 2, 2],
                                        'rankerB_id': [2, 2, 2, 2, 2, 3, 3],
                                        'query_id': [1, 2, 3, 6, 8, 3, 4],
                                        'avg_NDCG_winning_ranker': [1, 0.8, 1, 1, 0.7, 0.9, 0.68],
                                        'interleaving_total_clicks': [2, 3, 1, 3, 1, 2, 2],
                                        'interleaving_winner_clicks': [2, 2, 1, 1, 1, 1, 2],
                                        'interleaving_winner': [1, 0, 0, 2, 2, 0, 1]})

        zero_hypothesis_probability = 0.5

        # Expected input dataframe
        expected_dataframe = pd.DataFrame({'rankerA_id': [1, 1, 1, 1, 1, 2, 2],
                                           'rankerB_id': [2, 2, 2, 2, 2, 3, 3],
                                           'query_id': [1, 2, 3, 6, 8, 3, 4],
                                           'avg_NDCG_winning_ranker': [1, 0.8, 1, 1, 0.7, 0.9, 0.68],
                                           'interleaving_total_clicks': [2, 3, 1, 3, 1, 2, 2],
                                           'interleaving_winner_clicks': [2, 2, 1, 1, 1, 1, 2],
                                           'interleaving_winner': [1, 0, 0, 2, 2, 0, 1],
                                           'statistical_significance': [0.5000000000000002, 1.0000000000000004,
                                                                        1.0000000000000002, 0.3750000000000001,
                                                                        0.5, 1.5000000000000002, 0.5000000000000002]})

        result_dataframe = utils.statistical_significance_computation(input_dataframe, zero_hypothesis_probability)

        # Asserting
        assert_frame_equal(result_dataframe, expected_dataframe)
