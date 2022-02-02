import os
import numpy as np
import pandas as pd
import utils
from unittest import TestCase
from pandas.util.testing import assert_frame_equal, assert_numpy_array_equal

EXEC_DIR = os.path.dirname(os.path.abspath(__file__))


class UtilsTest(TestCase):
    def test_load_dataframe(self):
        input_dir = EXEC_DIR + '/resources/train.txt'
        processed_data_frame = utils.load_dataframe(input_dir)

        # Expected dataframe
        expected_dataframe = pd.DataFrame()
        expected_dataframe[1] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        expected_dataframe[2] = [3.0, 0.0, 0.11, 0.0, 0.0, 0.0, 3.0, 0.0, 0.11, 0.0, 0.0, 0.0]
        expected_dataframe[3] = [0.0, 3.0, 2.0, 3.0, 3.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0, 3.0]
        expected_dataframe[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected_dataframe[5] = [3.45, 3.0, 3.0, 3.0, 3.0, 3.0, 3.45, 3.0, 3.0, 3.0, 3.0, 3.0]
        expected_dataframe['relevance'] = [2, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1]
        expected_dataframe['query_id'] = [1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 3]

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
        expected_dataframe = pd.DataFrame()
        expected_dataframe['userId'] = ['0d7dca79487741cb22c0525d2c227a1d832cfeb4dcaa94f26fbb7ab2a61a719d',
                                        '6ddaf4574436fe1b1cd31286e7aaff209cb0e17330c97f95c54032d7c8df11d4',
                                        '8fbdcf068f7b35bc7f64d0d4594d056ae7127f294347c19bd4d67e23c84fe6a2',
                                        '70d224c5794db00100d367973056d3e69184f99d61eab80b3bcf004b2cbaae4c',
                                        'e45993176582c22eb0266561878ae18688e131a7ccdd4774b6e87d49d62c46b2',
                                        'f195c9a0ffd8717db412082a8c6ee6c25bcf474a316d2857261120c9483bf2a7',
                                        '3d468e5c64be793bf966ae8e34315de374ba51898a161d93383215fc3540dc3d',
                                        '87822d68c93f22b3af89dd0e2d8152686b6212f6e9f11e279693e96dd0148afb',
                                        'ffd883b352042b6ee7a15cc95a023175306612eb5d7a287150e02cb25af4ccba']
        expected_dataframe['click_per_userId'] = [151, 106, 68, 65, 85, 47, 36, 1, 1]
        expected_dataframe['query_id'] = [4577, 4577, 4577, 4577, 4403, 4403, 4403, 5989, 5992]
        expected_dataframe['query_id'] = expected_dataframe['query_id'].astype('object')
        expected_dataframe['click_per_query'] = [4, 4, 4, 4, 3, 3, 3, 1, 1]
        expected_dataframe['click_per_query'] = expected_dataframe['click_per_query'].astype('object')

        # Asserting
        assert_frame_equal(processed_data_frame, expected_dataframe)

    def test_get_long_tail_query_set(self):
        # Query document pairs
        input_dataframe = pd.DataFrame()
        input_dataframe[1] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        input_dataframe[2] = [3.0, 0.0, 0.11, 0.0, 0.0, 0.0, 3.0, 0.0, 0.11, 0.0, 0.0, 0.0]
        input_dataframe[3] = [0.0, 3.0, 2.0, 3.0, 3.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0, 3.0]
        input_dataframe[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_dataframe[5] = [3.45, 3.0, 3.0, 3.0, 3.0, 3.0, 3.45, 3.0, 3.0, 3.0, 3.0, 3.0]
        input_dataframe['relevance'] = [2, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1]
        input_dataframe['query_id'] = [1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 3]

        # Industrial dataset
        input_dir = EXEC_DIR + '/resources/query_click_user_big.json'
        industrial_dataframe = utils.parse_solr_json_facet_dataset(input_dir)

        # Expected dataframe
        expected_set_of_queries = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        result_set_of_queries = utils.get_long_tail_query_set(input_dataframe, industrial_dataframe, 1.0)

        # Asserting
        assert_numpy_array_equal(result_set_of_queries, expected_set_of_queries)
