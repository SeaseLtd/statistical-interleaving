import time
import utils
import numpy as np
import pandas as pd


def create_primary_dataset(adore_beauty_dataset):
    primary_data = adore_beauty_dataset.copy()
    primary_data['click_per_query_primary'] = adore_beauty_dataset['click_per_query'].max()
    primary_data['new_interactions_to_add'] = primary_data['click_per_query_primary'] - primary_data['click_per_query']
    data_to_add = primary_data.drop_duplicates(
        subset=['queryId', 'new_interactions_to_add'], keep='last')[['queryId', 'new_interactions_to_add',
                                                                     'click_per_query']].reset_index(drop=True)
    primary_data.drop(columns=['click_per_query', 'new_interactions_to_add'], inplace=True)
    primary_data.rename(columns={'click_per_query_primary': 'click_per_query'}, inplace=True)

    new_data = utils.generate_new_data(data_to_add, adore_beauty_dataset['click_per_query'].max())
    primary_data = primary_data.append(new_data, ignore_index=True)

    primary_data = primary_data[['userId', 'click_per_userId', 'queryId', 'click_per_query', 'click_per_model_A']]
    primary_data = primary_data.astype({'queryId': 'int64', 'click_per_query': 'int64', 'click_per_model_A': 'int64',
                                        'click_per_userId': 'int64', 'userId': 'int16'})
    primary_data['userId'] = primary_data.groupby('queryId').cumcount() + 1

    return primary_data


def create_variation_dataset(primary_data, click_per_query_adore):
    temp_variation_data_frame = primary_data.copy()
    temp_variation_data_frame.drop(columns=['click_per_query'], inplace=True)
    temp_variation_data_frame = pd.merge(temp_variation_data_frame, click_per_query_adore, left_on='queryId',
                                         right_index=True, how='inner')
    interactions_added_data_frames = []
    while temp_variation_data_frame['click_per_query'].values.sum() > 0:
        groupedByQueryId = temp_variation_data_frame.groupby(['queryId'])
        interactions_added_single_pass = groupedByQueryId.sample() #Allow or disallow sampling of the same row more than once.
        interactions_added_single_pass['click_per_query_after_addition'] = interactions_added_single_pass['click_per_query'] - interactions_added_single_pass['click_per_userId']

        interactions_added_single_pass['click_per_model_A'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            (interactions_added_single_pass['click_per_model_A'] * interactions_added_single_pass['click_per_query']) /
            interactions_added_single_pass['click_per_userId'], interactions_added_single_pass['click_per_model_A'])
        interactions_added_single_pass['click_per_userId'] = np.where(
            interactions_added_single_pass['click_per_query_after_addition'] < 0,
            interactions_added_single_pass['click_per_query'], interactions_added_single_pass['click_per_userId'])
        interactions_added_single_pass = interactions_added_single_pass.astype({'click_per_model_A': 'int64'})

        interactions_added_data_frames.append(interactions_added_single_pass)
        #update and reset initial pass data structures
        temp_variation_data_frame = pd.merge(temp_variation_data_frame, interactions_added_single_pass,
                                             left_on='queryId', right_on='queryId', how='left', suffixes=('', '_y'))
        temp_variation_data_frame.drop(columns=['click_per_query'], inplace=True)
        temp_variation_data_frame = temp_variation_data_frame[temp_variation_data_frame.columns[
            ~temp_variation_data_frame.columns.str.endswith('_y')]]
        temp_variation_data_frame.rename(columns={'click_per_query_after_addition': 'click_per_query'}, inplace=True)
        temp_variation_data_frame.drop(interactions_added_single_pass.index, inplace=True)
        temp_variation_data_frame = temp_variation_data_frame.loc[temp_variation_data_frame['click_per_query'] > 0]
        temp_variation_data_frame.reset_index(drop=True, inplace=True)

    variation_data_frame = pd.concat(interactions_added_data_frames, ignore_index=True, sort=True)
    variation_data_frame.drop(columns=['click_per_query_after_addition'], inplace=True)
    variation_data_frame.reset_index(drop=True, inplace=True)
    variation_data_frame.drop(columns=['click_per_query'], inplace=True)
    variation_data_frame = pd.merge(variation_data_frame, click_per_query_adore, left_on='queryId', right_index=True,
                                    how='inner')
    variation_data_frame.reset_index(drop=True, inplace=True)

    variation_data_frame = variation_data_frame[['userId', 'click_per_userId', 'queryId', 'click_per_query',
                                                 'click_per_model_A']]

    return variation_data_frame


if __name__ == '__main__':
    # h = hpy()
    percentage_dropped_queries = []

    print('------------- Creating adore dataset ----------------')
    start = time.time()
    adore_dataset = utils.create_adore_dataset()
    print('Rows: ' + str(len(adore_dataset.index)))
    end = time.time()
    print('Time: ' + str(end - start))
    print('------------- Elaborating adore dataset -------------')
    start = time.time()
    adore_dataset_for_score = utils.elaborate_dataset_for_score(adore_dataset)
    end = time.time()
    print('Time: ' + str(end - start))
    print('------------- Pruning on adore dataset --------------')
    start = time.time()
    adore_dataset_with_pruning = utils.pruning(adore_dataset_for_score, percentage_dropped_queries)
    end = time.time()
    print('Time: ' + str(end - start))
    # print('ADORE AFTER PRUNING')
    # print(h.heap())

    print('\n------------- Creating primary dataset --------------')
    start = time.time()
    primary_dataset = create_primary_dataset(adore_dataset)
    print('Rows: ' + str(len(primary_dataset.index)))
    end = time.time()
    print('Time: ' + str(end - start))
    print('------------- Elaborating primary dataset --------------')
    start = time.time()
    primary_dataset_for_score = utils.elaborate_dataset_for_score(primary_dataset)
    end = time.time()
    print('Time: ' + str(end - start))
    print('------------- Pruning on primary dataset --------------')
    start = time.time()
    primary_dataset_with_pruning = utils.pruning(primary_dataset_for_score, percentage_dropped_queries)
    end = time.time()
    print('Time: ' + str(end - start))
    # print('PRIMARY AFTER PRUNING')
    # print(h.heap())

    print('\n------------- Computing AB score for primary dataset --------------')
    start = time.time()
    ab_score_primary_no_pruning = utils.computing_ab_score(primary_dataset_for_score)
    end = time.time()
    print('Time: ' + str(end - start))
    start = time.time()
    ab_score_primary_with_pruning = utils.computing_ab_score(primary_dataset_with_pruning)
    end = time.time()
    print('Time: ' + str(end - start))
    print('AB score without pruning for primary dataset = ' + str(ab_score_primary_no_pruning))
    print('AB score with pruning for primary dataset = ' + str(ab_score_primary_with_pruning))
    # print('PRIMARY AFTER AB')
    # print(h.heap())

    print('------------- Computing AB score for adore dataset --------------')
    start = time.time()
    ab_score_adore_no_pruning = utils.computing_ab_score(adore_dataset_for_score)
    end = time.time()
    print('Time: ' + str(end - start))
    start = time.time()
    ab_score_adore_with_pruning = utils.computing_ab_score(adore_dataset_with_pruning)
    end = time.time()
    print('Time: ' + str(end - start))
    print('AB score without pruning for adore dataset = ' + str(ab_score_adore_no_pruning))
    print('AB score with pruning for adore dataset = ' + str(ab_score_adore_with_pruning))
    # print('ADORE AFTER AB')
    # print(h.heap())

    agree_no_pruning = 0
    agree_with_pruning = 0
    agree_between_variation = 0
    adore_total_click_for_variation = adore_dataset.drop_duplicates(
        subset=['queryId', 'click_per_query'], keep='last')[['queryId', 'click_per_query']]
    adore_total_click_for_variation.set_index('queryId', inplace=True)
    del adore_dataset
    del adore_dataset_for_score
    del adore_dataset_with_pruning
    del primary_dataset_for_score
    del primary_dataset_with_pruning
    # print('AFTER DEL')
    # print(h.heap())

    for i in range(0, 1000):
        print('\n\n***************************** Round ' + str(i) + ' ************************************')
        print('------------- Creating variation dataset --------------')
        start = time.time()
        variation_dataset = create_variation_dataset(primary_dataset, adore_total_click_for_variation)
        end = time.time()
        print('Rows: ' + str(len(variation_dataset.index)))
        print('Time: ' + str(end - start))
        print('------------- Elaborating variation dataset --------------')
        start = time.time()
        variation_dataset_for_score = utils.elaborate_dataset_for_score(variation_dataset)
        end = time.time()
        print('Time: ' + str(end - start))
        # print('BEFORE VAR DEL')
        # print(h.heap())
        del variation_dataset
        # print('AFTER VAR DEL')
        # print(h.heap())
        print('------------- Pruning variation dataset --------------')
        start = time.time()
        variation_dataset_with_pruning = utils.pruning(variation_dataset_for_score, percentage_dropped_queries)
        end = time.time()
        print('Time: ' + str(end - start))
        # print('AFTER VAR PRUNING')
        # print(h.heap())

        print('\n------------- Computing AB score for variation dataset --------------')
        start = time.time()
        ab_score_variation_no_pruning = utils.computing_ab_score(variation_dataset_for_score)
        end = time.time()
        print('Time: ' + str(end - start))
        start = time.time()
        ab_score_variation_with_pruning = utils.computing_ab_score(variation_dataset_with_pruning)
        end = time.time()
        print('Time: ' + str(end - start))
        print('AB score without pruning for variation dataset = ' + str(ab_score_variation_no_pruning))
        print('AB score with pruning for variation dataset = ' + str(ab_score_variation_with_pruning))
        # print('BEFORE VAR AB')
        # print(h.heap())
        del variation_dataset_for_score
        del variation_dataset_with_pruning
        # print('AFTER VAR DEL')
        # print(h.heap())

        print('------------- Comparing AB score between variation dataset and primary dataset --------------')
        start = time.time()
        comparison_with_primary = utils.same_score(ab_score_primary_no_pruning, ab_score_variation_no_pruning)
        end = time.time()
        print('Time: ' + str(end - start))
        # print('AFTER COMPARISON')
        # print(h.heap())
        if comparison_with_primary:
            agree_no_pruning = agree_no_pruning + 1

        start = time.time()
        comparison_with_primary_pruning = utils.same_score(ab_score_primary_with_pruning,
                                                           ab_score_variation_with_pruning)
        end = time.time()
        print('Time: ' + str(end - start))
        # print('AFTER COMPARISON')
        # print(h.heap())
        if comparison_with_primary_pruning:
            agree_with_pruning = agree_with_pruning + 1

        start = time.time()
        comparison_between_variation = utils.same_score(ab_score_variation_no_pruning, ab_score_variation_with_pruning)
        end = time.time()
        print('Time: ' + str(end - start))
        # print('AFTER COMPARISON')
        # print(h.heap())
        if comparison_between_variation:
            agree_between_variation = agree_between_variation + 1

    print('Number of times the TDI is correct = ' + str(agree_no_pruning) + '/100')
    print('Number of times the SS is correct = ' + str(agree_with_pruning) + '/100')
    print('Average percentage of dropped queries = ' + str(sum(percentage_dropped_queries) / len(
        percentage_dropped_queries)))
    # print('Number of times the variation with and without pruning agree = ' + str(agree_between_variation) + '/100')
