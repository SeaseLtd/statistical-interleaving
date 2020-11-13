import time
import utils
import numpy as np
from dataset_creation import create_adore_dataset, create_primary_dataset, create_variation_dataset


def start_experiment(num_variations, model_preference, max_clicks_per_user, is_test=False):
    # h = hpy()
    dataset_path = './dataset_from_adore/query_click_user.json'
    percentage_dropped_queries = []

    if model_preference >= 0.5:
        min_percentage_click_per_user_id = 1 - 0.5 / model_preference
        max_percentage_click_per_user_id = 1
    else:
        min_percentage_click_per_user_id = 0
        max_percentage_click_per_user_id = 0.5 / (1 - model_preference)

    print('------------- Creating adore dataset ----------------')
    start = time.time()
    adore_dataset = create_adore_dataset(dataset_path, min_percentage_click_per_user_id,
                                         max_percentage_click_per_user_id, is_test)
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
    primary_dataset = create_primary_dataset(adore_dataset, min_percentage_click_per_user_id,
                                             max_percentage_click_per_user_id, max_clicks_per_user)
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

    agree_no_pruning = 0
    agree_with_pruning = 0
    agree_between_variation = 0
    seeds = np.arange(start=0, stop=num_variations)

    for i in range(0, num_variations):
        print('\n\n***************************** Round ' + str(i) + ' ************************************')
        print('------------- Creating variation dataset --------------')
        start = time.time()
        variation_dataset = create_variation_dataset(primary_dataset, adore_total_click_for_variation, seeds[i])
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

    print('Number of times the TDI is correct = ' + str(agree_no_pruning) + '/' + str(num_variations))
    print('Number of times the SSP is correct = ' + str(agree_with_pruning) + '/' + str(num_variations))
    print('Average percentage of dropped queries = ' + str(sum(percentage_dropped_queries) / len(
        percentage_dropped_queries)))
