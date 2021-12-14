#!/usr/bin/python

import getopt
import sys

import experiment_1
import experiment_2


def main(argv):
    unix_options = "hp:s:o:q:m:e:a:u:n:c:r:"
    gnu_options = ["help", "dataset_path=", "seed=", "output_dir=", "query_set=", "max_range_pair=",
                   "number_of_splits=", "experiment=", "long_tail_path=", "users_scaling_factor",
                   "ndcg_top_k", "click_generation_top_k", "click_generation_realistic"]
    para_dict = dict()
    chosen_experiment = '1'

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main.py -p <dataset_path> -s <seed> -q <query_set> -m <max_range_pair> "
                  "-e <experiment> -a <long_tail_path> -u <users_scaling_factor>' "
                  "-n <ndcg_top_k>' -c <click_generation_top_k>' -r <click_generation_realistic>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            print("-p "+ currentValue)
            para_dict['dataset_path'] = currentValue
        elif currentArgument in ("-s", "--seed"):
            print("-s " + currentValue)
            para_dict['seed'] = int(currentValue)
        elif currentArgument in ("-q", "--query_set"):
            print("-q " + currentValue)
            para_dict['queries_to_evaluate_count'] = int(currentValue)
        elif currentArgument in ("-m", "--max_range_pair"):
            print("-m " + currentValue)
            para_dict['rankers_to_evaluate_count'] = int(currentValue)
        elif currentArgument in ("-e", "--experiment"):
            print("-experiment " + currentValue)
            chosen_experiment = currentValue
        elif currentArgument in ("-a", "--long_tail_aggregation_path"):
            print("-solr_aggregation_json_path " + currentValue)
            para_dict['solr_aggregation_json_path'] = currentValue
        elif currentArgument in ("-u", "--users_scaling_factor"):
            print("-users_scaling_factor " + currentValue)
            para_dict['users_scaling_factor'] = float(currentValue)
        elif currentArgument in ("-n", "--ndcg_top_k"):
            print("-ndcg_top_k " + currentValue)
            para_dict['ndcg_top_k'] = int(currentValue)
        elif currentArgument in ("-c", "--click_generation_top_k"):
            print("-click_generation_top_k " + currentValue)
            para_dict['click_generation_top_k'] = int(currentValue)
        elif currentArgument in ("-r", "--click_generation_realistic"):
            print("-click_generation_realistic " + currentValue)
            para_dict['click_generation_realistic'] = bool(currentValue)

    if chosen_experiment == '1':
        experiment_1.start_experiment(**para_dict)
    elif chosen_experiment == '1_long_tail':
        experiment_1.start_experiment(**para_dict, experiment_one_long_tail=True)
    else:
        print('Error. The selected experiment doesn\'t exist')


if __name__ == "__main__":
    main(sys.argv[1:])
