#!/usr/bin/python

import getopt
import sys

import experiment_1


def main(argv):
    unix_options = "hp:s:q:m:e:a:u:n:c:r:"
    gnu_options = ["help", "dataset_path=", "seed=", "query_set=", "max_range_pair=",
                   "experiment=", "long_tail_dataset_path=", "long_tail_scaling_factor",
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
            print("'main.py -p <dataset_path> -s <seed> -q <queries_to_evaluate_count> -m <rankers_to_evaluate_count> "
                  "-e <experiment> -a <long_tail_dataset_path> -u <long_tail_scaling_factor>' "
                  "-n <ndcg_top_k>' -c <click_generation_top_k>' -r <click_generation_realistic>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            print("dataset_path= "+ currentValue)
            para_dict['dataset_path'] = currentValue
        elif currentArgument in ("-s", "--seed"):
            print("seed= " + currentValue)
            para_dict['seed'] = int(currentValue)
        elif currentArgument in ("-q", "--queries_to_evaluate_count"):
            print("queries_to_evaluate_count= " + currentValue)
            para_dict['queries_to_evaluate_count'] = int(currentValue)
        elif currentArgument in ("-m", "--rankers_to_evaluate_count"):
            print("rankers_to_evaluate_count= " + currentValue)
            para_dict['rankers_to_evaluate_count'] = int(currentValue)
        elif currentArgument in ("-e", "--experiment"):
            print("-experiment " + currentValue)
            chosen_experiment = currentValue
        elif currentArgument in ("-a", "--long_tail_aggregation_path"):
            print("long_tail_dataset_path= " + currentValue)
            para_dict['long_tail_dataset_path'] = currentValue
        elif currentArgument in ("-u", "--long_tail_scaling_factor"):
            print("long_tail_scaling_factor= " + currentValue)
            para_dict['long_tail_scaling_factor'] = float(currentValue)
        elif currentArgument in ("-n", "--ndcg_top_k"):
            print("ndcg_top_k= " + currentValue)
            para_dict['ndcg_top_k'] = int(currentValue)
        elif currentArgument in ("-c", "--click_generation_top_k"):
            print("click_generation_top_k= " + currentValue)
            para_dict['click_generation_top_k'] = int(currentValue)
        elif currentArgument in ("-r", "--click_generation_realistic"):
            print("click_generation_realistic= " + currentValue)
            para_dict['click_generation_realistic'] = bool(currentValue)

    if chosen_experiment == '1':
        experiment_1.start_experiment(**para_dict)
    elif chosen_experiment == '1_long_tail':
        experiment_1.start_experiment(**para_dict, experiment_one_long_tail=True)
    else:
        print('Error. The selected experiment doesn\'t exist')


if __name__ == "__main__":
    main(sys.argv[1:])
