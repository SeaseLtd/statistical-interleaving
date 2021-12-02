#!/usr/bin/python

import getopt
import sys

import experiment_1
import experiment_2


def main(argv):
    unix_options = "hp:s:o:q:m:n:e:a:u:"
    gnu_options = ["help", "dataset_path=", "seed=", "output_dir=", "query_set=", "max_range_pair=",
                   "number_of_splits=", "experiment=", "long_tail_path=", "users_scaling_factor"]
    args1 = []
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
                  "-e <experiment> -a <long_tail_path> -u <users_scaling_factor>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            print("-p "+ currentValue)
            args1.append(currentValue)
        elif currentArgument in ("-s", "--seed"):
            print("-s " + currentValue)
            args1.append(int(currentValue))
        elif currentArgument in ("-q", "--query_set"):
            print("-q " + currentValue)
            args1.append(int(currentValue))
        elif currentArgument in ("-m", "--max_range_pair"):
            print("-m " + currentValue)
            args1.append(int(currentValue))
        elif currentArgument in ("-e", "--experiment"):
            print("-e " + currentValue)
            chosen_experiment = currentValue
        elif currentArgument in ("-a", "--long_tail_aggregation_path"):
            print("-a " + currentValue)
            args1.append(currentValue)
        elif currentArgument in ("-u", "--users_scaling_factor"):
            print("-u " + currentValue)
            args1.append(float(currentValue))

    if chosen_experiment == '1':
        experiment_1.start_experiment(*args1)
    elif chosen_experiment == '1_long_tail':
        experiment_1.start_experiment(*args1, experiment_one_long_tail=True)
    elif chosen_experiment == '2':
        experiment_2.start_experiment(*args1)
    elif chosen_experiment == '3':
        experiment_2.start_experiment(*args1, realistic_model=True)
    else:
        print('Error. The selected experiment doesn\'t exist')


if __name__ == "__main__":
    main(sys.argv[1:])
