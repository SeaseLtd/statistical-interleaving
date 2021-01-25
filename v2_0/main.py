#!/usr/bin/python

import getopt
import sys

import experiment_1
import experiment_2


def main(argv):
    unix_options = "hp:s:q:m:e:"
    gnu_options = ["help", "dataset_path=", "seed=", "query_set=", "max_range_pair=", "experiment="]
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
            print("'main.py -p <dataset_path> -s <seed> -q <query_set> -m <max_range_pair> -e <experiment>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--seed"):
            args1.append(int(currentValue))
        elif currentArgument in ("-q", "--query_set"):
            args1.append(int(currentValue))
        elif currentArgument in ("-m", "--max_range_pair"):
            args1.append(int(currentValue))
        elif currentArgument in ("-e", "--experiment"):
            chosen_experiment = currentValue

    if chosen_experiment == '1':
        experiment_1.start_experiment(*args1)
    elif chosen_experiment == '1bis':
        experiment_1.start_experiment(*args1, experiment_one_bis=True)
    elif chosen_experiment == '2':
        experiment_2.start_experiment(*args1)
    elif chosen_experiment == '3':
        experiment_2.start_experiment(*args1, realistic_model=True)
    else:
        print('Error. The selected experiment doesn\'t exist')


if __name__ == "__main__":
    main(sys.argv[1:])
