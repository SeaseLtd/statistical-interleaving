#!/usr/bin/python

import getopt
import sys

import experiment_1_with_time
import experiment_2_with_time


def main(argv):
    unix_options = "hp:s:e:t"
    gnu_options = ["help", "dataset_path=", "seed=", "experiment=", "monitoring_time"]
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
            print("'main.py -p <dataset_path> -s <seed> -e <experiment> -t <monitoring_time>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--seed"):
            args1.append(int(currentValue))
        elif currentArgument in ("-e", "--experiment"):
            chosen_experiment = currentValue
        elif currentArgument in ("-t", "--monitoring_time"):
            args1.append(True)

    if chosen_experiment == '1':
        experiment_1_with_time.start_experiment(*args1)
    elif chosen_experiment == '1bis':
        experiment_1_with_time.start_experiment(*args1, experiment_one_bis=True)
    elif chosen_experiment == '2':
        experiment_2_with_time.start_experiment(*args1)
    elif chosen_experiment == '3':
        experiment_2_with_time.start_experiment(*args1, realistic_model=True)
    else:
        print('Error. The selected experiment doesn\'t exist')


if __name__ == "__main__":
    main(sys.argv[1:])