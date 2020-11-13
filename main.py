#!/usr/bin/python

import sys
import getopt
from experiment import start_experiment


def main(argv):
    unix_options = "hn:p:m:t"
    gnu_options = ["help", "number_of_variations=", "model_preference=", "max_clicks_per_user=", "is_test"]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main.py -n <number_of_variations> -p <model_preference> -m <max_clicks_per_user> -t <is_test>'")
            sys.exit(1)
        elif currentArgument in ("-n", "--number_of_variations"):
            args1.append(int(currentValue))
        elif currentArgument in ("-p", "--model_preference"):
            args1.append(float(currentValue))
        elif currentArgument in ("-m", "--max_clicks_per_user"):
            args1.append(int(currentValue))
        elif currentArgument in ("-t", "--is_test"):
            args1.append(True)

    start_experiment(*args1)


if __name__ == "__main__":
    main(sys.argv[1:])
