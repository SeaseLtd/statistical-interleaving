#!/usr/bin/python

import getopt
import sys

from utils import preprocess_dataset


def main(argv):
    unix_options = "hp:o:"
    gnu_options = ["help", "dataset_path=", "output_path="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main.py -p <dataset_path> -o <output_path>'")
            sys.exit(1)
        elif currentArgument in ("-p", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-o", "--output_path"):
            args1.append(currentValue)

    preprocess_dataset(*args1)


if __name__ == "__main__":
    main(sys.argv[1:])
