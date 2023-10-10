"""
Author: Chris Dijkstra
Date: 10/10/2023

Contains functions to run segmentation from the command line
"""
import argparse as arg
import os


def arg_reader():
    """ Reads arguments from command line

    :return ..., class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Segments rgb images based on predetermined thresholds"
    )
    arg_parser.add_argument("filename", help="The path to the directory holding"
                                             " the images")
    arg_parser.add_argument(
        "-ht",
        help="The hue threshold, every pixel where the hue is below this will "
             "be marked as 0. Hue is given in a range between 0 and 1."
    )
    arg_parser.add_argument(
        "-st",
        help="The saturation threshold, every pixel where the saturation is "
             "below this will be marked as 0. Saturation is given in a range "
             "between 0 and 1."
    )
    arg_parser.add_argument(
        "-vt",
        help="the value threshold, every pixel where the value will be below"
             "this threshold will be marked as 0. Saturation is given between 0"
             "and 1"
    )
    return arg_parser.parse_args()


def main():
    args = arg_reader()
    print(args.filename)


if __name__ == "__main__":
    main()
