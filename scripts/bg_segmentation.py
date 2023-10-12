#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 10/10/2023

Contains functions to run segmentation from the command line
"""
import argparse as arg
import os
import segment
import matplotlib.pyplot as plt
from skimage import io, morphology
import numpy as np


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
        "seeds",
        help="The number of starting seeds for the watershed segmentation",
        type=int
    )
    arg_parser.add_argument(
        "-ht",
        help="The hue threshold, every pixel where the hue is below this will "
             "be marked as 0. Hue is given in a range between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-st",
        help="The saturation threshold, every pixel where the saturation is "
             "below this will be marked as 0. Saturation is given in a range "
             "between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-vt",
        help="the value threshold, every pixel where the value will be below"
             "this threshold will be marked as 0. Saturation is given between 0"
             "and 1",
        type=float, default=0.0
    )
    return arg_parser.parse_args()


def main():
    args = arg_reader()
    directory = args.filename
    if os.path.isdir(directory):
        files = os.listdir(directory)
        for file in files:
            try:
                image = io.imread(directory + "/" + file)
            except:
                print(f"Could not open {directory + '/' + file}")
            else:
                mask = segment.water_hsv_thresh(
                    rgb_im=image, n_seeds=args.seeds, h_th=args.ht,
                    s_th=args.st, v_th=args.vt
                )
                mask = morphology.binary_opening(mask)
                mask = morphology.remove_small_objects(mask)
                plt.imsave("out/" + file, arr=mask, cmap="gray")


if __name__ == "__main__":
    main()
