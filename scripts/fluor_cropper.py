#!/usr/bin/env python3
"""
Author: Chris Dijkstra

Contains functions for cropping fluorescence files.
"""
from skimage import io, measure, registration, transform, filters
import utils
import argparse as arg
import os
from multiprocessing import Pool
import segment
import matplotlib.pyplot as plt
import numpy as np


def arg_reader():
    """ Reads arguments from command line

    :return class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Creates crops of fluorescence images that match RGB crops."
    )
    arg_parser.add_argument("rgb_path", help="The path to the directory holding"
                                             " the rgb images")
    arg_parser.add_argument("fluor_path", help="The path to the directory "
                                               "holding the fluorescence "
                                               "images")
    arg_parser.add_argument("out", help="The directory to write the files "
                                        "to. Does not need to be "
                                        "pre_existing.")
    arg_parser.add_argument("-c", help="Cores used for multiprocessing, 1 by "
                                       "default",
                            type=int, default=1)
    arg_parser.add_argument("-d", help="If this flag is set, a subdirectory "
                                       "will be made in out directory that "
                                       "contains false color images of the "
                                       "fluorescence crops of the mask "
                                       "overlayed.",
                            action="store_true")
    return arg_parser.parse_args()


def rough_crop(mask_a, mask_b, step):
    """ Takes stepwise crops of mask b to get best match with mask a

    :param mask_a: np.ndarray, binary mask to be matched with b
    :param mask_b: np.ndarray, larger binary mask, mask a should fit within
    :param step: int, the step size used to move mask a over mask b
    :return: tuple, the centre of the crop where mask_a had the greatest
        similarity with mask_b
    """
    rough_overlap_dict = {}
    min_r = mask_a.shape[0] // 2
    max_r = mask_b.shape[0] - mask_a.shape[0] // 2
    min_c = mask_a.shape[1] // 2
    max_c = mask_b.shape[1] - mask_a.shape[1] // 2
    for row in range(min_r, max_r + 1, step):
        for col in range(min_c, max_c + 1, step):
            crop = utils.crop_region(mask_b, (col, row), mask_a.shape)
            overlap = (crop == mask_a).sum()
            rough_overlap_dict[overlap] = (col, row)
    optimum = rough_overlap_dict[max(rough_overlap_dict.keys())]
    return optimum


def overlap_crop(rgb_im, full_image):
    """ Uses initial rough crop with phase cross correlation to get match crops

    :param rgb_im: np.ndarray, 3d array representing an RGB image
    :param full_image: np.ndarray, 2d array representing grayscale image
    :return: tuple, the centre of the phase cross correlated rough crop
    """
    mask_a = segment.shw_segmentation(rgb_im)
    mask_b = full_image > filters.threshold_otsu(full_image)
    rough_cen = rough_crop(mask_a, mask_b, 50)
    crop = utils.crop_region(full_image, rough_cen, (1500, 1500))
    crop_mask = crop > filters.threshold_otsu(crop)
    rgb_labs = measure.label(mask_a)
    crop_labs = measure.label(crop_mask)
    rgb_mask = rgb_labs == utils.centre_primary_label(rgb_labs)
    crop_mask = crop_labs == utils.centre_primary_label(crop_labs)
    shift, _, _ = registration.phase_cross_correlation(
        reference_image=rgb_mask,
        moving_image=crop_mask
    )
    return (int(rough_cen[0] - shift[1]), int(rough_cen[1] - shift[0]))


def worker(arg_tup):
    """ Worker for multiprocessing

    :param arg_tup: tuple, contains arguments
    :return: None, writes outfiles
    """
    fm, rgbs, cmd_args = arg_tup
    if rgbs:
        try:
            fm_im = transform.resize(utils.read_fimg(fm), (2823, 3750))
        except Exception as e:
            print(f"could not read fluorescence images, "
                  f"Exception: {e}",
                  flush=True)
        for rgb in rgbs:
            rgb_im = io.imread(rgb)
            rgb_fn = rgb.split("/")[-1]
            try:
                new_centre = overlap_crop(rgb_im, fm_im)
            except Exception as e:
                print(f"Could not overlap {rgb_fn} with {fm}, "
                      f"Exception : {e}",
                      flush=True)
            fm_crop = utils.crop_region(np.pad(fm_im, 500),
                                        (new_centre[0] + 500,
                                         new_centre[1] + 500),
                                        (1500, 1500))
            np.save(cmd_args.out + "/" + rgb_fn.replace(".png", "_Fm"),
                    fm_crop)
            if cmd_args.d:
                if not os.path.isdir(cmd_args.out + "/diagnostic"):
                    os.mkdir(cmd_args.out + "/diagnostic")
                plt.imsave(cmd_args.out + "/diagnostic/" + rgb_fn.replace(
                    ".png", "_Fm.png"),
                           fm_crop)


def pool_handler(cores, fun, params):
    """ Multiprocessing pool handler

    :param cores: int, amount of cores to be used
    :param fun: function, the worker function
    :param params: tuple, the parameter tuples
    :return: None, maps arguments to function
    """
    pools = Pool(cores)
    pools.map(fun, params)


def main():
    """ The main function """
    # Read arguments
    args = arg_reader()
    # Create out_dir if not existing
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # Create list of files
    fluor_files = [args.fluor_path + "/" + file for file in
                   os.listdir(args.fluor_path) if file.endswith("-Fm.fimg")]
    rgb_files = [args.rgb_path + "/" + file for file in
                 os.listdir(args.rgb_path)]
    # Match dict
    match_dict = {}
    for file in fluor_files:
        ident = "-".join(file.split("/")[-1].split("-")[0:3]) + "-"
        match_dict[file] = [file for file in rgb_files if
                            file.split("/")[-1].startswith(ident)]
    params = [tuple(list(tup) + [args]) for tup in match_dict.items()]
    pool_handler(args.c,
                 worker,
                 params)


if __name__ == "__main__":
    main()