#!/usr/bin/env python3
"""
Author: Chris Dijkstra

Script for segmentation of RGB images into foreground and background,
and for splitting the foreground into healthy and unhealthy tissue.
"""
import os
import segment
import utils
from skimage import io, util, color, morphology, segmentation
from multiprocessing import Pool
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt
import re


def arg_reader():
    """ Reads arguments from command line

    :return ..., class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Segments rgb images based on predetermined thresholds"
    )
    arg_parser.add_argument("filename", help="The path to the directory holding"
                                             " the images")
    arg_parser.add_argument("out", help="The directory to write the files "
                                        "to. Does not need to be "
                                        "pre_existing. ../out as default.")
    arg_parser.add_argument("-s", help="Sigma used for canny edge detection. "
                                       "Used to remove high contrast bordering"
                                       "objects.",
                            type=float, default=3.0)
    arg_parser.add_argument("-c", help="Cores used for multiprocessing, 1 by "
                                       "default",
                            type=int, default=1)
    arg_parser.add_argument("-d", help="If this flag is set, a subdirectory "
                                       "will be made in out directory that "
                                       "contains RGB images with the outline "
                                       "of the mask overlayed.",
                            action="store_true")
    return arg_parser.parse_args()


def segment_file(arg_tup):
    """ Reads an image file and segments foreground from background

    :return arg_tup: tuple, contains all parameters in order filename, outfile,
        sigma
    :return: np.ndarray, binary segmentation of the image.
    """
    filename, outfile, sigma, diagnostic = arg_tup
    print(f"Starting bg_segmentation of {filename}")
    try:
        rgb_im = io.imread(filename)
        if rgb_im.shape[2] == 4:
            rgb_im = util.img_as_ubyte(color.rgba2rgb(rgb_im))
    except:
        print(f"Could not open {filename}")
        return
    # Background masking
    bg_mask = segment.shw_segmentation(rgb_im, distance=15, bg_mod=0.5,
                                       fg_mod=0.5)
    bg_mask = morphology.closing(bg_mask, footprint=morphology.disk(5))
    try:
        bg_mask = utils.canny_central_ob(rgb_im, bg_mask, sigma)
    except:
        print(f"Could not isolate primary object in {filename}")
    bg_mask = morphology.opening(bg_mask, footprint=morphology.disk(3.5))
    # Tipburn masking
    comp_mask = segment.barb_hue(rgb_im, bg_mask, 3.5)
    plt.imsave(
        fname=os.path.join(outfile,
                           filename.split("/")[-1].replace(".jpg", ".png")),
        arr=comp_mask.astype("uint8"),
        cmap="binary_r",
        vmin=0,
        vmax=2
    )
    if diagnostic:
        diag_path = os.path.join(outfile, "diagnostic")
        if not os.path.isdir(diag_path):
            os.mkdir(diag_path)
        plt.imsave(
            os.path.join(
                diag_path,
                filename.split("/")[-1]
            ).replace(".jpg", "_bg.jpg"),
            segmentation.mark_boundaries(rgb_im, bg_mask)
        )
        plt.imsave(
            os.path.join(
                diag_path,
                filename.split("/")[-1]
            ).replace(".jpg", "_tb.jpg"),
            segmentation.mark_boundaries(rgb_im, comp_mask == 2)
        )


def parse_segmentations(image_files, out_dir):
    """ Function to parse segmentation images

    :param image_files: str, the directory containing the RGB images matching
        the segmentation masks in out_dir
    :param out_dir: str, the directory to write the pixel table, also contains
        the segmentation masks as grayscale images
    :return: None, writes a tab separated file
    """
    filename_pattern = re.compile(r".*[\/\\]([0-9]+)-([0-9]+).+Tray_0([0-9]*)"
                                  r".+pos([0-9*])_(.*).png")
    outfile = open(out_dir + "/pixel_table.txt", "w")
    outfile.write("\t".join(
        ["experiment", "round", "tray", "position", "accession", "full_healthy",
         "full_brown", "hearth_healthy", "hearth_brown"]) + "\n"
    )
    count = 0
    for file in image_files:
        match = filename_pattern.search(file)
        if match:
            exp_dat = list(match.groups())
        else:
            exp_dat = ["NA"] * 5
        mask_fn = out_dir + "/" + file.split("/")[-1]
        mask_fn = mask_fn.replace(".png", "_comp.png")
        try:
            mask = io.imread(mask_fn, as_gray=True) // 0.5
        except:
            exp_dat += ["NA", "NA", "NA", "NA"]
        else:
            full_mask = morphology.disk((min(mask.shape[:2]) - 1)/2) * mask
            healthy_full = (full_mask == 1).sum()
            brown_full = (full_mask == 2).sum()
            hearth_disk = morphology.disk((min(mask.shape[:2]) - 2)/4)
            hearth = utils.crop_region(
                mask,
                (mask.shape[0] // 2, mask.shape[1] // 2),
                (hearth_disk.shape[0], hearth_disk.shape[1])
            )
            hearth_mask = hearth * hearth_disk
            healthy_hearth = (hearth_mask == 1).sum()
            brown_hearth = (hearth_mask == 2).sum()
            exp_dat += [healthy_full, brown_full, healthy_hearth, brown_hearth]
        exp_dat = [str(dat) for dat in exp_dat]
        outfile.write("\t".join(exp_dat) + "\n")
        count += 1
        print(f"File {count} of {len(image_files)} parsed")
    outfile.close()


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
    files = [args.filename + "/" + file for file in os.listdir(args.filename)]
    param_list = zip(
        files, [args.out] * len(files),
        [args.s] * len(files),
        [args.d] * len(files))
    # Pooled segmentation
    pool_handler(args.c, segment_file, param_list)


if __name__ == "__main__":
    main()
