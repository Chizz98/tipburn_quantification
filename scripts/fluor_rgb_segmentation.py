#!/usr/bin/env python3
"""
Author: Chris Dijkstra

Contains functions for segmenting tipburn and healthy tissue based on combined
masks of RGB and chlorophyll fluorescence.
"""
import os
import segment
import utils
from skimage import io, util, color, morphology, filters
from multiprocessing import Pool
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt


def arg_reader():
    """ Reads arguments from command line

    :return class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Segments rgb images based on predetermined thresholds"
    )
    arg_parser.add_argument("rgb_path", help="The path to the directory holding"
                                             " the rgb images")
    arg_parser.add_argument("fluor_path", help="The path to the directory "
                                               "holding the fluorescence crops")
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
    arg_parser.add_argument("-s", help="Sigma used for canny edge detection. "
                                       "Used to remove high contrast bordering"
                                       "objects.",
                            type=float, default=2.5)
    return arg_parser.parse_args()


def fluor_thresh(im_channel):
    """ Defines the threshold of an image channel based on its histogram

      :param im_channel: np.ndarray, 2d array, meant to be an Fm image
      :return float, the threshold of the image channel that separates it into
          healthy and unhealthy tissue
    """
    values, bins = np.histogram(im_channel, bins=100)
    peak_i = np.argmax(values)
    val_max = values[peak_i]
    bin_max = bins[peak_i]
    if bin_max <= 0.4:
        bound = 0.2 * val_max
    else:
        bound = 0.5 * val_max
    ref_val = values[values > bound][-1]
    ref_i = np.where(values == ref_val)[0][0]
    ref_bin = bins[ref_i]
    thresh = 2 * ref_bin / 4.5
    return thresh


def pool_handler(cores, fun, params):
    """ Multiprocessing pool handler

    :param cores: int, amount of cores to be used
    :param fun: function, the worker function
    :param params: tuple, the parameter tuples
    :return: None, maps arguments to function
    """
    pools = Pool(cores)
    pools.map(fun, params)


def worker(arg_tup):
    """ Worker for multiprocessing

    :param arg_tup: tuple, contains arguments
    :return: None, writes outfiles
    """
    rgb_crop, fluor_dir, outdir, diag, sigma = arg_tup
    rgb_fn = rgb_crop.split("/")[-1]
    ident = "-".join(rgb_fn.split("-")[:3])
    pos = rgb_fn.split("_")[-2]
    fluor_files = os.listdir(fluor_dir)
    try:
        fluor_match = [file for file in fluor_files if
                       file.startswith(ident + "-") and file.find(pos) != -1
                       and file.endswith("_Fm.npy")][0]
    except Exception as e:
        print(f"Could not find a fluor match for {rgb_fn}. Exception: {e}",
              flush=True)
        return
    # Handle RGB
    rgb_im = io.imread(rgb_crop)
    if rgb_im.shape[2] == 4:
        rgb_im = util.img_as_ubyte(color.rgba2rgb(rgb_im))
    try:
        bg_mask = segment.shw_segmentation(rgb_im)
    except Exception as e:
        print(f"Could not find canny central lab for {rgb_fn}. Exception: {e}",
              flush=True)
        return
    bg_mask = utils.canny_central_ob(rgb_im, bg_mask, sigma)
    bg_mask = morphology.opening(bg_mask, footprint=np.ones((5, 10)))
    bg_mask = morphology.opening(bg_mask, footprint=np.ones((10, 5)))
    bg_mask = morphology.remove_small_holes(
        bg_mask,
        area_threshold=bg_mask.shape[0] * bg_mask.shape[1] // 1000
    )
    comp_mask = segment.barb_hue(
        rgb_im,
        morphology.erosion(bg_mask.copy(), footprint=morphology.disk(2))
    )
    # Handle fluor
    fvfm_im = np.load(fluor_dir + "/" + fluor_match)
    fvfm_im = filters.median(fvfm_im, footprint=morphology.disk(2.5))
    fvfm_im = utils.increase_contrast(fvfm_im)
    fm_mask = fvfm_im > fluor_thresh(fvfm_im[bg_mask == 1])
    fm_comp = fm_mask.astype(int) + bg_mask.astype(int)
    # Combined mask
    final_mask = np.zeros_like(comp_mask)
    final_mask[comp_mask > 0] = 1
    final_mask[(comp_mask == 2) & (fm_mask == 0)] = 2
    # Save mask
    plt.imsave(
        fname=outdir + "/" + rgb_fn.replace(".png", "_comp.png"),
        arr=final_mask,
        cmap="binary_r",
        vmin=0,
        vmax=2
    )
    if diag:
        if not os.path.isdir(outdir + "/diagnostic"):
            os.mkdir(outdir + "/diagnostic")
        plot, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].imshow(rgb_im)
        axes[0, 1].imshow(fm_comp, cmap="viridis_r")
        axes[1, 0].imshow(comp_mask)
        axes[1, 1].imshow(final_mask)
        plot.set_size_inches(20, 20)
        plt.tight_layout()
        plot.savefig(outdir + "/diagnostic/" +
                     rgb_fn.replace(".png", "_mask.png"))
    plt.close("all")


def main():
    """ The main function """
    # Read arguments
    args = arg_reader()
    # Create out_dir if not existing
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # Create list of files
    files = [args.rgb_path + "/" + file for file in os.listdir(args.rgb_path)]
    # Create list of parameters
    params = zip(files, [args.fluor_path] * len(files), [args.out] * len(files),
                 [args.d] * len(files), [args.s] * len(files))
    # Send to pool handler
    pool_handler(args.c, worker, params)


if __name__ == "__main__":
    main()

