#!/usr/bin/env python3
import os
import segment
import utils
from skimage import io, util, color, morphology
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import argparse as arg


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
    return arg_parser.parse_args()


def segment_file(arg_tup):
    """ Reads an image file and segments foreground from background

    :param filename: str, the path to the file to be read
    :param outfile: str, the directory in which the masks will be saved, has to
        be pre-existing
    :param sigma: float, the sigma used for canny edge detection based object
        separation
    :return: np.ndarray, binary segmentation of the image.
    """
    filename, outfile, sigma = arg_tup
    out_fn = outfile + "/" + filename.split("/")[-1]
    try:
        rgb_im = io.imread(filename)
        if rgb_im.shape[2] == 4:
            rgb_im = util.img_as_ubyte(color.rgba2rgb(rgb_im))
    except:
        print(f"Could not open {filename}")
    else:
        try:
            bg_mask = segment.shw_segmentation(rgb_im)
        except:
            print(f"Could not segment foreground from background in {filename}")
        else:
            bg_mask = utils.canny_central_ob(rgb_im, bg_mask, sigma)
            bg_mask = morphology.opening(bg_mask,
                                         footprint=np.ones((5, 10)))
            bg_mask = morphology.opening(bg_mask,
                                         footprint=np.ones((10, 5)))
            plt.imsave(
                fname=out_fn.replace(".png", "_fg.png"),
                arr=utils.multichannel_mask(rgb_im, bg_mask)
            )

            plt.imsave(
                fname=out_fn.replace(".png", "_bg.png"),
                arr=utils.multichannel_mask(rgb_im, bg_mask == 0)
            )


def pool_handler(cores, fun, params):
    pools = Pool(cores)
    pools.map(fun, params)


def main():
    # Read arguments
    args = arg_reader()
    # Create out_dir if not existing
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # Create list of files
    files = [args.filename + "/" + file for file in os.listdir(args.filename)]
    param_list = zip(files, [args.out] * len(files), [args.s] * len(files))
    pool_handler(args.c, segment_file, param_list)


if __name__ == "__main__":
    main()
