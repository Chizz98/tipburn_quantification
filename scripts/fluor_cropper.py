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

    :return ..., class containing the arguments
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
    rough_overlap_dict = {}
    min_r = mask_a.shape[0] // 2
    max_r = mask_b.shape[0] - mask_a.shape[0] // 2
    min_c = mask_a.shape[1] // 2
    max_c = mask_b.shape[1] - mask_a.shape[1] // 2
    for row in range(min_r, max_r + 1, step):
        for col in range(min_c, max_c + 1, step):
            crop = utils.crop_region(mask_b, (col, row), mask_a.shape)
            if crop.shape == mask_a.shape:
                overlap = (crop == mask_a).sum() / \
                          (mask_a.shape[0] * mask_a.shape[1])
                rough_overlap_dict[overlap] = (col, row)
    optimum = rough_overlap_dict[max(rough_overlap_dict.keys())]
    return optimum


def overlap_crop(mask_a, mask_b, full_image, return_coords=False):
    rough_cen = rough_crop(mask_a, mask_b, 100)
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
    full_image = np.pad(full_image, 500)
    final_crop = utils.crop_region(
        full_image,
        (int(rough_cen[0] - shift[1] + 500),
         int(rough_cen[1] - shift[0] + 500)),
        (1500, 1500)
    )
    if return_coords:
        return(final_crop, (int(rough_cen[0] - shift[1]),
                            int(rough_cen[1] - shift[0])))
    return final_crop


def pool_handler(cores, fun, params):
    pools = Pool(cores)
    pools.map(fun, params)


def worker(arg_tup):
    rgb_crop, fluor_dir, outdir, diag = arg_tup
    rgb_fn = rgb_crop.split("/")[-1]
    pos = rgb_fn.split("_")[-2]
    ident = "-".join(rgb_fn.split("-")[:3])
    fluor_files = os.listdir(fluor_dir)
    fm_match = [file for file in fluor_files if
                file.startswith(ident) and file.endswith("-Fm.fimg")][0]
    fvfm_match = [file for file in fluor_files if
                  file.startswith(ident) and file.endswith("-Fv_Fm.fimg")][0]
    rgb_im = io.imread(rgb_crop)
    fm_im = utils.read_fimg(fluor_dir + "/" + fm_match)
    fvfm_im = utils.read_fimg(fluor_dir + "/" + fvfm_match)
    fm_im = transform.resize(fm_im, (2823, 3750))
    fvfm_im = transform.resize(fvfm_im, (2823, 3750))
    rgb_mask = segment.shw_segmentation(rgb_im)
    fm_mask = fm_im > filters.threshold_otsu(fm_im)
    fm_crop, centre = overlap_crop(rgb_mask, fm_mask, fm_im, True)
    fvfm_im = np.pad(fvfm_im, 500)
    fvfm_crop = utils.crop_region(
        fvfm_im,
        (centre[0] + 500, centre[1] + 500),
        shape=(1500, 1500)
    )
    np.save(outdir + "/" + rgb_fn.replace(".png", "_Fm"), fm_crop)
    np.save(outdir + "/" + rgb_fn.replace(".png", "_FvFm"), fvfm_crop)
    if diag:
        if not os.path.isdir(outdir + "/diagnostic"):
            os.mkdir(outdir + "/diagnostic")
        plt.imsave(outdir + "/diagnostic/" + rgb_fn.replace(
            ".png", "_Fm.png"),
                   fm_crop)
        plt.imsave(outdir + "/diagnostic/" + rgb_fn.replace(
            ".png", "_FvFm.png"),
                   fvfm_crop)


def main():
    # Read arguments
    args = arg_reader()
    # Create out_dir if not existing
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # Create list of files
    files = [args.rgb_path + "/" + file for file in os.listdir(args.rgb_path)]
    # Create list of parameters
    params = zip(files, [args.fluor_path] * len(files), [args.out] * len(files),
                 [args.d] * len(files))
    # Send to pool handler
    pool_handler(args.c, worker, params)



if __name__ == "__main__":
    main()
