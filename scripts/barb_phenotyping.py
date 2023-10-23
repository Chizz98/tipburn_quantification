from skimage import io, color, morphology, util, measure, feature
import matplotlib.pyplot as plt
import segment
import utils
import numpy as np
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
    arg_parser.add_argument("out", help="The directory to write the files "
                                        "to. Does not need to be "
                                        "pre_existing. ../out as default.")
    arg_parser.add_argument("-s", help="Sigma used for canny edge detection. "
                                       "Used to remove high contrast bordering"
                                       "objects.",
                            type=float, default=3.0)
    return arg_parser.parse_args()


def barb_thresh(im_channel):
    """ Defines the threshold of an image channel based on its histogram

      :param im_channel: np.ndarray, 2d array, meant to be hue channel of hsv or
          a channel of lab
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
    thresh = 2 * ref_bin / 3.5
    return thresh


def barb_hue(image, bg_mask=None):
    """ Takes an image of plant tissue and segments into healthy and brown

      :param image: np.ndarray, 3d array representing an rgb image
      :param bg_mask: np.ndarray, 2d array to mask the background
      :return np.ndarray, mask with background as 0, healthy tissue as 1 and
          brown tissue as 2
    """
    if bg_mask is not None:
        # Apply mask to rgb_im
        rgb_im = utils.multichannel_mask(image, bg_mask)
    # Get hue channel and scale from 0 to 1
    hue = color.rgb2hsv(image)[:, :, 0]
    hue_con = utils.increase_contrast(hue)
    hue_fg = hue_con[bg_mask == 1]
    # Healthy tissue masking
    thresh = barb_thresh(hue_fg)
    healthy_mask = (hue_con > thresh).astype(int)
    # Combine healthy and bg mask to get compound image
    bg_mask = bg_mask.astype(int)
    comp_mask = segment.merge_masks(bg_mask, healthy_mask)
    return comp_mask


def main():
    args = arg_reader()
    directory = args.filename
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    out_table = open(args.out + "/pixel_table.txt", "w")
    header = "\t".join([
        "Experiment", "Round", "Tray", "Position", "Accession", "Healthy",
        "Brown\n"
    ])
    out_table.write(header)
    if os.path.isdir(directory):
        files = os.listdir(directory)
        for file in files:
            try:
                rgb_im = io.imread(directory + "/" + file)
                if rgb_im.shape[2] == 4:
                    rgb_im = util.img_as_ubyte(color.rgba2rgb(rgb_im))
            except:
                print(f"Could not open {directory + '/' + file}")
            else:
                # Create bg_mask
                bg_mask = segment.shw_segmentation(rgb_im)
                # Only keep centre object
                bg_mask = utils.canny_central_ob(rgb_im, bg_mask, 3)
                # Remove lines
                bg_mask = morphology.opening(bg_mask,
                                             footprint=np.ones((5, 10)))
                bg_mask = morphology.opening(bg_mask,
                                             footprint=np.ones((10, 5)))
                # Create compound array
                comp_im = barb_hue(rgb_im, bg_mask)
                # Write image
                bg = utils.multichannel_mask(rgb_im, comp_im == 0)
                fg = utils.multichannel_mask(rgb_im, comp_im > 0)
                tb = utils.multichannel_mask(rgb_im, comp_im == 2)
                plt.imsave(args.out + "/" + file.replace(".png", "_bg.png"),
                           arr=bg)
                plt.imsave(args.out + "/" + file.replace(".png", "_fg.png"),
                           arr=fg)
                plt.imsave(args.out + "/" + file.replace(".png", "_tb.png"),
                           arr=tb)
                # Write table
                file_split = file.split("-")
                experiment = file_split[0]
                ex_round = file_split[1]
                tray = file_split[2].split("_")[-1]
                pos = file_split[4].split("_")[1]
                accession = file_split[4].split("_")[2].replace(".png", "")
                healthy = str((comp_im == 1).sum())
                brown = str((comp_im == 2).sum()) + "\n"
                out_table.write("\t".join([experiment, ex_round, tray, pos,
                                           accession, healthy, brown]))
    out_table.close()


if __name__ == "__main__":
    main()
