import segment
import argparse as arg
import os
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import utils


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
                                        "pre_existing.")
    arg_parser.add_argument(
        "seeds",
        help="The number of starting seeds for the watershed segmentation",
        type=int
    )
    arg_parser.add_argument(
        "-hb",
        help="The hue threshold of the background, every pixel where the hue is"
             "below this will be marked as 0. Hue is given in a range between "
             "0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-hp",
        help="The hue threshold of the phenotype, every pixel where the hue is"
             "below this but above the background thresh is marked as 2. Hue "
             "is given in a range between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-sb",
        help="The saturation threshold of the background, every pixel where "
             "the saturation is below this will be marked as 0. saturation is "
             "given in a range between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-sp",
        help="The saturation threshold of the phenotype, every pixel where "
             "the saturation is below this but above the background thresh is "
             "marked as 2. Saturation is given in a range between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-vb",
        help="The value threshold of the background, every pixel where "
             "the value is below this will be marked as 0. Value is "
             "given in a range between 0 and 1.",
        type=float, default=0.0
    )
    arg_parser.add_argument(
        "-vp",
        help="The value threshold of the phenotype, every pixel where "
             "the value is below this but above the background thresh is "
             "marked as 2. Value is given in a range between 0 and 1.",
        type=float, default=0.0
    )
    return arg_parser.parse_args()


def main():
    args = arg_reader()
    if os.path.isdir(args.filename):
        files = os.listdir(args.filename)
        for file in files:
            try:
                image = io.imread(args.filename + "/" + file)
            except:
                print(f"Could not open {args.filename + '/' + file}")
            else:
                bg_mask = segment.water_hsv_thresh(
                    rgb_im=image, n_seeds=args.seeds, h_th=args.hb,
                    s_th=args.sb, v_th=args.vb
                )
                footprint = morphology.disk(radius=1.5)
                bg_mask = morphology.erosion(
                    image=bg_mask, footprint=footprint
                )
                hsv_im = color.rgb2hsv(image)
                pheno_mask = utils.threshold_between(
                    image=hsv_im, x_low=args.hp, y_low=args.sp, z_low=args.vp
                )
                comb_mask = segment.merge_masks(bg_mask, pheno_mask)
                if not os.path.isdir(args.out):
                    os.mkdir(args.out)
                plt.imsave(
                    args.out + "/" + file.replace(".png", "_mask.png"),
                    arr=comb_mask, cmap="gray", vmin=0, vmax=2
                )


    else:
        raise Exception("Directory not found")


if __name__ == "__main__":
    main()
