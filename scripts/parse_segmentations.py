import tipburn_segmentation
import argparse as arg
import os


def arg_reader():
    """ Reads arguments from command line

    :return class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Segments rgb images based on predetermined thresholds"
    )
    arg_parser.add_argument("filename", help="The path to the directory holding"
                                             " the images")
    arg_parser.add_argument("out", help="The directory to write the pixel table"
                                        "to. Does not need to be "
                                        "pre_existing.")
    return arg_parser.parse_args()


def main():
    args = arg_reader()
    files = [args.filename + "/" + file for file in os.listdir(args.filename)]
    tipburn_segmentation.parse_segmentations(files, args.out)


if __name__ == "__main__":
    main()
