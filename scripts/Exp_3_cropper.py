import utils
import matplotlib.pyplot as plt
import os
from skimage import io
import re
import numpy as np
import argparse as arg


def arg_reader():
    """ Reads arguments from command line

    :return Object containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Segments rgb images based on predetermined thresholds"
    )
    arg_parser.add_argument("im_dir", help="The path to the directory holding"
                                             " the images")
    arg_parser.add_argument("out", help="The directory to write the files "
                                        "to. Does not need to be "
                                        "pre_existing.")
    arg_parser.add_argument("coord_file", help="File containing the coordinates"
                                               "of each plant")
    return arg_parser.parse_args()


def parse_coords(infile):
    out_dict = {}
    with open(infile) as layout_file:
        header = layout_file.readline().strip().split("\t")
        for line in layout_file:
            line = line.strip().split("\t")
            data_dict = {head: val for head, val in zip(header, line)}
            head = int(data_dict["Camerahead"])
            if head in out_dict:
                out_dict[head] += [data_dict]
            else:
                out_dict[head] = [data_dict]
    return out_dict


def crop_worker(image_path, out_path, tray_dict):
    pattern = re.compile(
        r"uEye(?P<camera>[0-9]+) (?P<year>[0-9]+)-(?P<month>[0-9]+)-"
        r"(?P<day>[0-9]+)--(?P<hour>[0-9]+)-.*"
    )
    files = [os.path.join(image_path, file) for file in os.listdir(image_path)]
    out_fn_format = "{}_Tray{}_Pos{}_Camera{}_{}-{}-{}_{}h.jpg"

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for file in files:
        match = pattern.search(file)
        out_dir = os.path.join(
            out_path,
            "-".join(
                [match.group("day"),
                 match.group("month"),
                 match.group("year")]
            )
        )
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        plants = tray_dict[int(match.group("camera"))]
        image = io.imread(file)
        pad = 500
        image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        for plant in plants:
            x = int(plant["x"]) + pad
            y = int(plant["y"]) + pad
            crop = utils.crop_region(
                image=image,
                centre=(x, y),
                shape=(1000, 1000)
            )
            io.imsave(
                os.path.join(
                    out_dir,
                    out_fn_format.format(
                        plant["Accession"],
                        plant["Tray_num"],
                        plant["Plant_pos"],
                        match.group("camera"),
                        match.group("day"),
                        match.group("month"),
                        match.group("year"),
                        match.group("hour")
                    )),
                crop
            )


def main():
    args = arg_reader()

    tray_dict = parse_coords(args.coord_file)
    crop_worker(args.im_dir, args.out, tray_dict)


if __name__ == "__main__":
    main()
