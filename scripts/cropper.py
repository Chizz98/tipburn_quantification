from skimage import io, color
import utils
import matplotlib.pyplot as plt
import argparse as arg
import os
import re


def arg_reader():
    """ Reads arguments from command line

    :return ..., class containing the arguments
    """
    arg_parser = arg.ArgumentParser(
        description="Crops phenovator files"
    )
    arg_parser.add_argument("filename", help="The path to the directory holding"
                                             " the images")
    arg_parser.add_argument("out", help="The directory to write the files "
                                        "to. Does not need to be "
                                        "pre_existing.")
    arg_parser.add_argument("trayfile", help="The .csv containing the plant "
                                             "positions on each tray")
    arg_parser.add_argument("-v", "--vocal", help="Makes script print progress",
                            action="store_true"
                            )
    return arg_parser.parse_args()


def crop_images(image, poslist, shape):
    poslist.sort()
    coord_dict = {
        1: (1330, 810),
        2: (2059, 810),
        3: (2790, 810),
        4: (1330, 1530),
        5: (2059, 1530),
        6: (2790, 1530),
        7: (1330, 2260),
        8: (2059, 2260),
        9: (2790, 2260)
    }
    croplist = []
    for pos in poslist:
        croplist.append(utils.crop_region(image, coord_dict[pos], shape))
    return croplist


def parse_trayfile(filename):
    acc_dict = {}
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split(";")
            tray_num = int(line[0].split("_")[-1])
            pos = line[2].split()[-1]
            acc = line[4]
            if tray_num not in acc_dict:
                acc_dict[tray_num] = [(pos, acc)]
            else:
                acc_dict[tray_num].append((pos, acc))
    return acc_dict


def main():
    args = arg_reader()
    acc_inf = parse_trayfile(args.trayfile)
    tray_pattern = re.compile(r"Tray_0(\d+)")
    filecount = 0
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    if os.path.isdir(args.filename):
        files = [file for file in os.listdir(args.filename) if
                 file.find("Original") != -1 and file.startswith("51-40")]
        tot_files = len(files)
        for file in files:
            try:
                image = io.imread(args.filename + "/" + file)
            except:
                print(f"Could not open {args.filename + '/' + file}")
            else:
                if image.shape[2] == 4:
                    image = color.rgba2rgb(image)
                match = tray_pattern.search(file)
                tray_num = int(match.group(1))
                accs = acc_inf[tray_num]
                if len(accs) == 5:
                    positions = [1, 3, 5, 7, 9]
                else:
                    positions = [2, 4, 6, 8]
                crops = crop_images(image, positions, (1500, 1500))
                for crop, acc_dat in zip(crops, accs):
                    pos, acc = acc_dat
                    out_fn = args.out + "/" + file.replace(
                        ".png", f"_pos{pos}_{acc}.png")
                    plt.imsave(fname=out_fn, arr=crop)
            if args.vocal:
                filecount += 1
                print(f"File {filecount} of {tot_files}")


if __name__ == "__main__":
    main()
