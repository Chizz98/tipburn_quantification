#!/usr/bin/env python3
from skimage import io, morphology
import utils
import os
import re
import matplotlib.pyplot as plt


def parse_segmentations(im_dir, out_dir):
    """ Function to parse segmentation images

    :param im_dir: str, the directory containing the RGB images matching
        the segmentation masks in out_dir
    :param out_dir: str, the directory to write the pixel table, also contains
        the segmentation masks as grayscale images
    :return: None, writes a tab separated file
    """
    filename_pattern = re.compile(
        r"(?P<Accession>LK[0-9]+)_Tray(?P<Tray>[0-9]+)_Pos(?P<Pos>[0-9]+)_"
        r"Camera(?P<Camera>[0-9]+)_(?P<Date>[0-9]+-[0-9]+-[0-9]+)_"
        r"(?P<Hour>[0-9]+).+"
    )
    outfile = open(out_dir + "/pixel_table.txt", "w")
    outfile.write("\t".join(
        ["Accession", "Tray", "Pos", "Camera", "Date", "Hour", "Healhy",
         "Brown"]) + "\n"
    )
    count = 0
    image_files = os.listdir(im_dir)
    for file in image_files:
        match = filename_pattern.search(file)
        if match:
            exp_dat = list(match.groups())
        else:
            exp_dat = ["NA"] * 6
        mask_fn = os.path.join(
            out_dir,
            file.split("/")[-1].replace(".jpg", ".png")
        )
        try:
            mask = io.imread(mask_fn, as_gray=True) // 0.5
        except:
            exp_dat += ["NA", "NA"]
        else:
            healthy = (mask == 1).sum()
            brown = (mask == 2).sum()
            exp_dat += [healthy, brown]
        exp_dat = [str(dat) for dat in exp_dat]
        outfile.write("\t".join(exp_dat) + "\n")
        count += 1
        print(f"File {count} of {len(image_files)} parsed")
    outfile.close()


def main():
    parse_segmentations(
        r"E:\Mscthesis\Exp3\crops\25-01-2024",
        r"E:\Mscthesis\Exp3\25-01-2024_s2.5_div3.5_nd"
    )


if __name__ == "__main__":
    main()

