from skimage import io, segmentation, morphology, util, color, measure, filters
from scipy import signal
import matplotlib.pyplot as plt
import segment
import os
import numpy as np
import utils


def canny_central_ob(image, mask, sigma):
    """ Uses canny filter and color channel thresholding to take central object

    :param image: np.ndarray, 3d array representing rgb image
    :param mask: np.ndarray, 2d boolean array representing background mask
    :param sigma: float, sigma used for gaussian blur step of canny edge
        detection
    :return np.ndarray, 2d binary mask of central object
    """
    bg_labs = measure.label(mask)
    mask = bg_labs == utils.centre_primary_label(bg_labs)
    canny_labelled = utils.canny_labs(color.rgb2gray(image), mask, sigma)
    prim_lab = utils.centre_primary_label(canny_labelled)
    average_cols = color.label2rgb(canny_labelled, image, kind="avg")
    average_cols = color.rgb2hsv(average_cols)
    prim_area = utils.multichannel_mask(average_cols,
                                        canny_labelled == prim_lab)
    h_main = np.unique(prim_area[:, :, 0])[1]
    s_main = np.unique(prim_area[:, :, 1])[1]
    v_main = np.unique(prim_area[:, :, 2])[1]
    mask = utils.threshold_between(
        image=average_cols,
        x_low=h_main - 0.1, x_high=h_main + 0.1,
        y_low=s_main - 0.25, y_high=s_main + 0.25,
        z_low=v_main - 0.25, z_high=v_main + 0.25
    )
    return mask


def shw_segmentation(image):
    """ Creates binary image through sobel + histogram thresholds + watershed

    :param image: np.ndarray representing a 3d image
    :return np.ndarray, 2D mask for the image
    """
    if image.shape[2] == 4:
        image = util.img_as_ubyte(color.rgba2rgb(image))
    comp_sob = filters.sobel(image)
    comp_sob = comp_sob[:, :, 0] + comp_sob[:, :, 1] + comp_sob[:, :, 2]
    elevation = filters.sobel(comp_sob)
    values, bins = np.histogram(comp_sob, bins=100)
    max_i, _ = signal.find_peaks(values, distance=10)
    max_bins = bins[max_i]
    min_i, _ = signal.find_peaks(-values, distance=10)
    min_bins = bins[min_i]
    min_bins = min_bins[min_bins > max_bins[0]]
    markers = np.zeros_like(comp_sob)

    markers[
        comp_sob >= min_bins[0] +
        (0.1 * (max_bins[1] - min_bins[0]))] = 2
    markers[
        morphology.erosion(comp_sob <= max_bins[0] +
                           (0.15 * (min_bins[0] - max_bins[0])),
                           footprint=morphology.disk(4)) |
        (color.rgb2hsv(image)[:, :, 0] > 0.45)] = 1

    plt.imshow(markers)
    plt.show()

    mask = segmentation.watershed(elevation, markers)
    mask = morphology.erosion(mask, footprint=morphology.disk(2))
    return mask - 1


def main():
    indir = r"C:\Users\chris\Documents\BPW\Potato\crops"
    outdir = r"C:\Users\chris\Documents\BPW\Potato\output"

    if not os.path.isdir(indir):
        raise Exception("Input directory does not exist")
    infiles = [indir + "/" + file for file in os.listdir(indir)]

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for file in infiles:
        image = io.imread(file)
        if image.shape[2] == 4:
            image = util.img_as_ubyte(color.rgba2rgb(image))
        bg_mask = shw_segmentation(image)
        bg_mask = morphology.closing(bg_mask, footprint=morphology.disk(3.5))
        bg_mask = canny_central_ob(image, bg_mask, 1.5)
        bg_mask = morphology.opening(bg_mask, footprint=morphology.disk(1))
        name = file.split("/")[-1]
        plt.imsave(
            fname=outdir + "/" + name.replace(".png", "_masked.png"),
            arr=utils.multichannel_mask(image, bg_mask)
        )


if __name__ == "__main__":
    main()
