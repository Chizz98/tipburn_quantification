#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 10/10/2023

Contains functions for segmenting image arrays.
"""
from skimage import io, color, filters, segmentation, util, morphology
import numpy as np
from scipy import signal
import utils


def elevation_map(rgb_im):
    """ Creates an elevation map of an RGB image based on sobel filtering

    :param rgb_im: numpy.ndarray, 3 dimensional array representing an RGB image
    :return: numpy.ndarray, 2 dimensional array representing an edge map
    """
    compound_sobel = filters.sobel(rgb_im)
    compound_sobel = compound_sobel[:, :, 0] + compound_sobel[:, :, 1] + \
                     compound_sobel[:, :, 2]
    elevation = filters.sobel(compound_sobel)
    return elevation


def map_grid(n_points, shape):
    """ Creates point grid within a certain shape

    :param n_points: int, number of points that the grid should contain
    :param shape: iterable, an iterable containing the rows (int) and
        columns (int)
    :return np.ndarray, a numpy array with the specified shape, containing an
        equally spaced grid with the specified number of points
    """
    grid = util.regular_grid(shape, n_points)
    grid_map = np.zeros(shape)
    grid_map[grid] = np.arange(
        grid_map[grid].size).reshape(grid_map[grid].shape) + 1
    return grid_map


def multichannel_threshold(multi_ch_im, x_th=0.0, y_th=0.0, z_th=0.0,
                           inverse=False):
    """ Takes a three-channel image and returns a mask based on thresholds

    :param multi_ch_im: np.nd_array a numpy array representing an image with
        three color channels
    :param x_th: float, the threshold for the first channel, 0.0 by default
    :param y_th: float, the threshold for the second channel, 0.0 by default
    :param z_th: float, the threshold for the third channel, 0.0 by default
    :param inverse: bool, if False pixels below the threshold are marked as 0,
        if True, pixels above the threshold are marked as 0.
    :return: np.nd_array, the mask created based on the thresholds, 2D array
        same width and height as the input
    """
    mask = np.ones(multi_ch_im.shape[0:2])
    mask[multi_ch_im[:, :, 0] < x_th] = 0
    mask[multi_ch_im[:, :, 1] < y_th] = 0
    mask[multi_ch_im[:, :, 2] < z_th] = 0
    mask = mask.astype(int)
    if inverse:
        mask = np.invert(mask)
    return mask


def watershed_blur(rgb_im, n_seeds):
    """ Performs watershed averaging of color, preserving edges

    :param rgb_im: np.ndarray, 3 dimensional array representing an RGB image
    :param n_seeds: int, number of points that the grid should contain
    :return: np.ndarray, 3 dimensional array representing an RGB image. Colors
        are the colors of the input image averaged over watershed regions.
    """
    elevation = elevation_map(rgb_im)
    seeds = map_grid(n_seeds, rgb_im.shape[0:2])
    labels = segmentation.watershed(elevation, seeds)
    average_cols = color.label2rgb(labels, rgb_im, kind="avg",
                                   bg_label=0).astype(np.uint8)
    return average_cols


def water_hsv_thresh(rgb_im, n_seeds, h_th=0.0, s_th=0.0, v_th=0.0):
    """ Segments an image based on hsv thresholds, after watershed averaging

    :param rgb_im: np.ndarray, 3 dimensional array representing an RGB image
    :param n_seeds:
    :param h_th: float, the threshold for the hue channel, everything below this
        value is marked as background
    :param s_th: float, the threshold for the value channel, everything below
        this value is marked as background
    :param v_th: float, the threshold for the saturation channel everything
        below this value is marked as background
    :return: np.ndarray, 2D mask with the
    """
    blurred = watershed_blur(rgb_im, n_seeds)
    hsv_blurred = color.rgb2hsv(blurred)
    mask = multichannel_threshold(hsv_blurred, h_th, s_th, v_th)
    return mask.astype(int)


def sw_segmentation(image):
    """ Separates an image from the background with sobel filtering + watershed

    :param image: np.ndarray representing a 3d image
    :return np.ndarray, 2D mask for the image
    """
    comp_sob = filters.sobel(image)
    comp_sob = comp_sob[:, :, 0] + comp_sob[:, :, 1] + comp_sob[:, :, 2]
    elevation = filters.sobel(comp_sob)
    markers = np.zeros_like(comp_sob)
    markers[comp_sob <= 0.025] = 1
    markers[comp_sob >= 0.175] = 2
    mask = segmentation.watershed(elevation, markers)
    mask = morphology.erosion(mask, footprint=morphology.disk(2))
    return mask


def shw_segmentation(image, distance=10, bg_mod=0.15, fg_mod=0.2):
    """ Creates binary image through sobel + histogram thresholds + watershed

    :param image: np.ndarray representing a 3d image
    :param distance: int, minimal distance between local maxima and minima
    :param bg_mod: float, modifier for histogram segmentation
    :param fg_mod: float, modifier for histogram segmentation
    :return np.ndarray, 2D mask for the image
    """
    if image.shape[2] == 4:
        image = util.img_as_ubyte(color.rgba2rgb(image))
    comp_sob = filters.sobel(image)
    comp_sob = comp_sob[:, :, 0] + comp_sob[:, :, 1] + comp_sob[:, :, 2]
    elevation = filters.sobel(comp_sob)
    values, bins = np.histogram(comp_sob, bins=100)
    max_i, _ = signal.find_peaks(values, distance=distance)
    max_bins = bins[max_i]
    min_i, _ = signal.find_peaks(-values, distance=distance)
    min_bins = bins[min_i]
    min_bins = min_bins[min_bins > max_bins[0]]
    markers = np.zeros_like(comp_sob)
    markers[comp_sob <= max_bins[0] + (bg_mod * (min_bins[0] - max_bins[0]))] = 1
    markers[comp_sob >= min_bins[0] + (fg_mod * (max_bins[1] - min_bins[0]))] = 2
    markers = markers.astype(int)
    mask = segmentation.watershed(elevation, markers)
    mask = morphology.erosion(mask, footprint=morphology.disk(2))
    return mask - 1


def merge_masks(bg_mask, pheno_mask):
    """ Merges 2 binary masks into one mask, where phenotype has high values

    :param bg_mask: np.ndarray, 2D mask with background as 0 and foreground as
        1
    :param pheno_mask: np.ndarray, 2D mask phenotype marked as 1 and everything
        else as 0
    :return np.ndarray, 2D mask with background marked as 0, foreground as 1 and
        phenotype area as 2
    """
    substep = bg_mask.astype(int) + pheno_mask.astype(int)
    comb_mask = np.zeros_like(bg_mask)
    comb_mask[substep == 1] = 2
    comb_mask[substep == 2] = 1
    comb_mask[bg_mask == 0] = 0
    return comb_mask


def barb_thresh(im_channel, div=3):
    """ Defines the threshold of an image channel based on its histogram

    :param im_channel: np.ndarray, 2d array, meant to be hue channel of hsv or
        a channel of lab
    :param div: int, the divisor used at the end of the algorithm. A higher
        divisor will lead to a lower threshold
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
    thresh = 2 * ref_bin / div
    return thresh


def barb_hue(image, bg_mask=None, div=3):
    """ Takes an image of plant tissue and segments into healthy and brown

    :param image: np.ndarray, 3d array representing an rgb image
    :param bg_mask: np.ndarray, 2d array to mask the background
    :param div: int, the divisor used at the end of the algorithm. A higher
        divisor will lead to a lower threshold
    :return np.ndarray, mask with background as 0, healthy tissue as 1 and
        brown tissue as 2
    """
    if bg_mask is not None:
        # Apply mask to rgb_im
        image = utils.multichannel_mask(image, bg_mask)
    # Get hue channel and scale from 0 to 1
    hue = color.rgb2hsv(image)[:, :, 0]
    hue_con = utils.increase_contrast(hue)
    hue_fg = hue_con[bg_mask == 1]
    # Healthy tissue masking
    thresh = barb_thresh(hue_fg, div)
    healthy_mask = (hue_con > thresh).astype(int)
    # Remove noise
    healthy_mask = morphology.remove_small_holes(
        healthy_mask,
        area_threshold=(image.shape[0] + image.shape[1]) // 200
    )
    # Combine healthy and bg mask to get compound image
    bg_mask = bg_mask.astype(int)
    comp_mask = merge_masks(bg_mask, healthy_mask)
    return comp_mask


def main():
    """ Main function, contains a test case """
    import matplotlib.pyplot as plt
    image = io.imread("snapshots/tb_snap3.png")
    # Background segmentation
    plant_mask = water_hsv_thresh(image, 1500, s_th=0.25, v_th=0.2)
    plant_mask = morphology.binary_opening(plant_mask)
    # Tipburn segmentation
    hsv_im = color.rgb2hsv(image)
    tb_mask = multichannel_threshold(hsv_im, x_th=0.105, inverse=True)
    multi_mask = merge_masks(plant_mask, tb_mask)
    # Visualization
    plt.imshow(multi_mask)
    plt.show()


if __name__ == "__main__":
    main()
