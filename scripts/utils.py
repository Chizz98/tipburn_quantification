#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 11/10/2023

Utility functions for image analysis
"""
import numpy as np
from skimage import feature, measure, morphology, color


def crop_region(image, centre, shape):
    """ Crops an image area of specified width and height around a central point

    :param image: np.ndarray, matrix representing the image
    :param centre: tuple, contains the x and y coordinate of the centre as
        integers
    :param shape: tuple, contains the height and width of the subregion in
        pixels as integers
    :return: The cropped region of the original image
    """
    shape_r = np.array(shape)
    shape_r[shape_r % 2 == 1] += 1
    if image.ndim == 2:
        crop = image[
               centre[1] - shape_r[1] // 2: centre[1] + shape[1] // 2,
               centre[0] - shape_r[0] // 2: centre[0] + shape[0] // 2
               ]
    else:
        crop = image[
               centre[1] - shape_r[1] // 2: centre[1] + shape[1] // 2,
               centre[0] - shape_r[0] // 2: centre[0] + shape[0] // 2,
               :
               ]
    return crop


def read_fimg(filename):
    """ Turns an FIMG value into a normalized file with data between 0 and 1

    :param filename: str, name of the file that is to be opened
    :return np.ndarray, 2D array representing the fimg image
    """
    image = np.fromfile(filename, np.dtype("float32"))
    image = image[2:]
    image = np.reshape(image, newshape=(1024, 1360))
    image[image < 0] = 0
    return image


def threshold_between(image, x_low=None, x_high=None, y_low=None, y_high=None,
                      z_low=None, z_high=None, and_mask=True):
    """ Thresholds an image array for being between two values for each channels

    :param image: np.ndarray, 3d matrix representing the image
    :param x_low: low boundary for channel 1. Defaults to minimum of channel 1.
    :param x_high: high boundary for channel 1. Defaults to maximum of channel
        1.
    :param y_low: low boundary for channel 2. Defaults to minimum of channel 2.
    :param y_high: high boundary for channel 2. Defaults to maximum of channel
        2.
    :param z_low: low boundary for channel 3. Defaults to minimum of channel 3.
    :param z_high: high boundary for channel 3. Defaults to maximum of channel
        3.
    :param and_mask: bool, if true returned mask is only true when all
        thresholds apply. If false returned mask is true if at least one of the
        thresholds apply.
    :return np.ndarray, binary mask
    """
    # channel x thresholding
    if not x_low:
        x_low = image[:, :, 0].min()
    if not x_high:
        x_high = image[:, :, 0].max()
    x_mask = (image[:, :, 0] >= x_low) & (image[:, :, 0] <= x_high)
    # channel y thresholding
    if not y_low:
        y_low = image[:, :, 1].min()
    if not y_high:
        y_high = image[:, :, 1].max()
    y_mask = (image[:, :, 1] >= y_low) & (image[:, :, 1] <= y_high)
    # channel z thresholding
    if not z_low:
        z_low = image[:, :, 2].min()
    if not z_high:
        z_high = image[:, :, 2].max()
    z_mask = (image[:, :, 2] >= z_low) & (image[:, :, 2] <= z_high)
    if and_mask:
        comp_mask = x_mask & y_mask & z_mask
    else:
        comp_mask = x_mask | y_mask | z_mask
    return comp_mask


def increase_contrast(im_channel):
    """ Takes a 2d array and makes its values range from 0 to 1

    :param im_channel: np.ndarray, numpy array with 2 dimensions
    :return np.ndarray: input channel scaled from 0 to 1.
    """
    ch_min = im_channel.min()
    ch_max = im_channel.max()
    out = (im_channel - ch_min) / (ch_max - ch_min)
    return out


def multichannel_mask(image, mask):
    """ Takes an image and applies a mask to every channel

    :param image, np.dnarray, 3 dimensional array representing an image
    :param mask, np.ndarray, 2d binary mask
    :return np.ndarray, masked input image
    """
    mask = mask.astype(image.dtype)
    image = image.copy()
    image[:, :, 0] *= mask
    image[:, :, 1] *= mask
    image[:, :, 2] *= mask
    return image


def canny_labs(image, mask, sigma):
    """ Separates objects trough canny lines and then labels the output

    :param image: np.dnarray, 2d array representing an image
    :param mask: np.ndarray, 2d binary mask
    :param sigma: float, the sigma used for the gaussian blur component of canny
        segmentation
    :return np.ndarray, labelled image
    """
    canny_f = feature.canny(image, sigma=sigma)
    canny_f = morphology.dilation(canny_f)
    mask = mask.copy()
    mask[canny_f == 1] = 0
    labels = measure.label(mask, connectivity=1)
    return labels


def centre_primary_label(lab_im, radius=50):
    """ Takes labelled image and returns the label of the central object

    :param lab_im: np.ndarray, labelled image with only positive values and 0
    :param radius: int, height and width of the square used on the centre
    :return: int, primary label
    """
    centre = (lab_im.shape[0] // 2, lab_im.shape[1] // 2)
    crop = crop_region(lab_im, centre, (radius, radius))
    return np.argmax(np.bincount(crop.ravel()))


def canny_central_ob(image, mask, sigma):
    """ Uses canny filter and color channel thresholding to take central object

    :param image: np.ndarray, 3d array representing rgb image
    :param mask: np.ndarray, 2d boolean array representing background mask
    :param sigma: float, sigma used for gaussian blur step of canny edge
        detection
    :return np.ndarray, 2d binary mask of central object
    """
    bg_labs = measure.label(mask)
    mask = bg_labs == centre_primary_label(bg_labs, 10)
    canny_labelled = canny_labs(color.rgb2gray(image), mask, sigma)
    prim_lab = centre_primary_label(canny_labelled, 10)
    average_cols = color.label2rgb(canny_labelled, image, kind="avg")
    average_cols = color.rgb2hsv(average_cols)
    prim_area = multichannel_mask(average_cols, canny_labelled == prim_lab)
    h_main = np.unique(prim_area[:, :, 0])[1]
    s_main = np.unique(prim_area[:, :, 1])[1]
    v_main = np.unique(prim_area[:, :, 2])[1]
    mask = threshold_between(
        image=average_cols,
        x_low=h_main - 0.2, x_high=h_main + 0.2,
        y_low=s_main - 0.2, y_high=s_main + 0.2,
        z_low=v_main - 0.2, z_high=v_main + 0.2
    )
    mask = morphology.closing(mask, footprint=morphology.disk(3))
    mask = morphology.remove_small_holes(mask, area_threshold=150)
    return mask

