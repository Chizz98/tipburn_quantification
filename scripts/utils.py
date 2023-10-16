#!/usr/bin/env python3
"""
Author: Chris Dijkstra
Date: 11/10/2023

Utility functions for image analysis
"""
import numpy as np


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
    return image

