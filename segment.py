from skimage import io, color, filters, segmentation, util
import matplotlib.pyplot as plt
import numpy as np


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


def multichannel_threshold(multi_ch_im, x_th, y_th, z_th):
    """ Takes a three-channel image and returns a mask based on thresholds

    :param multi_ch_im: np.nd_array a numpy array representing an image with
        three color channels
    :param x_th: float, the threshold for the first channel
    :param y_th: float, the threshold for the second channel
    :param z_th: float, the threshold for the third channel
    :return: np.nd_array, the mask created based on the thresholds, 2D array
        same width and height as the input
    """
    mask = np.ones(multi_ch_im.shape[0:2])
    mask[multi_ch_im[:, :, 0] < x_th] = 0
    mask[multi_ch_im[:, :, 1] < y_th] = 0
    mask[multi_ch_im[:, :, 2] < z_th] = 0
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
    average_cols = color.label2rgb(labels, rgb_im, kind="avg")
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
    return


if __name__ == "__main__":
    image = io.imread("test_images/tb_snap.png")
    plant_mask = multichannel_threshold(image, 1, 2, 3)
