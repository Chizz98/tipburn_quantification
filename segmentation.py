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
    print(image.shape)
    print(grid_map.shape)
    grid_map[grid] = np.arange(
        grid_map[grid].size).reshape(grid_map[grid].shape) + 1
    return grid_map


if __name__ == "__main__":
    image = io.imread("test_images/tb_snap.png")
    el_map = elevation_map(image)
    seeds = map_grid(1000, image.shape[0:2])
    plt.imshow(seeds)
    plt.show()
