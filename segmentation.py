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


if __name__ == "__main__":
    image = io.imread("test_images/tb_snap.png")
    elevation = elevation_map(image)
    plt.imshow(elevation)
    plt.show()
