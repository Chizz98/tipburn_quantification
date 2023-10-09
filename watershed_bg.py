"""
Author: Chris Dijkstra
Date: 8/10/2023


"""

from skimage import io, color, filters, segmentation, util
import matplotlib.pyplot as plt
import numpy as np
from time import time


def binary_mask(rgb_im, n_seeds, h_thresh=0.0, s_thresh=0.0, v_thresh=0.0,
                show_steps=False):
    """ Creates a binary mask based of an image on hue and value thresholds

    :param rgb_im: numpy.ndarray, 3 dimensional array representing an RGB image
    :param n_seeds: int, the number of points in the grid used to seed the
        watershed segmentation step.
    :param h_thresh: float, hue value below which the background lies, 0 by
        default.
    :param s_thresh: float, saturation value below which the background lies, 0
        by default.
    :param v_thresh: float, value value below which the background lies, 0 by
        default.
    :param show_steps: boolean, if True, shows the intermediate steps in a
        matplotlib popup.
    :return: numpy.ndarray, 2D binary mask with same width and height as the
        input. 0 represents the background and 1 the foreground.
    """
    # Create elevation map
    comp_sobel = filters.sobel(rgb_im)
    comp_sobel = comp_sobel[:, :, 0] + comp_sobel[:, :, 1] + comp_sobel[:, :, 2]
    elevation = filters.sobel(comp_sobel)
    # Create grid
    grid = util.regular_grid(rgb_im[:, :, 0].shape, n_seeds)
    seeds = np.zeros_like(elevation)
    seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
    # Watershed segmentation
    labs_1 = segmentation.watershed(elevation, seeds)
    average_cols = color.label2rgb(labs_1, rgb_im, kind="avg")
    # Create mask based on hsv
    average_hsv = color.rgb2hsv(average_cols)
    mask = np.ones_like(average_hsv[:, :, 0])
    mask[average_hsv[:, :, 0] < h_thresh] = 0
    mask[average_hsv[:, :, 1] < s_thresh] = 0
    mask[average_hsv[:, :, 2] < v_thresh] = 0
    # Visualization
    if show_steps:
        _, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        axes[0, 0].imshow(rgb_im)
        axes[0, 0].title.set_text("1: Original image")
        axes[0, 1].imshow(comp_sobel, cmap="inferno")
        axes[0, 1].title.set_text("2: Compound sobel")
        axes[0, 2].imshow(elevation, cmap="inferno")
        axes[0, 2].title.set_text("3: Elevation map")
        axes[0, 3].imshow(average_cols)
        axes[0, 3].title.set_text("4: Watershed segmentation")
        axes[1, 0].imshow(average_hsv[:, :, 0], cmap="hsv")
        axes[1, 0].title.set_text("5a: Hue")
        axes[1, 1].imshow(average_hsv[:, :, 1])
        axes[1, 1].title.set_text("5b: Saturation")
        axes[1, 2].imshow(average_hsv[:, :, 2])
        axes[1, 2].title.set_text("5c: Value")
        axes[1, 3].imshow(mask)
        axes[1, 3].title.set_text("6: Mask")
        for ax in axes.ravel():
            ax.axis("off")
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0,
                            wspace=0)
        plt.show()
    return mask


if __name__ == "__main__":
    start = time()

    image = io.imread("test_images/phenovator.png")
    if image.shape[2] == 4:
        rgb_im = color.rgba2rgb(image)
    bin_mask = binary_mask(image, 1500, s_thresh=0.3, v_thresh=0.2,
                           show_steps=True)

    end = int(time())
    print(f"Time taken:\nminutes: {(end - start) // 60}"
          f"\nseconds: {(end - start) % 60}")

    plt.imshow(bin_mask)
    plt.axis("off")
    plt.show()
