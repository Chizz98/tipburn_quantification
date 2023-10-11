"""
Author: Chris Dijkstra
Date: 11/10/2023
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
                centre[0] - shape_r[0] // 2: centre[0] + shape[0] // 2,
                centre[1] - shape_r[1] // 2: centre[1] + shape[1] // 2
        ]
    else:
        crop = image[
                centre[0] - shape_r[0] // 2: centre[0] + shape[0] // 2,
                centre[1] - shape_r[1] // 2: centre[1] + shape[1] // 2,
                :
        ]
    return crop


def main():
    """ Test function """
    test = np.array([
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10)),
        list(range(1, 10))
    ])
    print(test)
    crop = crop_region(test, (5, 5), (2, 6))
    print(crop)


if __name__ == "__main__":
    main()
