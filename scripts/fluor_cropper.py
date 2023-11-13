from skimage import io, measure, transform, segmentation, color, \
    util, registration
import utils
import segment
import barb_phenotyping
import matplotlib.pyplot as plt
import numpy as np


def rough_crop(mask_a, mask_b, step):
    rough_overlap_dict = {}
    min_r = mask_a.shape[0] // 2
    max_r = mask_b.shape[0] - mask_a.shape[0] // 2
    min_c = mask_a.shape[1] // 2
    max_c = mask_b.shape[1] - mask_a.shape[1] // 2
    for row in range(min_r, max_r + 1, step):
        for col in range(min_c, max_c + 1, step):
            crop = utils.crop_region(mask_b, (col, row), mask_a.shape)
            if crop.shape == mask_a.shape:
                overlap = (crop == mask_a).sum() / \
                          (mask_a.shape[0] * mask_a.shape[1])
                rough_overlap_dict[overlap] = (col, row)
    optimum = rough_overlap_dict[max(rough_overlap_dict.keys())]
    return optimum


def overlap_crop(mask_a, mask_b, full_image):
    rough_cen = rough_crop(mask_a, mask_b, 100)
    crop = utils.crop_region(full_image, rough_cen, (1500, 1500))
    crop_mask = crop > 3 * 10 ** 3
    rgb_labs = measure.label(mask_a)
    crop_labs = measure.label(crop_mask)
    rgb_mask = rgb_labs == utils.centre_primary_label(rgb_labs)
    crop_mask = crop_labs == utils.centre_primary_label(crop_labs)
    shift, _, _ = registration.phase_cross_correlation(
        reference_image=rgb_mask,
        moving_image=crop_mask
    )
    final_crop = utils.crop_region(
        full_image,
        (int(rough_cen[0] - shift[1]), int(rough_cen[1] - shift[0])),
        (1500, 1500)
    )
    return final_crop


def main():
    fm_im = utils.read_fimg("../test_images/51-78-Lettuce_Correct_Tray_054-"
                            "FC1-FcParamImage-Fm.fimg")
    fm_resized = transform.resize(fm_im, (2823, 3750))
    fm_mask = fm_resized > 3 * 10 ** 3

    rgb_im = io.imread("../test_images/51-78-Lettuce_Correct_Tray_054"
                       "-RGB-Original_pos5_LK190.png")
    rgb_mask = segment.shw_segmentation(rgb_im)

    fm_crop = overlap_crop(rgb_mask, fm_mask, fm_resized)


if __name__ == "__main__":
    main()
