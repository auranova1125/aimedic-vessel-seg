"""Build NumPy image and label volumes from per-slice case directories."""

import glob
import os
import re

import cv2
import numpy as np
from tqdm import tqdm


def build_image_stacks(split):
    """Store the grayscale ``Series`` images of each case as ``img.npy``."""
    for case_directory in tqdm(glob.glob("dataset/{}/*".format(split))):
        image_directory = case_directory + "/Series/"
        image_names = sort_by_numeric_suffix(os.listdir(image_directory))
        image_stack = np.zeros((512, 512, len(image_names)), dtype=np.uint8)

        for index, image_name in enumerate(image_names):
            image_stack[:, :, index] = cv2.imread(
                image_directory + image_name,
                cv2.IMREAD_GRAYSCALE,
            )

        np.save(case_directory + "/img.npy", image_stack)


def build_label_stacks(split):
    """Store artery and vein labels of each case as a compact ``gt.npy`` volume.

    A pixel is encoded as 0 for background, 1 for coronary artery, and 2 for vein.
    Pixels present in both source masks remain background, as in the original rule.
    """
    for case_directory in tqdm(glob.glob("dataset/{}/*".format(split))):
        artery_directory = case_directory + "/GT/"
        vein_directory = case_directory + "/vein/"
        artery_names = sort_by_numeric_suffix(os.listdir(artery_directory))
        vein_names = sort_by_numeric_suffix(os.listdir(vein_directory))
        label_stack = np.zeros((512, 512, len(artery_names)), dtype=np.uint8)

        for index, artery_name in enumerate(artery_names):
            vein_name = vein_names[index]
            artery_image = cv2.imread(artery_directory + artery_name)
            vein_image = cv2.imread(vein_directory + vein_name)
            artery_mask = (artery_image[:, :, 0] != artery_image[:, :, 1]).astype(
                np.uint8
            )
            vein_mask = (vein_image[:, :, 0] != vein_image[:, :, 1]).astype(
                np.uint8
            )
            label_stack[:, :, index] = (
                (artery_mask == 1) * (vein_mask == 0)
                + (vein_mask == 1) * (artery_mask == 0) * 2
            ).astype(np.uint8)

        np.save(case_directory + "/gt.npy", label_stack)


def sort_by_numeric_suffix(file_names):
    """Sort slice names using the first integer contained in each file name."""
    return sorted(
        file_names,
        key=lambda file_name: int(re.findall(r"\d+", file_name)[0]),
    )


if __name__ == "__main__":
    for dataset_split in ("train", "val", "test"):
        build_image_stacks(dataset_split)
        build_label_stacks(dataset_split)
