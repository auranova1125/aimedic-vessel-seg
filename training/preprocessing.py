"""Image preprocessing shared by RCA dataset preparation and inference."""

from __future__ import annotations

import cv2
import numpy as np


def equalize_luminance(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization without changing a BGR image's color data."""
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or three-channel BGR image.")

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])
    return cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an image into the uint8 range while preserving all-zero inputs."""
    max_value = np.max(image)
    if max_value == 0:
        return image.astype(np.uint8)
    return (image / max_value * 255).astype(np.uint8)


def prepare_input_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize and equalize an angiography frame for the segmentation model."""
    resized_image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_AREA)
    return equalize_luminance(normalize_to_uint8(resized_image))
