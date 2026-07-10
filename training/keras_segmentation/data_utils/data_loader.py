"""Dataset pairing, validation, and batch generation for segmentation models."""

from __future__ import annotations

import itertools
from pathlib import Path
import random

import cv2
import numpy as np
import six

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable):
        """Return the iterable unchanged when tqdm is unavailable."""
        return iterable


from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg


CLASS_COLORS = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
DATA_LOADER_SEED = 1
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}

random.seed(DATA_LOADER_SEED)


class DataLoaderError(Exception):
    """Raised when image and segmentation data cannot be used safely."""


def _files_by_stem(directory: str, file_role: str) -> dict[str, Path]:
    """Index supported files by stem and reject ambiguous duplicates."""
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise DataLoaderError(f"Missing {file_role} directory: {directory_path}")

    files_by_stem: dict[str, Path] = {}
    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        if file_path.stem in files_by_stem:
            existing_path = files_by_stem[file_path.stem]
            raise DataLoaderError(
                f"Duplicate {file_role} filename stem '{file_path.stem}': "
                f"{existing_path.name}, {file_path.name}"
            )
        files_by_stem[file_path.stem] = file_path

    return files_by_stem


def _format_stems(stems: set[str]) -> str:
    """Format a deterministic, compact list of filename stems for errors."""
    return ", ".join(sorted(stems))


def get_class_colors(
    n_classes: int, colors: list[tuple[int, int, int]] = CLASS_COLORS
) -> list[tuple[int, int, int]]:
    """Return enough deterministic BGR colors to visualize every class."""
    resolved_colors = list(colors[:n_classes])
    for class_index in range(len(resolved_colors), n_classes):
        hue = int((class_index * 179) / max(n_classes, 1))
        hsv_color = np.uint8([[[hue, 200, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        resolved_colors.append(tuple(int(value) for value in bgr_color))
    return resolved_colors


def _load_bgr_image(image_input: np.ndarray | str) -> np.ndarray:
    """Load an array or a path and return a three-channel BGR image."""
    if isinstance(image_input, np.ndarray):
        image = image_input
    elif isinstance(image_input, six.string_types):
        image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise DataLoaderError(f"Could not read image: {image_input}")
    else:
        raise DataLoaderError(f"Unsupported image input type: {type(image_input)}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise DataLoaderError(f"Expected a 2D or 3D image, received shape {image.shape}")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise DataLoaderError(f"Expected one, three, or four image channels, received {image.shape[2]}")


def _load_mask(mask_input: np.ndarray | str) -> np.ndarray:
    """Load an array or a path and return a single-channel class-index mask."""
    if isinstance(mask_input, np.ndarray):
        mask = mask_input
    elif isinstance(mask_input, six.string_types):
        mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise DataLoaderError(f"Could not read segmentation mask: {mask_input}")
    else:
        raise DataLoaderError(f"Unsupported segmentation input type: {type(mask_input)}")

    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask[:, :, 0]
    raise DataLoaderError(f"Expected a 2D or 3D segmentation mask, received shape {mask.shape}")


def get_pairs_from_paths(
    images_path: str,
    segs_path: str,
    ignore_non_matching: bool = False,
) -> list[tuple[str, str]]:
    """Match input images and masks by filename stem.

    Strict matching is the default because an omitted image or an unused mask
    otherwise makes the training set difficult to audit.
    """
    image_files = _files_by_stem(images_path, "image")
    segmentation_files = _files_by_stem(segs_path, "segmentation")

    image_stems = set(image_files)
    segmentation_stems = set(segmentation_files)
    unmatched_images = image_stems - segmentation_stems
    unmatched_segmentations = segmentation_stems - image_stems

    if not ignore_non_matching and (unmatched_images or unmatched_segmentations):
        details = []
        if unmatched_images:
            details.append(f"images without masks: {_format_stems(unmatched_images)}")
        if unmatched_segmentations:
            details.append(f"masks without images: {_format_stems(unmatched_segmentations)}")
        raise DataLoaderError(f"Image-mask filename mismatch: {'; '.join(details)}")

    matched_stems = sorted(image_stems & segmentation_stems)
    return [(str(image_files[stem]), str(segmentation_files[stem])) for stem in matched_stems]


def get_image_array(
    image_input: np.ndarray | str,
    width: int,
    height: int,
    img_norm: str = "sub_mean",
    ordering: str = "channels_first",
    image_preprocessor=None,
) -> np.ndarray:
    """Load, resize, optionally preprocess, normalize, and order an image."""
    image = _load_bgr_image(image_input)
    image = cv2.resize(image, (width, height))

    if image_preprocessor is not None:
        image = image_preprocessor(image)

    if img_norm == "sub_and_divide":
        image = np.float32(image) / 127.5 - 1
    elif img_norm == "sub_mean":
        image = image.astype(np.float32)
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        image = image[:, :, ::-1]
    elif img_norm == "divide":
        image = image.astype(np.float32) / 255.0
    else:
        raise DataLoaderError(f"Unsupported image normalization: {img_norm}")

    if ordering == "channels_first":
        image = np.rollaxis(image, 2, 0)
    return image


def get_segmentation_array(
    image_input: np.ndarray | str,
    n_classes: int,
    width: int,
    height: int,
    no_reshape: bool = False,
) -> np.ndarray:
    """Load a class-index mask and convert it into a one-hot array."""
    mask = _load_mask(image_input)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    segmentation = np.zeros((height, width, n_classes), dtype=np.float32)
    for class_index in range(n_classes):
        segmentation[:, :, class_index] = (mask == class_index).astype(np.float32)

    if no_reshape:
        return segmentation
    return np.reshape(segmentation, (width * height, n_classes))


def image_segmentation_generator(
    images_path: str,
    segs_path: str,
    batch_size: int,
    n_classes: int,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    do_augment: bool = False,
    augmentation_name: str = "aug_all",
):
    """Yield randomized image and one-hot mask batches for model training."""
    image_mask_pairs = get_pairs_from_paths(images_path, segs_path)
    if not image_mask_pairs:
        raise DataLoaderError("No matched image-mask pairs are available for training.")

    random.shuffle(image_mask_pairs)
    pair_cycle = itertools.cycle(image_mask_pairs)

    while True:
        images = []
        segmentations = []
        for _ in range(batch_size):
            image_path, segmentation_path = next(pair_cycle)
            image = _load_bgr_image(image_path)
            segmentation = _load_mask(segmentation_path)

            if do_augment:
                image, segmentation = augment_seg(image, segmentation, augmentation_name)

            images.append(
                get_image_array(
                    image,
                    input_width,
                    input_height,
                    ordering=IMAGE_ORDERING,
                )
            )
            segmentations.append(
                get_segmentation_array(segmentation, n_classes, output_width, output_height)
            )

        yield np.array(images), np.array(segmentations)


def verify_segmentation_dataset(
    images_path: str, segs_path: str, n_classes: int, show_all_errors: bool = False
) -> bool:
    """Verify file pairing, spatial dimensions, and class-index ranges.

    Validation errors raise ``DataLoaderError`` so training never continues with
    a partially valid dataset.
    """
    if n_classes <= 0:
        raise DataLoaderError("n_classes must be greater than zero.")

    image_mask_pairs = get_pairs_from_paths(images_path, segs_path)
    if not image_mask_pairs:
        raise DataLoaderError(
            f"No matched image-mask pairs found in {images_path} and {segs_path}."
        )

    errors = []
    for image_path, segmentation_path in tqdm(image_mask_pairs):
        image = _load_bgr_image(image_path)
        segmentation = _load_mask(segmentation_path)

        if image.shape[:2] != segmentation.shape[:2]:
            errors.append(
                f"Image and mask dimensions differ: {image_path} "
                f"({image.shape[:2]}) vs {segmentation_path} ({segmentation.shape[:2]})"
            )
        elif int(np.max(segmentation)) >= n_classes:
            errors.append(
                f"Mask class index out of range in {segmentation_path}: "
                f"expected 0-{n_classes - 1}, found {int(np.max(segmentation))}"
            )

        if errors and not show_all_errors:
            break

    if errors:
        raise DataLoaderError("Dataset validation failed:\n- " + "\n- ".join(errors))

    print("Dataset verified.")
    return True
