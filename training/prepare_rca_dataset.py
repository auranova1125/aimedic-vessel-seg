"""Prepare RCA angiography frames and class-index segmentation masks.

The annotation workflow stores a raw angiography frame, a red vessel label,
and a blue/green endpoint and entrance label for each filename stem. This
script validates those relationships before generating 256x256 model inputs
and four-class masks for ``keras_segmentation``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from preprocessing import normalize_to_uint8, prepare_input_image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
IMAGE_SIZE = (256, 256)
COLOR_THRESHOLD_HIGH = 200
COLOR_THRESHOLD_LOW = 255 - COLOR_THRESHOLD_HIGH


def annotation_files_by_stem(directory: Path, directory_name: str) -> dict[str, Path]:
    """Return supported image files indexed by filename stem.

    A duplicate stem is ambiguous because one frame must have one annotation
    file in each source directory.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {directory_name} directory: {directory}")

    files_by_stem: dict[str, Path] = {}
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if file_path.stem in files_by_stem:
            previous_path = files_by_stem[file_path.stem]
            raise ValueError(
                f"Duplicate filename stem '{file_path.stem}' in {directory_name}: "
                f"{previous_path.name}, {file_path.name}"
            )
        files_by_stem[file_path.stem] = file_path

    if not files_by_stem:
        raise ValueError(f"No supported image files found in {directory_name}: {directory}")
    return files_by_stem


def create_class_mask(vessel_label: np.ndarray, point_label: np.ndarray) -> np.ndarray:
    """Create one class-index mask from the red, blue, and green labels."""
    vessel_b, vessel_g, vessel_r = cv2.split(vessel_label)
    point_b, point_g, point_r = cv2.split(point_label)

    vessel_mask = (
        (vessel_b <= COLOR_THRESHOLD_LOW)
        & (vessel_g <= COLOR_THRESHOLD_LOW)
        & (vessel_r >= COLOR_THRESHOLD_HIGH)
    )
    endpoint_mask = (
        (point_b >= COLOR_THRESHOLD_HIGH)
        & (point_g <= COLOR_THRESHOLD_LOW)
        & (point_r <= COLOR_THRESHOLD_LOW)
    )
    entrance_mask = (
        (point_b <= COLOR_THRESHOLD_LOW)
        & (point_g >= COLOR_THRESHOLD_HIGH)
        & (point_r <= COLOR_THRESHOLD_LOW)
    )

    class_mask = vessel_mask.astype(np.uint8)
    class_mask[(vessel_mask) & (endpoint_mask)] = 2
    class_mask[(vessel_mask) & (entrance_mask)] = 3
    return class_mask


def generate_ground_truth_masks(
    vessel_labels: dict[str, Path], point_labels: dict[str, Path], output_dir: Path
) -> None:
    """Build class-index masks in ``gt/`` from color-coded annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_stem in sorted(vessel_labels):
        vessel_label = read_and_resize_image(vessel_labels[file_stem])
        point_label = read_and_resize_image(point_labels[file_stem])
        class_mask = create_class_mask(vessel_label, point_label)
        output_path = output_dir / f"{file_stem}.bmp"

        if not cv2.imwrite(str(output_path), class_mask):
            raise OSError(f"Could not write ground-truth mask: {output_path}")

    print(f"Ground-truth masks written to {output_dir}")


def prepare_training_images(image_files: dict[str, Path], output_dir: Path) -> None:
    """Resize and histogram-equalize raw frames for the segmentation model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_stem, image_path in sorted(image_files.items()):
        image = read_image(image_path)
        prepared_image = prepare_input_image(image, IMAGE_SIZE)
        output_path = output_dir / f"{file_stem}.bmp"

        if not cv2.imwrite(str(output_path), prepared_image):
            raise OSError(f"Could not write processed image: {output_path}")

    print(f"Processed images written to {output_dir}")


def read_and_resize_image(image_path: Path) -> np.ndarray:
    """Load, resize, and normalize an annotation image without filtering it."""
    image = read_image(image_path)
    resized_image = cv2.resize(
        image,
        dsize=IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    return normalize_to_uint8(resized_image)


def read_image(image_path: Path) -> np.ndarray:
    """Load an image or raise a path-specific error."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def validate_annotation_sets(
    train_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Verify that each raw frame has exactly one red and blue/green label."""
    image_files = annotation_files_by_stem(train_dir / "image", "image")
    vessel_labels = annotation_files_by_stem(train_dir / "label_r", "label_r")
    point_labels = annotation_files_by_stem(train_dir / "label_b_g", "label_b_g")

    expected_stems = set(image_files)
    for directory_name, files_by_stem in (("label_r", vessel_labels), ("label_b_g", point_labels)):
        missing_stems = sorted(expected_stems - set(files_by_stem))
        extra_stems = sorted(set(files_by_stem) - expected_stems)
        if missing_stems or extra_stems:
            details = []
            if missing_stems:
                details.append(f"missing labels for: {', '.join(missing_stems)}")
            if extra_stems:
                details.append(f"labels without matching images: {', '.join(extra_stems)}")
            raise ValueError(
                f"Filename mismatch between image and {directory_name}: "
                f"{'; '.join(details)}"
            )

    return image_files, vessel_labels, point_labels


def validate_source_dimensions(
    image_files: dict[str, Path],
    vessel_labels: dict[str, Path],
    point_labels: dict[str, Path],
) -> None:
    """Require each raw frame and its labels to share the same dimensions."""
    for file_stem, image_path in image_files.items():
        image_shape = read_image(image_path).shape[:2]
        vessel_shape = read_image(vessel_labels[file_stem]).shape[:2]
        point_shape = read_image(point_labels[file_stem]).shape[:2]

        if image_shape != vessel_shape or image_shape != point_shape:
            raise ValueError(
                f"Spatial dimension mismatch for '{file_stem}': image {image_shape}, "
                f"label_r {vessel_shape}, label_b_g {point_shape}"
            )


def parse_args() -> argparse.Namespace:
    """Parse the directory containing the RCA training inputs."""
    parser = argparse.ArgumentParser(
        description="Prepare RCA training images and segmentation masks."
    )
    parser.add_argument(
        "--train-dir",
        default="data/RCA_train",
        help="Directory containing image/, label_r/, and label_b_g/.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate annotations and create model-ready images and masks."""
    args = parse_args()
    train_dir = Path(args.train_dir)
    image_files, vessel_labels, point_labels = validate_annotation_sets(train_dir)
    validate_source_dimensions(image_files, vessel_labels, point_labels)

    prepare_training_images(image_files, train_dir / "image_processed")
    generate_ground_truth_masks(vessel_labels, point_labels, train_dir / "gt")


if __name__ == "__main__":
    main()
