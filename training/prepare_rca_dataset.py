"""Prepare RCA vessel segmentation inputs and class masks.

The annotation workflow stores three concepts separately:

- ``label_r``: red vessel center/area annotation
- ``label_b_g``: blue endpoint and green entrance annotation
- ``image``: angiography frame

This script resizes images to 256x256 and converts the color-coded labels into
the single-channel class-index masks expected by ``keras_segmentation``.
Class indices are:

0. background
1. vessel only
2. vessel endpoint
3. vessel entrance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (256, 256)
COLOR_THRESHOLD_HIGH = 200
COLOR_THRESHOLD_LOW = 255 - COLOR_THRESHOLD_HIGH


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an image to the 0-255 uint8 range used by the training pipeline."""
    max_value = np.max(image)
    if max_value == 0:
        return image.astype(np.uint8)
    return (image / max_value * 255).astype(np.uint8)


def read_color_label(label_path: Path) -> np.ndarray:
    """Load one color-coded annotation image at the model input resolution."""
    label_image = cv2.imread(str(label_path))
    if label_image is None:
        raise FileNotFoundError(f"Could not read label image: {label_path}")
    label_image = cv2.resize(label_image, dsize=IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return normalize_to_uint8(label_image)


def label_files_for_class(train_dir: Path, class_name: str) -> list[Path]:
    """Return the label files used for a class."""
    if class_name == "vessel":
        label_dir = train_dir / "label_r"
    elif class_name in {"end", "entrance"}:
        label_dir = train_dir / "label_b_g"
    else:
        raise ValueError("class_name must be one of: vessel, end, entrance")

    return sorted(p for p in label_dir.iterdir() if p.suffix.lower() in {".bmp", ".jpg", ".png"})


def extract_binary_label(train_dir: Path, class_name: str) -> np.ndarray:
    """Extract one binary mask stack from the color-coded labels."""
    label_files = label_files_for_class(train_dir, class_name)
    labels = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], len(label_files)), dtype=np.uint8)

    for index, label_path in enumerate(label_files):
        label_image = read_color_label(label_path)
        channel_b = label_image[:, :, 0]
        channel_g = label_image[:, :, 1]
        channel_r = label_image[:, :, 2]

        if class_name == "vessel":
            label = (channel_b <= COLOR_THRESHOLD_LOW) & (channel_g <= COLOR_THRESHOLD_LOW) & (channel_r >= COLOR_THRESHOLD_HIGH)
        elif class_name == "end":
            label = (channel_b >= COLOR_THRESHOLD_HIGH) & (channel_g <= COLOR_THRESHOLD_LOW) & (channel_r <= COLOR_THRESHOLD_LOW)
        else:
            label = (channel_b <= COLOR_THRESHOLD_LOW) & (channel_g >= COLOR_THRESHOLD_HIGH) & (channel_r <= COLOR_THRESHOLD_LOW)

        labels[:, :, index] = label.astype(np.uint8)

    return labels


def generate_ground_truth_masks(train_dir: Path) -> None:
    """Build class-index masks in ``gt/`` from the separated annotation folders."""
    vessel_mask = extract_binary_label(train_dir, class_name="vessel")
    endpoint_mask = extract_binary_label(train_dir, class_name="end")
    entrance_mask = extract_binary_label(train_dir, class_name="entrance")

    gt_dir = train_dir / "gt"
    gt_dir.mkdir(exist_ok=True)

    vessel_label_files = label_files_for_class(train_dir, "vessel")
    if len(vessel_label_files) != vessel_mask.shape[2]:
        raise ValueError(
            f"Label count mismatch: {len(vessel_label_files)} label files, {vessel_mask.shape[2]} generated masks"
        )

    # Background starts at 0. Vessel-only pixels are 1, endpoint pixels are 2,
    # and entrance pixels are 3. Endpoint/entrance are only assigned where a
    # vessel mask is also present.
    gt = (
        (vessel_mask == 1).astype(np.uint8)
        + ((endpoint_mask == 1) & (vessel_mask == 1)).astype(np.uint8)
        + (((entrance_mask == 1) & (vessel_mask == 1)).astype(np.uint8) * 2)
    )

    for index, label_path in enumerate(vessel_label_files):
        output_path = gt_dir / f"{label_path.stem}.bmp"
        cv2.imwrite(str(output_path), gt[:, :, index])

    print(f"GT data generated in {gt_dir}")
    print(f"Max class index: {int(np.max(gt))}")


def downscale_training_images(train_dir: Path) -> None:
    """Resize raw frames into ``image_processed/`` for 256x256 model input."""
    image_dir = train_dir / "image"
    output_dir = train_dir / "image_processed"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".bmp", ".jpg", ".png"})
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read training image: {image_path}")
        image = cv2.resize(image, dsize=IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        image = normalize_to_uint8(image)
        cv2.imwrite(str(output_dir / f"{image_path.stem}.bmp"), image)

    print(f"Processed images written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RCA training images and segmentation masks.")
    parser.add_argument(
        "--train-dir",
        default="data/RCA_train",
        help="Path to the RCA_train folder containing image/, label_r/, and label_b_g/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    downscale_training_images(train_dir)
    generate_ground_truth_masks(train_dir)


if __name__ == "__main__":
    main()
