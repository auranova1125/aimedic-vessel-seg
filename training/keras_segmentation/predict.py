"""Prediction, visualization, and evaluation helpers for segmentation models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import six
from tqdm import tqdm

from .data_utils.data_loader import (
    CLASS_COLORS,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_class_colors,
    get_image_array,
    get_pairs_from_paths,
    get_segmentation_array,
)
from .models.config import IMAGE_ORDERING
from .train import find_latest_checkpoint


def _load_prediction_image(image_input: np.ndarray | str) -> np.ndarray:
    """Load an inference image and normalize its channel layout to BGR."""
    if isinstance(image_input, six.string_types):
        image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read input image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("Input must be a NumPy image array or an image path.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, received shape {image.shape}.")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Expected one, three, or four image channels, received {image.shape[2]}.")


def _save_prediction_image(output_path: str, image: np.ndarray) -> None:
    """Create an output directory and save a visualization image."""
    parent_directory = Path(output_path).parent
    parent_directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise OSError(f"Could not write prediction image: {output_path}")


def get_colored_segmentation_image(
    segmentation: np.ndarray, n_classes: int, colors: list[tuple[int, int, int]] = CLASS_COLORS
) -> np.ndarray:
    """Convert a class-index mask into a BGR visualization image."""
    if segmentation.ndim != 2:
        raise ValueError(f"Expected a 2D class-index mask, received shape {segmentation.shape}.")
    if n_classes <= 0:
        raise ValueError("n_classes must be greater than zero.")

    minimum_class = int(np.min(segmentation))
    maximum_class = int(np.max(segmentation))
    if minimum_class < 0 or maximum_class >= n_classes:
        raise ValueError(
            f"Segmentation class indices must be in 0-{n_classes - 1}, "
            f"received {minimum_class}-{maximum_class}."
        )

    color_table = np.asarray(get_class_colors(n_classes, colors), dtype=np.uint8)
    colored_image = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(color_table):
        colored_image[segmentation == class_index] = color
    return colored_image


def get_legends(
    class_names: list[str], colors: list[tuple[int, int, int]] = CLASS_COLORS
) -> np.ndarray:
    """Create a BGR legend image for the supplied class names."""
    resolved_colors = get_class_colors(len(class_names), colors)
    legend = np.full(((len(class_names) * 25) + 25, 125, 3), 255, dtype=np.uint8)

    for index, (class_name, color) in enumerate(zip(class_names, resolved_colors)):
        cv2.putText(
            legend,
            class_name,
            (5, (index * 25) + 17),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(legend, (100, index * 25), (125, (index * 25) + 25), color, -1)
    return legend


def overlay_segmentation_image(
    input_image: np.ndarray, segmentation_image: np.ndarray
) -> np.ndarray:
    """Blend a segmentation visualization over its input image."""
    input_height, input_width = input_image.shape[:2]
    resized_segmentation = cv2.resize(segmentation_image, (input_width, input_height))
    return (input_image / 2 + resized_segmentation / 2).astype(np.uint8)


def concatenate_legend(segmentation_image: np.ndarray, legend_image: np.ndarray) -> np.ndarray:
    """Place a legend image to the left of a segmentation visualization."""
    output_height = max(segmentation_image.shape[0], legend_image.shape[0])
    output_width = segmentation_image.shape[1] + legend_image.shape[1]
    output_image = np.full((output_height, output_width, 3), legend_image[0, 0, 0], dtype=np.uint8)
    output_image[: legend_image.shape[0], : legend_image.shape[1]] = legend_image
    output_image[: segmentation_image.shape[0], legend_image.shape[1] :] = segmentation_image
    return output_image


def model_from_checkpoint_path(checkpoints_path: str):
    """Create a model from its checkpoint configuration and latest weights."""
    from .models.all_models import model_from_name

    config_path = f"{checkpoints_path}_config.json"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Checkpoint configuration not found: {config_path}")

    latest_weights = find_latest_checkpoint(checkpoints_path)
    if latest_weights is None:
        raise FileNotFoundError(f"No checkpoint weights found for prefix: {checkpoints_path}")

    with open(config_path, encoding="utf-8") as config_file:
        model_config = json.load(config_file)

    model = model_from_name[model_config["model_class"]](
        model_config["n_classes"],
        input_height=model_config["input_height"],
        input_width=model_config["input_width"],
    )
    print(f"Loaded weights {latest_weights}")
    model.load_weights(latest_weights)
    return model


def predict(
    model=None,
    inp: np.ndarray | str | None = None,
    out_fname: str | None = None,
    checkpoints_path: str | None = None,
    overlay_img: bool = False,
    class_names: list[str] | None = None,
    show_legends: bool = False,
    colors: list[tuple[int, int, int]] = CLASS_COLORS,
    prediction_width: int | None = None,
    prediction_height: int | None = None,
    image_preprocessor=None,
) -> np.ndarray:
    """Predict one class-index mask and optionally save its visualization."""
    if model is None:
        if checkpoints_path is None:
            raise ValueError("Provide a model instance or checkpoints_path.")
        model = model_from_checkpoint_path(checkpoints_path)
    if inp is None:
        raise ValueError("Provide an input image array or image path.")

    input_image = _load_prediction_image(inp)
    input_array = get_image_array(
        input_image,
        model.input_width,
        model.input_height,
        ordering=IMAGE_ORDERING,
        image_preprocessor=image_preprocessor,
    )
    prediction = model.predict(np.array([input_array]))[0]
    class_mask = prediction.reshape(
        (model.output_height, model.output_width, model.n_classes)
    ).argmax(axis=2)

    segmentation_image = visualize_segmentation(
        class_mask,
        input_image,
        n_classes=model.n_classes,
        colors=colors,
        overlay_img=overlay_img,
        show_legends=show_legends,
        class_names=class_names,
        prediction_width=prediction_width,
        prediction_height=prediction_height,
    )
    if out_fname is not None:
        _save_prediction_image(out_fname, segmentation_image)
    return class_mask


def predict_multiple(
    model=None,
    inps: list[np.ndarray | str] | None = None,
    inp_dir: str | None = None,
    out_dir: str | None = None,
    checkpoints_path: str | None = None,
    overlay_img: bool = False,
    class_names: list[str] | None = None,
    show_legends: bool = False,
    colors: list[tuple[int, int, int]] = CLASS_COLORS,
    prediction_width: int | None = None,
    prediction_height: int | None = None,
    image_preprocessor=None,
) -> list[np.ndarray]:
    """Predict every supported image in a directory or an explicit input list."""
    if model is None:
        if checkpoints_path is None:
            raise ValueError("Provide a model instance or checkpoints_path.")
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None:
        if inp_dir is None:
            raise ValueError("Provide inps or inp_dir.")
        input_directory = Path(inp_dir)
        if not input_directory.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
        inps = [
            str(path)
            for path in sorted(input_directory.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    if not isinstance(inps, list):
        raise TypeError("inps must be a list of image arrays or image paths.")
    if not inps:
        raise ValueError("No input images were found for prediction.")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    predictions = []
    for index, input_image in enumerate(tqdm(inps)):
        if out_dir is None:
            output_path = None
        elif isinstance(input_image, six.string_types):
            output_path = str(Path(out_dir) / Path(input_image).name)
        else:
            output_path = str(Path(out_dir) / f"{index}.png")

        predictions.append(
            predict(
                model=model,
                inp=input_image,
                out_fname=output_path,
                overlay_img=overlay_img,
                class_names=class_names,
                show_legends=show_legends,
                colors=colors,
                prediction_width=prediction_width,
                prediction_height=prediction_height,
                image_preprocessor=image_preprocessor,
            )
        )
    return predictions


def evaluate(
    model=None,
    inp_images: list[np.ndarray | str] | None = None,
    annotations: list[np.ndarray | str] | None = None,
    inp_images_dir: str | None = None,
    annotations_dir: str | None = None,
    checkpoints_path: str | None = None,
    image_preprocessor=None,
) -> dict[str, np.ndarray | float]:
    """Calculate intersection-over-union scores for a labeled dataset."""
    if model is None:
        if checkpoints_path is None:
            raise ValueError("Provide a model instance or checkpoints_path.")
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        if inp_images_dir is None or annotations_dir is None:
            raise ValueError("Provide image and annotation lists or both input directories.")
        matched_pairs = get_pairs_from_paths(inp_images_dir, annotations_dir)
        inp_images = [pair[0] for pair in matched_pairs]
        annotations = [pair[1] for pair in matched_pairs]

    if annotations is None or len(inp_images) != len(annotations):
        raise ValueError("Input images and annotations must have the same length.")

    true_positives = np.zeros(model.n_classes)
    false_positives = np.zeros(model.n_classes)
    false_negatives = np.zeros(model.n_classes)
    class_pixels = np.zeros(model.n_classes)

    for input_image, annotation in tqdm(zip(inp_images, annotations)):
        prediction = predict(model=model, inp=input_image, image_preprocessor=image_preprocessor)
        ground_truth = get_segmentation_array(
            annotation,
            model.n_classes,
            model.output_width,
            model.output_height,
            no_reshape=True,
        ).argmax(axis=-1)

        for class_index in range(model.n_classes):
            true_positives[class_index] += np.sum(
                (prediction == class_index) & (ground_truth == class_index)
            )
            false_positives[class_index] += np.sum(
                (prediction == class_index) & (ground_truth != class_index)
            )
            false_negatives[class_index] += np.sum(
                (prediction != class_index) & (ground_truth == class_index)
            )
            class_pixels[class_index] += np.sum(ground_truth == class_index)

    class_iou = true_positives / (true_positives + false_positives + false_negatives + 1e-12)
    normalized_class_pixels = class_pixels / np.sum(class_pixels)
    return {
        "frequency_weighted_IU": float(np.sum(class_iou * normalized_class_pixels)),
        "mean_IU": float(np.mean(class_iou)),
        "class_wise_IU": class_iou,
    }


def visualize_segmentation(
    segmentation: np.ndarray,
    input_image: np.ndarray | None = None,
    n_classes: int | None = None,
    colors: list[tuple[int, int, int]] = CLASS_COLORS,
    class_names: list[str] | None = None,
    overlay_img: bool = False,
    show_legends: bool = False,
    prediction_width: int | None = None,
    prediction_height: int | None = None,
) -> np.ndarray:
    """Render a class-index segmentation with optional overlay and legend."""
    if n_classes is None:
        n_classes = int(np.max(segmentation)) + 1

    segmentation_image = get_colored_segmentation_image(segmentation, n_classes, colors=colors)
    if input_image is not None:
        input_height, input_width = input_image.shape[:2]
        segmentation_image = cv2.resize(segmentation_image, (input_width, input_height))

    if prediction_height is not None and prediction_width is not None:
        segmentation_image = cv2.resize(segmentation_image, (prediction_width, prediction_height))
        if input_image is not None:
            input_image = cv2.resize(input_image, (prediction_width, prediction_height))

    if overlay_img:
        if input_image is None:
            raise ValueError("input_image is required when overlay_img is enabled.")
        segmentation_image = overlay_segmentation_image(input_image, segmentation_image)

    if show_legends:
        if class_names is None:
            raise ValueError("class_names is required when show_legends is enabled.")
        legend_image = get_legends(class_names, colors=colors)
        segmentation_image = concatenate_legend(segmentation_image, legend_image)
    return segmentation_image
