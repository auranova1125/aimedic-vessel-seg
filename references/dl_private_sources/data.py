"""Dataset loading and augmentation for stacked-volume segmentation."""

import glob
import itertools

import cv2
import numpy as np
import tensorflow as tf

import config


def create_dataset(case_pattern, batch_size, is_training):
    """Create a dataset of consecutive slice windows from stored case volumes."""
    window_queue, case_volumes = load_case_volumes(case_pattern)
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_windows(
            window_queue,
            case_volumes,
            batch_size=batch_size,
            is_training=is_training,
        ),
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape(
                [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.WINDOW_SIZE]
            ),
            tf.TensorShape(
                [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.WINDOW_SIZE]
            ),
        ),
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE), len(window_queue)


def generate_windows(window_queue, case_volumes, batch_size, is_training):
    """Yield image and label batches with shape ``[B, H, W, WINDOW_SIZE]``."""
    if is_training:
        for _ in range(5):
            np.random.shuffle(window_queue)

    window_iterator = itertools.cycle(window_queue)
    while True:
        batch_images = []
        batch_labels = []

        for _ in range(batch_size):
            case_path, window_start = next(window_iterator)
            image_volume, label_volume = case_volumes[case_path]
            window_end = window_start + config.WINDOW_SIZE

            image_window = image_volume[:, :, window_start:window_end].astype(
                np.float32
            ) / 255.0
            label_window = label_volume[:, :, window_start:window_end].astype(
                np.float32
            )

            if is_training:
                image_window, label_window = randomly_augment_window(
                    image_window,
                    label_window,
                )

            batch_images.append(image_window)
            batch_labels.append(label_window)

        yield tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_labels)


def load_case_volumes(case_pattern):
    """Load stored volume pairs and enumerate every valid slice-window start.

    Each case directory must contain ``img.npy`` and ``gt.npy`` arrays with
    shape ``[height, width, slices]``.
    """
    case_paths = glob.glob(case_pattern)
    window_queue = []
    case_volumes = {}

    for case_path in case_paths:
        image_volume = np.load(case_path + "/img.npy")
        label_volume = np.load(case_path + "/gt.npy")

        for window_start in range(label_volume.shape[-1] - config.WINDOW_SIZE + 1):
            window_queue.append((case_path, window_start))
        case_volumes[case_path] = (image_volume, label_volume)

    return window_queue, case_volumes


def randomly_augment_window(image_window, label_window):
    """Apply the crop-and-resize augmentation used during training.

    Cubic interpolation can create fractional label values. Thresholding returns
    the mask to binary values before Gaussian smoothing expands positive pixels.
    """
    crop_height = np.random.randint(
        int(config.IMAGE_HEIGHT * 0.8),
        config.IMAGE_HEIGHT + 1,
    )
    crop_width = np.random.randint(
        int(config.IMAGE_WIDTH * 0.8),
        config.IMAGE_WIDTH + 1,
    )
    offset_height = np.random.randint(0, config.IMAGE_HEIGHT - crop_height + 1)
    offset_width = np.random.randint(0, config.IMAGE_WIDTH - crop_width + 1)

    image_window = image_window[
        offset_height : offset_height + crop_height,
        offset_width : offset_width + crop_width,
        :,
    ]
    label_window = label_window[
        offset_height : offset_height + crop_height,
        offset_width : offset_width + crop_width,
        :,
    ]

    target_size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    image_window = cv2.resize(
        image_window,
        dsize=target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    label_window = cv2.resize(
        label_window,
        dsize=target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    label_window = (label_window >= 0.5).astype(np.float32)
    label_window = (
        cv2.GaussianBlur(label_window, (3, 3), 1) > 0
    ).astype(np.float32)
    return image_window, label_window


print("\nLoading training data")
train_dataset, _ = create_dataset(
    config.DATASET_DIRECTORY + "/train/*",
    batch_size=config.BATCH_SIZE,
    is_training=True,
)

print("\nLoading validation data")
validation_dataset, validation_steps = create_dataset(
    config.DATASET_DIRECTORY + "/val/*",
    batch_size=1,
    is_training=False,
)

print("\nLoading test data")
test_dataset, test_steps = create_dataset(
    config.DATASET_DIRECTORY + "/test/*",
    batch_size=1,
    is_training=False,
)
