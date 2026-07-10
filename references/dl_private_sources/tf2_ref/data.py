"""GOPRO image-pair loading and augmentation for DMPHN training."""

import glob

import tensorflow as tf

import config


def collect_image_paths(is_training):
    """Collect matching blurred and sharp image paths from the configured split."""
    data_pattern = (
        config.TRAIN_DATA_PATTERN if is_training else config.TEST_DATA_PATTERN
    )
    blurred_image_paths = []
    sharp_image_paths = []

    for case_path in glob.glob(data_pattern):
        blurred_image_paths.extend(sorted(glob.glob(case_path + "/blur_gamma/*")))
        sharp_image_paths.extend(sorted(glob.glob(case_path + "/sharp/*")))

    return blurred_image_paths, sharp_image_paths


def generate_image_pairs(is_training, blurred_image_paths, sharp_image_paths):
    """Yield normalized blurred/sharp image pairs.

    Training pairs are concatenated before random cropping and flipping so both
    images receive the identical spatial transformation.
    """
    for blurred_path, sharp_path in zip(blurred_image_paths, sharp_image_paths):
        blurred_image = decode_and_normalize_image(blurred_path)
        sharp_image = decode_and_normalize_image(sharp_path)

        if is_training:
            paired_image = tf.concat([blurred_image, sharp_image], axis=-1)
            paired_image = tf.image.random_crop(
                paired_image,
                [config.CROP_SIZE, config.CROP_SIZE, 6],
                seed=config.RANDOM_SEED,
            )
            paired_image = tf.image.random_flip_left_right(
                paired_image,
                seed=config.RANDOM_SEED,
            )
            blurred_image, sharp_image = (
                paired_image[:, :, :3],
                paired_image[:, :, 3:],
            )

        yield blurred_image, sharp_image


def decode_and_normalize_image(image_path):
    """Decode a PNG image and map pixel values from ``[0, 255]`` to ``[-0.5, 0.5]``."""
    image = tf.image.decode_png(
        tf.io.read_file(image_path),
        channels=3,
        dtype=tf.uint8,
    )
    return tf.cast(image, tf.float32) / 255.0 - 0.5


train_blurred_paths, train_sharp_paths = collect_image_paths(is_training=True)
train_dataset = tf.data.Dataset.from_generator(
    lambda: generate_image_pairs(True, train_blurred_paths, train_sharp_paths),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        tf.TensorShape([config.CROP_SIZE, config.CROP_SIZE, 3]),
        tf.TensorShape([config.CROP_SIZE, config.CROP_SIZE, 3]),
    ),
).batch(config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

test_blurred_paths, test_sharp_paths = collect_image_paths(is_training=False)
test_dataset = tf.data.Dataset.from_generator(
    lambda: generate_image_pairs(False, test_blurred_paths, test_sharp_paths),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
        tf.TensorShape([None, None, 3]),
        tf.TensorShape([None, None, 3]),
    ),
).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
