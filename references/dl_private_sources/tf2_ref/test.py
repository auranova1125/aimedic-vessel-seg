"""Run DMPHN inference and save blurred/predicted/sharp comparison images."""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image

import config
import data
import utils
from nets import DMPHN_1


def main():
    """Load the final training checkpoint and save one comparison image per pair."""
    results_directory = os.path.join(config.OUTPUT_DIRECTORY, "results")
    os.makedirs(results_directory, exist_ok=True)

    model = DMPHN_1.DMPHN1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.INITIAL_LEARNING_RATE),
        loss=utils.l2_loss,
    )
    model.load_weights(
        os.path.join(
            config.OUTPUT_DIRECTORY,
            "model_{}.ckpt".format(config.TRAINING_EPOCHS),
        )
    )

    for index, (blurred_image, sharp_image) in enumerate(data.test_dataset):
        predicted_image = model.predict_on_batch(blurred_image)
        comparison_image = tf.concat(
            [
                tf.squeeze(blurred_image) + 0.5,
                tf.clip_by_value(tf.squeeze(predicted_image) + 0.5, 0.0, 1.0),
                tf.squeeze(sharp_image) + 0.5,
            ],
            axis=0,
        )
        Image.fromarray(
            (comparison_image.numpy() * 255.0).astype(np.uint8)
        ).save(os.path.join(results_directory, "{}.png".format(index)))


if __name__ == "__main__":
    main()
