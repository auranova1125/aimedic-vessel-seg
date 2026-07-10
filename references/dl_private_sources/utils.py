"""Loss functions, metrics, schedules, and callbacks for segmentation training."""

import os

import numpy as np
import tensorflow as tf

import config


def cosine_decay_learning_rate(epoch):
    """Return the cosine-decayed learning rate for a training epoch."""
    return config.INITIAL_LEARNING_RATE * 0.5 * (
        1 + np.cos(np.pi * epoch / config.TRAINING_EPOCHS)
    )


def dice_loss(target_mask, prediction):
    """Compute the Dice loss over the height and width dimensions."""
    numerator = 2 * tf.reduce_sum(target_mask * prediction, axis=[1, 2])
    denominator = tf.reduce_sum(prediction, axis=[1, 2]) + tf.reduce_sum(
        target_mask,
        axis=[1, 2],
    )
    return tf.reduce_mean(1 - numerator / denominator)


def false_positive_rate(target_mask, prediction):
    """Return the false-positive rate after rounding the prediction mask."""
    binary_prediction = tf.round(prediction)
    true_negative = tf.reduce_sum(
        (1 - binary_prediction) * (1 - target_mask),
        axis=[1, 2, 3],
    )
    false_positive = tf.reduce_sum(
        binary_prediction * (1 - target_mask),
        axis=[1, 2, 3],
    )
    return 1 - true_negative / (true_negative + false_positive)


def jaccard_index(target_mask, prediction):
    """Return the Jaccard index after rounding the prediction mask."""
    binary_prediction = tf.round(prediction)
    true_positive = tf.reduce_sum(binary_prediction * target_mask, axis=[1, 2, 3])
    false_negative = tf.reduce_sum(
        (1 - binary_prediction) * target_mask,
        axis=[1, 2, 3],
    )
    false_positive = tf.reduce_sum(
        binary_prediction * (1 - target_mask),
        axis=[1, 2, 3],
    )
    return true_positive / (true_positive + false_positive + false_negative)


def sorensen_dice(target_mask, prediction):
    """Return the Sorensen-Dice score after rounding the prediction mask."""
    binary_prediction = tf.round(prediction)
    true_positive = tf.reduce_sum(binary_prediction * target_mask, axis=[1, 2, 3])
    false_negative = tf.reduce_sum(
        (1 - binary_prediction) * target_mask,
        axis=[1, 2, 3],
    )
    false_positive = tf.reduce_sum(
        binary_prediction * (1 - target_mask),
        axis=[1, 2, 3],
    )
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def true_positive_rate(target_mask, prediction):
    """Return the true-positive rate after rounding the prediction mask."""
    binary_prediction = tf.round(prediction)
    true_positive = tf.reduce_sum(binary_prediction * target_mask, axis=[1, 2, 3])
    false_negative = tf.reduce_sum(
        (1 - binary_prediction) * target_mask,
        axis=[1, 2, 3],
    )
    return true_positive / (true_positive + false_negative)


def warmup_learning_rate(epoch):
    """Increase the learning rate linearly during the warmup phase."""
    return config.INITIAL_LEARNING_RATE / config.WARMUP_EPOCHS * (epoch + 1)


class CheckpointSaver(tf.keras.callbacks.Callback):
    """Save a checkpoint at every configured validation interval."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_checkpoint_iteration = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % config.VALIDATION_FREQUENCY == 0:
            iteration = (epoch + 1) * config.TOTAL_ITERATIONS / config.TRAINING_EPOCHS
            checkpoint_path = os.path.join(
                config.OUTPUT_DIRECTORY,
                "model_{:.0f}.ckpt".format(iteration),
            )
            self.model.save_weights(checkpoint_path)
            self.last_checkpoint_iteration = iteration


class TrainingLogger(tf.keras.callbacks.Callback):
    """Write training loss to the experiment log at a fixed interval."""

    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % config.TRAIN_LOG_FREQUENCY == 0:
            iteration = (epoch + 1) * config.TOTAL_ITERATIONS / config.TRAINING_EPOCHS
            self.log_file.write(
                "Train iteration: {:.0f}, loss: {:.6f}\n".format(
                    iteration,
                    logs["loss"],
                )
            )
            self.log_file.flush()
