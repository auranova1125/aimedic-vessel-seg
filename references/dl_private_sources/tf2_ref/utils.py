"""Loss, metrics, schedule, and logging callback for DMPHN training."""

import numpy as np
import tensorflow as tf

import config


def cosine_decay_learning_rate(epoch):
    """Return a cosine-decayed learning rate for a training epoch."""
    return config.INITIAL_LEARNING_RATE * 0.5 * (
        1 + np.cos(np.pi * epoch / config.TRAINING_EPOCHS)
    )


def l2_loss(target_image, prediction):
    """Compute the mean squared error between sharp targets and predictions."""
    return tf.reduce_mean(tf.square(target_image - prediction))


def psnr(target_image, prediction):
    """Compute PSNR after converting tensors back to the ``[0, 1]`` range."""
    return tf.image.psnr(
        target_image + 0.5,
        tf.clip_by_value(prediction + 0.5, 0.0, 1.0),
        max_val=1.0,
    )


def ssim(target_image, prediction):
    """Compute SSIM after converting tensors back to the ``[0, 1]`` range."""
    return tf.image.ssim(
        target_image + 0.5,
        tf.clip_by_value(prediction + 0.5, 0.0, 1.0),
        max_val=1.0,
    )


class TrainingLogger(tf.keras.callbacks.Callback):
    """Write train and validation metrics at each validation interval."""

    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % config.VALIDATION_FREQUENCY == 0:
            self.log_file.write(
                "Epoch: {}, train loss: {:.6f}, PSNR: {:.6f}, SSIM: {:.6f}\n".format(
                    epoch + 1,
                    logs["loss"],
                    logs["psnr"],
                    logs["ssim"],
                )
            )
            self.log_file.write(
                "Epoch: {}, validation PSNR: {:.6f}, SSIM: {:.6f}\n\n".format(
                    epoch + 1,
                    logs["val_psnr"],
                    logs["val_ssim"],
                )
            )
            self.log_file.flush()
