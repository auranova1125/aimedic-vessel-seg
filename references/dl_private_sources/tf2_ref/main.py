"""Train the single-scale RGB DMPHN deblurring model on GOPRO image pairs."""

import os

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import data
import utils
from nets import DMPHN_1


def main():
    """Fit the DMPHN model and write its final checkpoint and training log."""
    tf.random.set_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

    with open(
        os.path.join(config.OUTPUT_DIRECTORY, "log.txt"),
        "w",
        encoding="utf-8",
    ) as log_file:
        model = DMPHN_1.DMPHN1()
        optimizer = tf.keras.optimizers.Adam(config.INITIAL_LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=utils.l2_loss,
            metrics=[utils.psnr, utils.ssim],
        )
        learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(
            utils.cosine_decay_learning_rate
        )
        training_logger = utils.TrainingLogger(log_file)

        model.fit(
            data.train_dataset,
            epochs=config.TRAINING_EPOCHS,
            callbacks=[learning_rate_schedule, training_logger],
            validation_data=data.test_dataset,
            validation_freq=config.VALIDATION_FREQUENCY,
        )
        model.save_weights(
            os.path.join(
                config.OUTPUT_DIRECTORY,
                "model_{}.ckpt".format(config.TRAINING_EPOCHS),
            )
        )


if __name__ == "__main__":
    main()
