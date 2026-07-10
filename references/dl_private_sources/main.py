"""Train and evaluate the stacked-volume hourglass segmentation model."""

import os

import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_LIST
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import data
import utils
from nets import hourglass


def enable_gpu_memory_growth():
    """Allow TensorFlow to allocate GPU memory on demand when GPUs are available."""
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)


def main():
    """Run warmup training, scheduled training, checkpointing, and evaluation."""
    enable_gpu_memory_growth()
    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

    with open(
        os.path.join(config.OUTPUT_DIRECTORY, "log.txt"),
        "w",
        encoding="utf-8",
    ) as log_file:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = hourglass.HourglassSegmentationNet()
            optimizer = tf.keras.optimizers.Adam(config.INITIAL_LEARNING_RATE)
            model.compile(
                optimizer=optimizer,
                loss=utils.dice_loss,
                metrics=[
                    utils.true_positive_rate,
                    utils.false_positive_rate,
                    utils.sorensen_dice,
                    utils.jaccard_index,
                ],
            )

            warmup_scheduler = tf.keras.callbacks.LearningRateScheduler(
                utils.warmup_learning_rate
            )
            training_scheduler = tf.keras.callbacks.LearningRateScheduler(
                utils.cosine_decay_learning_rate
            )
            checkpoint_saver = utils.CheckpointSaver(model)
            training_logger = utils.TrainingLogger(log_file)

            print("Starting warmup")
            model.fit(
                data.train_dataset,
                epochs=config.WARMUP_EPOCHS,
                steps_per_epoch=config.WARMUP_STEPS_PER_EPOCH,
                callbacks=[warmup_scheduler],
            )

            print("Starting training and validation")
            model.fit(
                data.train_dataset,
                epochs=config.TRAINING_EPOCHS,
                steps_per_epoch=config.TRAINING_STEPS_PER_EPOCH,
                callbacks=[training_scheduler, training_logger, checkpoint_saver],
                validation_data=data.validation_dataset,
                validation_freq=config.VALIDATION_FREQUENCY,
                validation_steps=data.validation_steps,
            )

            if checkpoint_saver.last_checkpoint_iteration is None:
                raise RuntimeError(
                    "No checkpoint was saved. Set VALIDATION_FREQUENCY to a value "
                    "that is reached during training."
                )

            checkpoint_path = os.path.join(
                config.OUTPUT_DIRECTORY,
                "model_{:.0f}.ckpt".format(
                    checkpoint_saver.last_checkpoint_iteration
                ),
            )
            print("Starting test evaluation")
            model.load_weights(checkpoint_path)
            results = model.evaluate(
                data.test_dataset,
                steps=data.test_steps,
                return_dict=True,
            )

            log_file.write("\nTest\n")
            log_file.write(
                "Iteration: {:.0f}, loss: {:.6f}\n".format(
                    checkpoint_saver.last_checkpoint_iteration,
                    results["loss"],
                )
            )
            log_file.write(
                "True-positive rate: {:.6f}, false-positive rate: {:.6f}\n".format(
                    results["true_positive_rate"],
                    results["false_positive_rate"],
                )
            )
            log_file.write(
                "Sorensen-Dice: {:.6f}, Jaccard: {:.6f}\n".format(
                    results["sorensen_dice"],
                    results["jaccard_index"],
                )
            )


if __name__ == "__main__":
    main()
