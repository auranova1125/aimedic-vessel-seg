"""Train the RCA vessel segmentation model.

The script keeps the project intentionally small: explicit paths, one model
definition, and no separate experiment framework.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50-UNet on RCA vessel segmentation data.")
    parser.add_argument("--train-dir", default="data/RCA_train", help="Folder containing image_processed/ and gt/.")
    parser.add_argument("--checkpoints-path", default="training/checks/RCA/", help="Checkpoint prefix path.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=892,
        help="Training steps per epoch. Reduce this for a quick smoke run.",
    )
    parser.add_argument("--gpu", default="0,1", help="CUDA_VISIBLE_DEVICES value. Use '' for CPU-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Import after CUDA environment setup, matching TensorFlow/Keras practice.
    from keras_segmentation.models.unet import resnet50_unet

    train_dir = Path(args.train_dir)
    checkpoints_path = args.checkpoints_path

    model = resnet50_unet(n_classes=4, input_height=256, input_width=256)

    model.train(
        train_images=str(train_dir / "image_processed"),
        train_annotations=str(train_dir / "gt"),
        checkpoints_path=checkpoints_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
    )


if __name__ == "__main__":
    main()
