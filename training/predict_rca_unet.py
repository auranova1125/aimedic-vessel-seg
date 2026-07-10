"""Run inference with an RCA vessel segmentation checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict RCA vessel segmentation masks.")
    parser.add_argument("--input-dir", default="data/RCA_test/image", help="Folder of test images.")
    parser.add_argument("--output-dir", default="testing/outputs/RCA", help="Folder for predicted masks.")
    parser.add_argument("--checkpoints-path", default="training/checks/RCA/", help="Checkpoint prefix path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    from keras_segmentation.predict import predict_multiple

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_multiple(
        inp_dir=str(Path(args.input_dir)),
        out_dir=str(output_dir),
        checkpoints_path=args.checkpoints_path,
    )


if __name__ == "__main__":
    main()
