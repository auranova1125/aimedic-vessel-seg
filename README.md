# AIMedic Vessel Segmentation

This project segments right coronary artery (RCA) angiography frames into four
classes: background, vessel, vessel endpoint, and vessel entrance. The primary
pipeline prepares color-coded annotations, trains a ResNet50 U-Net model, and
writes colorized segmentation results for individual or directory-based
inference.

## Project Layout

```text
aimedic_vessel_segmentation/
├── data/
│   ├── README.md                 # Expected dataset layout and matching rules
│   ├── RCA_train/
│   │   ├── image/                # Raw angiography frames
│   │   ├── label_r/              # Red vessel annotations
│   │   ├── label_b_g/            # Blue endpoint and green entrance annotations
│   │   ├── image_processed/      # Generated 256x256 training frames
│   │   └── gt/                   # Generated class-index masks
│   └── RCA_test/
│       └── image/                # Inference input frames
├── training/
│   ├── preprocessing.py          # Shared resize and histogram equalization
│   ├── prepare_rca_dataset.py    # Dataset validation and mask generation
│   ├── train_rca_unet.py         # ResNet50 U-Net training entry point
│   ├── predict_rca_unet.py       # Checkpoint inference entry point
│   ├── checks/RCA/               # Default checkpoint prefix directory
│   └── keras_segmentation/       # Model, data loading, training, and prediction utilities
├── references/
│   └── dl_private_sources/       # Independent stacked-volume experiment code
└── requirements.txt
```

Image data and model parameters are excluded from Git. The `.gitkeep` files
preserve the required data directory structure. See [data/README.md](data/README.md)
for the complete input layout and filename rules.

## Class Definition

The preparation script reads BGR color annotations and produces one
single-channel mask per frame.

| Class index | Annotation | Visualization color |
| --- | --- | --- |
| `0` | Background | Black |
| `1` | Red vessel label | Red |
| `2` | Blue endpoint on a vessel | Blue |
| `3` | Green entrance on a vessel | Green |

Endpoint and entrance pixels are assigned only where the red vessel annotation
is present. The script rejects missing, extra, or duplicate filename stems
before creating training outputs.

## Environment

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses TensorFlow and standalone Keras. Select versions that are
compatible with the local Python version and available accelerator drivers.

## Data Preparation

Place each raw frame and its two annotation files under `data/RCA_train` with
the same filename stem. The accepted extensions are `.bmp`, `.jpeg`, `.jpg`,
and `.png`.

```bash
python training/prepare_rca_dataset.py --train-dir data/RCA_train
```

For every input frame, the command performs the following operations:

1. Verifies one-to-one matching between `image`, `label_r`, and `label_b_g`.
2. Resizes the frame to `256x256`, normalizes it to `uint8`, and applies
   histogram equalization to the luminance channel.
3. Resizes the annotation images with nearest-neighbor interpolation.
4. Writes the prepared frame to `image_processed/<stem>.bmp`.
5. Writes the class-index mask to `gt/<stem>.bmp`.

Histogram equalization is applied to angiography frames only. Labels remain
class-index data and are never filtered or interpolated with non-nearest-neighbor
operations.

## Model and Training

`training/train_rca_unet.py` builds the existing `resnet50_unet` model from
`training/keras_segmentation`. It uses a ResNet50 encoder, U-Net decoder,
four-class softmax output, and categorical cross-entropy loss.

```bash
python training/train_rca_unet.py \
  --train-dir data/RCA_train \
  --checkpoints-path training/checks/RCA/ \
  --epochs 1 \
  --batch-size 1
```

The script validates that `image_processed` and `gt` have identical filename
stems, matching spatial dimensions, and mask values in the range `0` to `3`.
When `--steps-per-epoch` is omitted, it is calculated from the number of
matched image-mask pairs and the batch size. Set `--gpu` only when a specific
`CUDA_VISIBLE_DEVICES` value is required; use `--gpu ''` for CPU-only execution.

Training writes a checkpoint configuration to
`training/checks/RCA/_config.json` and saves weights using numeric suffixes,
such as `training/checks/RCA/.0`.

## Inference

Put test frames in `data/RCA_test/image` and run:

```bash
python training/predict_rca_unet.py \
  --input-dir data/RCA_test/image \
  --output-dir testing/outputs/RCA \
  --checkpoints-path training/checks/RCA/
```

Inference reads every supported image extension, creates the output directory
when necessary, resizes each frame to the model input shape, and applies the
same luminance histogram equalization used during data preparation. Each output
is a colorized BGR segmentation image written with the input filename.

## Main Code Flow

```text
RCA_train/image + label_r + label_b_g
                |
                v
training/prepare_rca_dataset.py
                |
                v
RCA_train/image_processed + RCA_train/gt
                |
                v
training/train_rca_unet.py
                |
                v
training/checks/RCA/_config.json + numbered weight files
                |
                v
training/predict_rca_unet.py
                |
                v
testing/outputs/RCA/<input filename>
```

## Additional Experiment Code

`references/dl_private_sources` contains a separate TensorFlow experiment for
stacked-volume segmentation and deblurring. Its README documents that code and
its execution paths independently. It is not required for the RCA 2D training
and inference commands above.

## Data Notes

The repository contains no medical images, annotations, or trained weight
files. A full training run therefore requires an authorized RCA dataset with
the directory structure described in [data/README.md](data/README.md).
