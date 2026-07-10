# TensorFlow Segmentation and Deblurring Experiments

This directory contains two independent TensorFlow experiment paths:

1. A stacked-volume coronary artery and vein segmentation pipeline based on a
   2.5D hourglass network.
2. A GOPRO image-deblurring reference implementation based on DMPHN models in
   [`tf2_ref/`](tf2_ref/).

They use different data formats, models, and entry points. Run each path from
its own directory so local imports such as `config`, `data`, and `utils` resolve
to the intended modules.

## Dependencies

The scripts require Python with the following packages:

```bash
pip install tensorflow numpy opencv-python Pillow tqdm
```

TensorFlow, CUDA, and the installed Python version must be mutually compatible.
The hourglass training script uses `tf.distribute.MirroredStrategy`; it uses the
GPUs listed in `config.py` when available and otherwise TensorFlow can fall back
to CPU execution.

## Directory layout

```text
dl_private_sources/
├── config.py               # Stacked-volume segmentation settings
├── data_stack.py           # Converts per-slice files into .npy volumes
├── data.py                 # Slice-window dataset loader and augmentation
├── main.py                 # Hourglass training, validation, and test evaluation
├── utils.py                # Segmentation loss, metrics, schedules, callbacks
├── nets/
│   └── hourglass.py        # Stacked-volume hourglass segmentation network
└── tf2_ref/
    ├── config.py           # GOPRO DMPHN settings
    ├── data.py             # Blurred/sharp image-pair loader
    ├── main.py             # DMPHN training entry point
    ├── test.py             # DMPHN inference and comparison-image export
    ├── measure.py          # DMPHN_124 FLOP measurement utility
    ├── utils.py            # Deblurring loss, metrics, schedule, logger
    └── nets/
        ├── DMPHN.py        # Hierarchical single-channel DMPHN variants
        └── DMPHN_1.py      # Single-scale RGB DMPHN model
```

## Stacked-volume segmentation

### Purpose and flow

The root-level pipeline reads a volume of grayscale slices and predicts a mask
for every slice in a fixed 32-slice window. The processing flow is:

```text
Series/, GT/, vein/ images
        │
        ▼
data_stack.py
        │
        ▼
img.npy and gt.npy per case
        │
        ▼
data.py: consecutive 32-slice windows and training augmentation
        │
        ▼
nets/hourglass.py: HourglassSegmentationNet
        │
        ▼
main.py: warmup, training, validation, checkpoints, and test metrics
```

### Input data

`data_stack.py` expects the following case layout for each split:

```text
dataset/
├── train/
│   └── <case>/
│       ├── Series/        # Grayscale slice images
│       ├── GT/            # Coronary artery color labels
│       └── vein/          # Vein color labels
├── val/
│   └── <case>/
│       ├── Series/
│       ├── GT/
│       └── vein/
└── test/
    └── <case>/
        ├── Series/
        ├── GT/
        └── vein/
```

Filenames must contain an integer because `data_stack.py` sorts slices using the
first integer in each filename. For every case, the script writes:

- `img.npy`: `uint8` grayscale array with shape `[512, 512, slices]`.
- `gt.npy`: `uint8` label array with shape `[512, 512, slices]`.

`gt.npy` uses `0` for background, `1` for coronary artery, and `2` for vein.
When the two source masks overlap, the implementation writes background at that
pixel. The number and ordering of files under `GT/` and `vein/` must match.

### Preparing volumes

From this directory, run:

```bash
python data_stack.py
```

The script processes `train`, `val`, and `test` in sequence. It overwrites
`img.npy` and `gt.npy` in each case when those files already exist.

### Training and evaluation

Review the paths, image size, slice-window size, GPU list, and schedule in
`config.py`, then run:

```bash
python main.py
```

`data.py` loads every valid 32-slice window from the stored volumes. Training
windows receive random crop-and-resize augmentation; validation and test windows
are evaluated without augmentation. The module exposes the exact validation and
test window counts so `main.py` can terminate those stages after one pass.

Checkpoints and `log.txt` are written below the configured `OUTPUT_DIRECTORY`.
The test phase loads the last checkpoint saved at a validation interval.

### Segmentation notes

- A case must contain at least `WINDOW_SIZE` slices; otherwise it contributes no
  training, validation, or test windows.
- The hourglass output uses a sigmoid activation, while `data_stack.py` can
  produce label value `2` for veins. Confirm that this target encoding matches
  the intended loss and model output before starting a training run.
- `main.py` expects at least one validation and test window. Empty splits cause
  TensorFlow training or evaluation to fail.

## GOPRO DMPHN deblurring (`tf2_ref`)

### Purpose and flow

The DMPHN reference path learns to map a blurred RGB image to its sharp RGB
counterpart. Its training flow is:

```text
GOPRO blur_gamma/ and sharp/ PNG pairs
        │
        ▼
tf2_ref/data.py: decode, normalize to [-0.5, 0.5], augment training pairs
        │
        ▼
tf2_ref/nets/DMPHN_1.py: single-scale RGB DMPHN
        │
        ▼
tf2_ref/main.py: train, validate, save final checkpoint
        │
        ▼
tf2_ref/test.py: predict and save vertical blurred/predicted/sharp comparisons
```

### Input data

Set `TRAIN_DATA_PATTERN` and `TEST_DATA_PATTERN` in `tf2_ref/config.py`. Each
matched case directory must contain these sibling directories:

```text
<case>/
├── blur_gamma/            # Input PNG images
└── sharp/                 # Target PNG images
```

The loader sorts the two file lists independently and pairs them by position.
The directories therefore need the same filename ordering and the same number
of images. The code does not validate pair names or counts.

### Training

From `tf2_ref/`, run:

```bash
python main.py
```

Training crops each blurred/sharp pair jointly to `CROP_SIZE` and may flip it
horizontally. Both images are normalized to `[-0.5, 0.5]`. The script trains
`DMPHN1`, validates every `VALIDATION_FREQUENCY` epochs, and writes
`model_<TRAINING_EPOCHS>.ckpt` and `log.txt` under `OUTPUT_DIRECTORY`.

### Inference

After training, run:

```bash
python test.py
```

The script loads `model_<TRAINING_EPOCHS>.ckpt`, predicts every test image, and
writes a vertically concatenated PNG containing the blurred input, prediction,
and sharp target to `OUTPUT_DIRECTORY/results/`.

### Model measurement

`measure.py` profiles `DMPHN_124` with a fixed single-channel input shape of
`[1, 3400, 3400, 1]`:

```bash
python measure.py
```

This operation has a large memory requirement. It uses TensorFlow 1-style graph
profiling APIs through `tf.compat.v1`, so its behavior depends on the installed
TensorFlow version and runtime configuration.

## General usage notes

- Do not run the root scripts from `tf2_ref/`, or `tf2_ref` scripts from this
  directory. Both paths intentionally contain modules named `config.py`,
  `data.py`, and `utils.py`.
- Dataset files and outputs are excluded by `.gitignore`. Keep data paths in the
  corresponding `config.py` files rather than embedding local paths in loaders.
- The project has no bundled datasets or checkpoints. A full training or
  inference run requires the matching input data and, for inference, a trained
  checkpoint.
