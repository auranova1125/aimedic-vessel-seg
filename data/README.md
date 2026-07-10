# RCA Dataset Layout

This directory defines the input and generated files used by the RCA vessel
segmentation pipeline. The repository keeps the directory layout but excludes
medical images, annotations, generated masks, and model artifacts.

```text
data/
├── RCA_train/
│   ├── image/             # Raw angiography frames
│   ├── label_r/           # Red vessel annotations
│   ├── label_b_g/         # Blue endpoint and green entrance annotations
│   ├── image_processed/   # Generated 256x256 training frames
│   └── gt/                # Generated class-index masks
└── RCA_test/
    └── image/             # Test frames used for inference
```

## Training Input Rules

Each training frame must have one file with the same filename stem in all
three source directories. For example:

```text
RCA_train/image/frame_001.jpg
RCA_train/label_r/frame_001.bmp
RCA_train/label_b_g/frame_001.bmp
```

`training/prepare_rca_dataset.py` accepts `.bmp`, `.jpeg`, `.jpg`, and `.png`
files. It rejects duplicate stems, source images without labels, and labels
without a matching source image. Each frame and its two annotation images must
also have identical spatial dimensions.

## Generated Training Files

For every valid source frame, the preparation command writes these files:

```text
RCA_train/image_processed/frame_001.bmp
RCA_train/gt/frame_001.bmp
```

`image_processed` contains the raw frame resized to `256x256`, normalized to
the `uint8` range, and histogram-equalized on the luminance channel. `gt`
contains a single-channel class-index mask:

| Value | Meaning |
| --- | --- |
| `0` | Background |
| `1` | Vessel |
| `2` | Vessel endpoint |
| `3` | Vessel entrance |

The blue endpoint and green entrance labels are retained only where the red
vessel annotation is present.

## Test Input

Place inference frames in `RCA_test/image`. The inference script accepts the
same image extensions, creates the requested output directory, and writes one
colorized segmentation image for every input frame.

## Commands

Run the following commands from the project root:

```bash
python training/prepare_rca_dataset.py --train-dir data/RCA_train

python training/train_rca_unet.py \
  --train-dir data/RCA_train \
  --checkpoints-path training/checks/RCA/

python training/predict_rca_unet.py \
  --input-dir data/RCA_test/image \
  --output-dir testing/outputs/RCA \
  --checkpoints-path training/checks/RCA/
```
