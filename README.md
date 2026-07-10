# AIMedic Vessel Segmentation

RCA angiography frame에서 혈관 영역과 입구/출구 지점을 분할하기 위한 프로젝트입니다.


## 구성 개요

```text
aimedic_vessel_segmentation/
├── data/
│   ├── README.md             # data layout and file correspondence
│   ├── RCA_train/
│   │   ├── image/             # angiography frame
│   │   ├── image_processed/   # 256x256 학습 입력
│   │   ├── label_r/           # red vessel annotation
│   │   ├── label_b_g/         # blue endpoint / green entrance annotation
│   │   └── gt/                # class-index mask
│   └── RCA_test/
│       └── image/             # inference input frames
├── training/
│   ├── prepare_rca_dataset.py # 라벨 변환 및 이미지 resize
│   ├── train_rca_unet.py      # ResNet50-UNet 학습
│   ├── predict_rca_unet.py    # 체크포인트 추론
│   ├── checks/RCA/            # Keras checkpoint prefix
│   └── keras_segmentation/    # 학습/추론에 사용하는 segmentation 코드
├── references/
│   └── dl_private_sources/    # stacked-volume 기반 참고 실험
└── requirements.txt
```

## 코드 흐름

```text
RCA_train/image
RCA_train/label_r
RCA_train/label_b_g
        |
        v
training/prepare_rca_dataset.py
        |
        v
RCA_train/image_processed
RCA_train/gt
        |
        v
training/train_rca_unet.py
        |
        v
training/checks/RCA/
        |
        v
training/predict_rca_unet.py
```

`data/README.md`에는 원본 이미지, 색상 라벨, 전처리 결과, class-index mask의
역할과 파일명 대응 규칙을 별도로 설명해 두었습니다. 이미지와 라벨 파일은
저장소에 포함하지 않지만, `.gitkeep` 파일로 필요한 폴더 구조는 유지합니다.

## 환경 설치

```bash
cd aimedic_vessel_segmentation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow/Keras는 Python 버전과 OS에 따라 설치 조합이 민감합니다. 먼저 현재 환경에서 설치 가능한 TensorFlow 버전을 맞춘 뒤 실행하는 편이 좋습니다.

## 데이터 준비

```bash
python training/prepare_rca_dataset.py --train-dir data/RCA_train
```

라벨 색상은 다음 의미로 사용됩니다.

- red: vessel
- blue: vessel endpoint
- green: vessel entrance

생성되는 `gt/` mask class는 다음과 같습니다.

- `0`: background
- `1`: vessel only
- `2`: endpoint on vessel
- `3`: entrance on vessel

학습에서는 `image_processed/`와 `gt/`의 파일 stem이 서로 같아야 합니다. 예를 들어 `image_processed/K02001RCA00008.bmp`에는 `gt/K02001RCA00008.bmp`가 대응됩니다.

## 학습

```bash
python training/train_rca_unet.py \
  --train-dir data/RCA_train \
  --checkpoints-path training/checks/RCA/ \
  --epochs 1 \
  --batch-size 1 \
  --steps-per-epoch 892
```

빠르게 실행 여부만 확인하려면 `--steps-per-epoch 1`처럼 줄여서 시도할 수 있습니다.

기본 설정:

- model: `resnet50_unet`
- input size: `256x256`
- classes: `4`
- checkpoint prefix: `training/checks/RCA/`

## 추론

```bash
python training/predict_rca_unet.py \
  --input-dir data/RCA_test/image \
  --output-dir testing/outputs/RCA \
  --checkpoints-path training/checks/RCA/
```

`keras_segmentation` 코드는 checkpoint path를 디렉터리라기보다 파일 prefix처럼 사용합니다. 이 프로젝트에서는 `training/checks/RCA/` 형태를 기본값으로 둡니다.

## 주요 파일

- `training/prepare_rca_dataset.py`: 색상 annotation을 class-index mask로 변환하고 이미지를 256x256으로 맞춥니다.
- `training/train_rca_unet.py`: RCA 학습 데이터를 읽어 `resnet50_unet` 모델을 학습합니다.
- `training/predict_rca_unet.py`: checkpoint를 불러와 테스트 이미지에 대한 mask를 저장합니다.
- `training/keras_segmentation/`: 모델, data loader, train/predict 유틸리티가 들어 있습니다.
- `references/dl_private_sources/`: 2D RCA pipeline과 별개로, stacked slice 입력을 다루는 참고 실험 코드입니다.

## 작업 맥락

이 프로젝트의 관심사는 단순 vessel/background 분할보다 조금 더 구체적입니다.

- RCA 혈관 영역을 안정적으로 분리하기
- 혈관 입구와 출구를 따로 라벨링해 후속 분석에 활용하기
- false positive를 줄이고 배경 구조와 구분하기
- annotation 폴더와 학습 입력을 일관된 형태로 관리하기

## 한계

- 현재 포함된 데이터는 샘플 수준입니다. (의료 데이터의 권한 문제)
- 전체 학습 재현에는 전체 RCA 데이터셋이 필요합니다.
- TensorFlow/Keras 버전 차이에 따라 일부 API 조정이 필요할 수 있습니다.
- `references/dl_private_sources/`는 같은 주제의 참고 실험이지만, 메인 RCA 2D 학습 흐름과 직접 연결되지는 않습니다.
