"""Configuration for the stacked-volume hourglass experiment."""

import numpy as np

np.random.seed(10000)

EXPERIMENT_ID = 1
INITIAL_LEARNING_RATE = 1e-4
BATCH_SIZE = 2
TOTAL_ITERATIONS = 500

# Expected layout: dataset/<split>/<case>/{img.npy,gt.npy}
DATASET_DIRECTORY = "dataset"
OUTPUT_DIRECTORY = "./outputs/exp{}_lr_{}_batch_{}".format(
    EXPERIMENT_ID,
    INITIAL_LEARNING_RATE,
    BATCH_SIZE,
)

# Each training example contains this many consecutive slices.
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
WINDOW_SIZE = 32
WEIGHT_DECAY = 1e-5

WARMUP_EPOCHS = 5
WARMUP_STEPS_PER_EPOCH = int(TOTAL_ITERATIONS * 0.05 / WARMUP_EPOCHS)

TRAINING_EPOCHS = 500
TRAINING_STEPS_PER_EPOCH = int(TOTAL_ITERATIONS / TRAINING_EPOCHS)
TRAIN_LOG_FREQUENCY = 10
VALIDATION_FREQUENCY = 100

GPU_LIST = "0,1"
