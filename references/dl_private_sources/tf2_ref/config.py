"""Configuration for the GOPRO DMPHN deblurring experiment."""

TRAIN_DATA_PATTERN = "../../4. dataset/GOPRO_Large/train/*"
TEST_DATA_PATTERN = "../../4. dataset/GOPRO_Large/test/*"

MODEL_NAME = "2. DMPHN_1"
OUTPUT_DIRECTORY = "./outputs/{}/".format(MODEL_NAME)

RANDOM_SEED = 10000
INITIAL_LEARNING_RATE = 1e-4
BATCH_SIZE = 6
CROP_SIZE = 256
TRAINING_EPOCHS = 3000
VALIDATION_FREQUENCY = 30
