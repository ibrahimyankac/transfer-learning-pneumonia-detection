"""
Configuration file for pneumonia detection model
"""
import os

# Dataset paths
DATA_DIR = "chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE = "binary"
INPUT_SHAPE = (*IMG_SIZE, 3)

# Training parameters
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
VALIDATION_SPLIT = 0.1
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

# Model architecture
BASE_MODEL_LAYERS_TO_FREEZE = 150
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.3
DENSE_UNITS = 256

# Callbacks parameters
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-7

# Data augmentation parameters
RESCALE_FACTOR = 1/255.0
HORIZONTAL_FLIP = True
ROTATION_RANGE = 10
BRIGHTNESS_RANGE = (0.8, 1.2)

# Model saving
MODEL_PATH = "best_model.h5"

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']