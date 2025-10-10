"""
Data preprocessing and augmentation utilities
"""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

def create_data_generators():
    """
    Create train, validation, and test data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=RESCALE_FACTOR,
        horizontal_flip=HORIZONTAL_FLIP,
        rotation_range=ROTATION_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
        validation_split=VALIDATION_SPLIT
    )

    # Only rescaling for test data
    test_datagen = ImageDataGenerator(rescale=RESCALE_FACTOR)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="training",
        shuffle=True,
    )

    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation",
        shuffle=False,
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def get_class_names(generator):
    """
    Get class names from data generator
    """
    return list(generator.class_indices.keys())