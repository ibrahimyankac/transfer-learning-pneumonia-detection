"""
Model architecture and utilities
"""
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import *

def create_model():
    """
    Create transfer learning model with DenseNet121
    """
    # Load base model
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = False

    # Add custom classifier layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE_1)(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE_2)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def compile_model(model, learning_rate=INITIAL_LR):
    """
    Compile model with specified learning rate
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_callbacks(monitor_metric="val_accuracy"):
    """
    Get training callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric, 
            patience=EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True, 
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=REDUCE_LR_FACTOR, 
            patience=REDUCE_LR_PATIENCE, 
            min_lr=MIN_LR, 
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH, 
            monitor=monitor_metric, 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )
    ]
    return callbacks

def prepare_for_fine_tuning(base_model):
    """
    Prepare model for fine-tuning by unfreezing some layers
    """
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:BASE_MODEL_LAYERS_TO_FREEZE]:
        layer.trainable = False
    
    return base_model