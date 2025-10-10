"""
Model training utilities
"""
from model_utils import compile_model, get_callbacks, prepare_for_fine_tuning
from config import *

def train_classifier_phase(model, train_gen, val_gen):
    """
    Phase 1: Train only classifier layers
    """
    print("Phase 1: Classifier Training")
    
    # Compile model with initial learning rate
    model = compile_model(model, INITIAL_LR)
    callbacks = get_callbacks()
    
    # Train classifier
    history_phase1 = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_phase1

def train_fine_tuning_phase(model, base_model, train_gen, val_gen):
    """
    Phase 2: Fine-tuning with unfrozen layers
    """
    print("Phase 2: Fine-tuning")
    
    # Prepare for fine-tuning
    base_model = prepare_for_fine_tuning(base_model)
    
    # Recompile with lower learning rate
    model = compile_model(model, FINE_TUNE_LR)
    callbacks = get_callbacks()
    
    # Fine-tuning training
    history_phase2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_phase2

def train_model(model, base_model, train_gen, val_gen):
    """
    Complete training pipeline: classifier + fine-tuning
    """
    
    # Phase 1: Classifier training
    history_phase1 = train_classifier_phase(model, train_gen, val_gen)
    
    # Phase 2: Fine-tuning
    history_phase2 = train_fine_tuning_phase(model, base_model, train_gen, val_gen)
    
    return history_phase1, history_phase2