"""
Training Script - Train new pneumonia detection model
"""
import os
from data_utils import create_data_generators, get_class_names
from model_utils import create_model
from train import train_model
from evaluate import evaluate_model

def main():
    """Train a new model from scratch"""
    
    print("🚀 Training New Model")
    
    # Backup existing model
    if os.path.exists("best_model.h5"):
        import shutil
        shutil.copy("best_model.h5", "best_model_backup.h5")
    
    # Load data and create model
    train_gen, val_gen, test_gen = create_data_generators()
    class_names = get_class_names(train_gen)
    model, base_model = create_model()
    
    # Training
    train_model(model, base_model, train_gen, val_gen)
    
    # Evaluation
    result = evaluate_model(test_gen, class_names)
    
    if result:
        print(f"\n✅ Training Complete - Test Accuracy: {result['accuracy']:.1%}")
    
    return result

if __name__ == "__main__":
    main()