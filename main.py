"""
Pneumonia Detection - Model Results
"""
import os
from data_utils import create_data_generators, get_class_names
from evaluate import evaluate_model

def main():
    """Show model results"""
    
    if not os.path.exists("best_model.h5"):
        print("❌ Model not found! Download dataset first.")
        return
    
    train_gen, val_gen, test_gen = create_data_generators()
    class_names = get_class_names(train_gen)
    result = evaluate_model(test_gen, class_names)
    
    if result:
        print(f"\n📊 Test Accuracy: {result['accuracy']:.1%}")
        print(f"🫁 Normal Detection: {result['normal_recall']:.1%}")  
        print(f"🦠 Pneumonia Detection: {result['pneumonia_recall']:.1%}")
    
    return result

if __name__ == "__main__":
    main()