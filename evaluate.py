"""
Model evaluation utilities
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model(test_gen, class_names):
    """
    Evaluate the trained model and show results
    """
    if not os.path.exists("best_model.h5"):
        print("❌ Model not found!")
        return None
    
    # Load model and predict
    model = tf.keras.models.load_model("best_model.h5")
    pred_probs = model.predict(test_gen, verbose=0)
    pred_labels = (pred_probs > 0.5).astype(int).ravel()
    true_labels = test_gen.classes
    
    # Calculate metrics
    test_accuracy = (pred_labels == true_labels).mean()
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    
    # Show confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Test Accuracy: {test_accuracy:.1%}")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    # Return results
    return {
        'accuracy': test_accuracy,
        'normal_recall': report['NORMAL']['recall'],
        'pneumonia_recall': report['PNEUMONIA']['recall'],
        'normal_precision': report['NORMAL']['precision'],
        'pneumonia_precision': report['PNEUMONIA']['precision']
    }

