# 🩺 Pneumonia Detection

Deep learning model for detecting pneumonia from chest X-ray images using DenseNet121 transfer learning.

**Accuracy: 91.7%** | **Pneumonia Detection: 96.9%**

## 📥 Dataset

Download the chest X-ray dataset from Kaggle:

**https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**

Extract to `chest_xray/` folder in project directory.

## 🚀 Quick Start

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **Download and extract dataset** to `chest_xray/` folder

3. **Run model**

```bash
python main.py
```

## 🔄 Options

**View Results:** `python main.py` - Shows model performance and confusion matrix

**Train New Model:** `python train_model.py` - Trains from scratch (takes time)

## 📊 Model Architecture

- **Transfer Learning:** DenseNet121 (pre-trained on ImageNet)
- **Custom Layers:** GlobalAveragePooling2D + Dense layers
- **Training:** Two-phase approach (classifier training + fine-tuning)

## 📈 Results

| Metric           | Value |
| ---------------- | ----- |
| Test Accuracy    | 91.7% |
| Normal Recall    | 82.9% |
| Pneumonia Recall | 96.9% |

> **🔬 Research Note:** These results can be further improved! With longer training (increased epochs), different optimization techniques (learning rate scheduling, ensemble methods) and advanced data augmentation strategies, 95%+ accuracy is achievable.

## 📁 Files

- `main.py` - Show results
- `train_model.py` - Train new model
- `config.py` - Configuration settings
- `best_model.h5` - Pre-trained model

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- See `requirements.txt` for full list

## ⚕️ Medical Disclaimer

This project is for research/educational purposes only. Not intended for actual medical diagnosis.
