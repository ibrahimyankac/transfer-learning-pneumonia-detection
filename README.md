# 🩺 Pneumonia Detection

Deep learning model for detecting pneumonia from chest X-ray images using DenseNet121 transfer learning.

**Accuracy: 91.7%** | **Pneumonia Detection: 96.9%**

## 📥 Dataset

Download the chest X-ray dataset from Kaggle:

**https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**

Extract it into a `chest_xray/` folder in the project directory:

```
chest_xray/
├─ train/
│  ├─ NORMAL/
│  └─ PNEUMONIA/
└─ test/
   ├─ NORMAL/
   └─ PNEUMONIA/
```

The validation set is split from `train/` automatically (10%), so the dataset's small `val/` folder is not used.

## 🚀 Quick Start

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **Download and extract the dataset** into `chest_xray/`

3. **Evaluate the pre-trained model**

```bash
python main.py
```

## 🔄 Options

**View Results:** `python main.py` evaluates `best_model.h5` on the test set, prints the metrics and shows the confusion matrix.

**Train New Model:** `python train_model.py` trains from scratch (this takes a while). The current `best_model.h5` is backed up to `best_model_backup.h5` first.

## 📊 Model Architecture

- **Base model:** DenseNet121 pre-trained on ImageNet, 224×224 RGB input
- **Classifier head:** GlobalAveragePooling2D → Dropout (0.5) → Dense (256, ReLU) → Dropout (0.3) → Dense (1, sigmoid)
- **Training in two phases:**
  1. Classifier training with the base model frozen (Adam, lr `1e-4`)
  2. Fine-tuning with the first 150 base layers still frozen (Adam, lr `1e-5`)
- **Augmentation:** horizontal flip, ±10° rotation, brightness 0.8–1.2
- **Callbacks:** EarlyStopping, ReduceLROnPlateau and ModelCheckpoint (best `val_accuracy`)

All hyperparameters live in `config.py`.

## 📈 Results

| Metric           | Value |
| ---------------- | ----- |
| Test Accuracy    | 91.7% |
| Normal Recall    | 82.9% |
| Pneumonia Recall | 96.9% |

> **🔬 Research Note:** These results can be improved further. Longer training (more epochs), other optimization techniques (learning rate scheduling, ensemble methods) and stronger data augmentation could push accuracy to 95%+.

## 📁 Files

| File | Purpose |
| --- | --- |
| `main.py` | Evaluate the pre-trained model and show results |
| `train_model.py` | Train a new model end to end |
| `config.py` | Paths, hyperparameters and augmentation settings |
| `data_utils.py` | Train / validation / test data generators |
| `model_utils.py` | Model definition, compilation, callbacks and fine-tuning setup |
| `train.py` | Two-phase training loop |
| `evaluate.py` | Metrics, classification report and confusion matrix |
| `best_model.h5` | Pre-trained model weights |

## 🛠️ Requirements

- Python 3.11+
- TensorFlow 2.20 / Keras 3
- See `requirements.txt` for the full, pinned list

## ⚕️ Medical Disclaimer

This project is for research and educational purposes only. It is not intended for actual medical diagnosis.
