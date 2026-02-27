# 🧠 Medical Image Tumor Segmentation using U-Net

## 📌 Overview
This project implements an end-to-end deep learning pipeline for tumor detection and segmentation in medical images using a U-Net architecture. The model takes MRI images as input and generates pixel-wise segmentation masks highlighting tumor regions.

The project emphasizes research-grade practices such as proper data splitting, reproducibility, and clean evaluation.

---

## 🎯 Purpose
To build an automated system that detects tumor regions in medical images using deep learning, reducing reliance on manual inspection.

---

## 📊 Results
The trained model analyzes an input medical image and produces a segmentation mask indicating the presence and location of tumor regions. Performance is evaluated using Dice and IoU metrics to ensure accurate and reliable segmentation.

---

## 🧠 Key Features
- U-Net based medical image segmentation
- Custom Dice Loss and IoU metrics
- Proper Train / Validation / Test split (no data leakage)
- Data augmentation for better generalization
- Reproducible training with fixed random seeds
- Modular and clean project structure
- Google Colab compatible notebook

---

## 🏗️ Project Structure
```
.
├── App.py                 # Main training and evaluation script
├── unet.py               # U-Net architecture
├── utils.py              # Metrics, generators, helper functions
├── Colab_App.ipynb       # Colab runnable version
├── requirements.txt      # Dependencies
├── results/              # Sample outputs
└── datasets/             # Dataset placeholder (not included)
```

---

## 📂 Dataset
The dataset is not included due to size limitations.

You can use a brain tumor MRI segmentation dataset such as:
- Kaggle Brain Tumor Segmentation Dataset

Place the dataset inside:
```
datasets/kaggle_3m/
```

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/medical-unet-segmentation.git
cd medical-unet-segmentation
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run locally
```bash
python App.py
```

### Run on Google Colab
Open `Colab_App.ipynb` and run all cells.

---

## 📈 Evaluation Metrics
- Dice Coefficient
- Intersection over Union (IoU)
- Binary Accuracy

---

## 🖼️ Sample Results
The model outputs:
- Original MRI image
- Ground truth mask
- Predicted tumor segmentation

---
