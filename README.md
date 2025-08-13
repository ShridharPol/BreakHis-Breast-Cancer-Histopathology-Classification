# BreakHis-Breast-Cancer-Histopathology-Classification


This repository contains two EfficientNetB0-based deep learning pipelines trained on the [BreakHis Breast Cancer Histopathology Dataset (Kaggle Mirror)](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset).

The project explores:

1. **Binary classification** → Benign vs. Malignant
2. **Multiclass classification** → 8 tumor subtypes

Both pipelines use **patient-wise splits** to avoid image leakage between training, validation, and test sets.

---

## 1. Binary Classification — Benign vs Malignant

**Notebook:** `breakhis-binaryclassification-effnetb0.ipynb`

### **Objective**

Train a robust binary classifier to distinguish between benign and malignant breast tumor histopathology images.

### **Data Preparation**

* Used the **binary folder structure** from the BreakHis dataset.
* Included images from **all four magnifications**: 40X, 100X, 200X, 400X.
* Applied **patient-wise stratified split**:

  * 70% train, 15% validation, 15% test (by patient ID, stratified to preserve benign/malignant ratio).
* Data augmentation: simple flips, brightness, and contrast changes.

### **Model**

* **Base:** EfficientNetB0 (`imagenet` weights, frozen for initial training).
* **Top Layers:** GlobalAveragePooling → Dense(512, relu) → Dropout → Dense(1, sigmoid).
* **Loss:** Binary cross-entropy
* **Optimizer:** Adam
* **Metrics:** Accuracy, AUC, Precision, Recall, F1.

### **Results**

* **Test AUC:** 0.853
* **Test PR-AUC:** 0.893
* **Chosen threshold:** 0.2067 (Youden’s J statistic)
* **Confusion Matrix:**

  ```
  [[220 162]
   [ 59 468]]
  ```
* Strong malignant recall (0.888), slightly lower benign recall (0.576).

---

## 2. Multiclass Classification — 8 Tumor Subtypes

**Notebook:** `breakhis-effnetb0-classification.ipynb`

### **Objective**

Classify histopathology images into **8 distinct tumor subtypes**.

**Classes:**

1. Adenosis
2. Fibroadenoma
3. Phyllodes Tumor
4. Tubular Adenoma
5. Ductal Carcinoma
6. Lobular Carcinoma
7. Mucinous Carcinoma
8. Papillary Carcinoma

### **Data Preparation**

* Used **all images** across magnifications.
* **Patient-wise stratified split** via `StratifiedGroupKFold` in **two stages**:

  1. Train+Val (85%) vs Test (15%) — patient-wise
  2. Train (70% of total) vs Val (15% of total) — patient-wise
* Ensured **no patient overlap** between splits.

### **Model**

* **Base:** EfficientNetB0 (`imagenet` weights, frozen for initial training).
* **Top Layers:** GlobalAveragePooling → Dense(512, relu) → Dropout → Dense(8, softmax).
* **Loss:** Sparse categorical cross-entropy
* **Optimizer:** Adam
* **Metrics:** Accuracy, Precision, Recall, F1.

### **Results**

* **Micro Avg F1:** 0.788
* **Macro Avg F1:** 0.327 (indicating performance imbalance across classes).
* **Observation:**

  * Classes with abundant samples (e.g., Ductal Carcinoma) achieved strong performance.
  * Rare classes (e.g., Adenosis, Fibroadenoma) had **zero recall** due to severe imbalance.
* **Reason for Challenge:**
  The dataset is **highly imbalanced** at the subtype level, making it difficult for a single model to generalize without heavy augmentation, resampling, or specialized loss functions.

---

## Key Learnings

* **Binary classification** is more robust on the BreakHis dataset due to better class balance and clearer separation of features.
* **Multiclass classification** struggles with severe class imbalance and rarity of certain subtypes.
* Patient-wise splitting is essential to prevent data leakage in medical imaging tasks.
* Simple augmentations suffice for binary, but multiclass likely needs targeted augmentation or class-aware sampling.

---

## Repository Structure

```
.
├── breakhis-binaryclassification-effnetb0.ipynb
├── breakhis-effnetb0-classification.ipynb
└── README.md
```

---

## Dataset Reference

* [BreakHis Breast Cancer Histopathology Dataset — Kaggle Mirror](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)
