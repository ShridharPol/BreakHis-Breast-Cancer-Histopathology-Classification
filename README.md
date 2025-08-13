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

Results (Binary Classification – Patient-wise Evaluation)

Test AUC: 0.853

Test PR-AUC: 0.893

Chosen threshold: 0.2067 (Youden’s J statistic)

Image-level Classification Report
Class	Precision	Recall	F1-score	Support
Benign	0.736	0.670	0.701	382
Malignant	0.775	0.825	0.799	527

Accuracy: 0.760

Macro Avg: Precision 0.756, Recall 0.748, F1-score 0.751

Weighted Avg: Precision 0.759, Recall 0.760, F1-score 0.758

Patient-level Classification Report
Class	Precision	Recall	F1-score	Support
Benign	0.667	1.000	0.800	2
Malignant	1.000	0.857	0.923	7

Accuracy: 0.889

Macro Avg: Precision 0.833, Recall 0.929, F1-score 0.862

Weighted Avg: Precision 0.926, Recall 0.889, F1-score 0.896

Patient-level AUC: 0.929

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
## How to Reproduce

Clone the repository
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Download the dataset

BreakHis dataset (Kaggle mirror): https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset

Place it inside a data/ folder in the project root.

Install dependencies
```
pip install -r requirements.txt
```

Run the notebooks

For binary classification:
Open and run ```breakhis-binaryclassification-effnetb0.ipynb```

For multiclass classification:
Open and run ```breakhis-effnetb0-classification.ipynb```

---

## Dataset Reference

* [BreakHis Breast Cancer Histopathology Dataset — Kaggle Mirror](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)
