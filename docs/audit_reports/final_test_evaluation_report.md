# Final Test-Set Evaluation Report
## AI-Based Crop Disease Detection and Treatment Recommendation System

**Evaluation Date**: `2026-10-20T12:36:25.445049Z`  
**Selected Model**: [`ml/models/experiment_weighted/best_model.keras`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/experiment_weighted/best_model.keras)  
**Selection Criterion**: Highest Validation Macro-F1 (`0.968587`)  
**Dataset Evaluated**: Fixed Test Split (`split == 'test'`) from [`ml/data/dataset_manifest.csv`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/data/dataset_manifest.csv)  
**Total Test Samples**: **3,413** images  
**Evaluation Duration**: **19.04 seconds**  

---

## 1. Executive Summary & Primary Metrics

| Primary Metric | Test Set Result | Validation Result (Model B) | Generalization Delta (Test − Val) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | **`0.965288`** | `0.968587` | **`-0.003299`** |
| **Accuracy** | **`0.968942`** (3,307 / 3,413) | `0.975132` | **`-0.006189`** |
| **Macro-Precision** | **`0.962876`** | `0.963556` | **`-0.000680`** |
| **Macro-Recall** | **`0.967902`** | `0.974257` | **`-0.006355`** |
| **Weighted-F1** | **`0.969006`** | `0.975168` | **`-0.006163`** |
| **Total Compiled Loss** | **`0.187019`** | `0.174981` | `+0.012037` |
| **Pure Cross-Entropy Loss** | **`0.102802`** | `0.087627` | `+0.015175` |

- **Correct Classifications**: **3,307** / 3,413 (96.89%)
- **Misclassifications**: **106** / 3,413 (3.11%)

---

## 2. Per-Class Test Performance (All 15 Solanaceae Classes)

| ID | Class Name | Support | Correct | Errors | Precision | Recall | F1-Score |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | `Pepper,_bell___Bacterial_spot` | 149 | 146 | 3 | 0.9669 | 0.9799 | 0.9733 |
| 1 | `Pepper,_bell___healthy` | 221 | 217 | 4 | 0.9864 | 0.9819 | 0.9841 |
| 2 | `Potato___Early_blight` | 150 | 148 | 2 | 1.0000 | 0.9867 | 0.9933 |
| 3 | `Potato___Late_blight` | 150 | 150 | 0 | 0.9868 | 1.0000 | 0.9934 |
| 4 | `Potato___healthy` | 23 | 23 | 0 | 1.0000 | 1.0000 | 1.0000 |
| 5 | `Tomato___Bacterial_spot` | 319 | 309 | 10 | 0.9748 | 0.9687 | 0.9717 |
| 6 | `Tomato___Early_blight` | 150 | 129 | 21 | 0.8431 | 0.8600 | 0.8515 |
| 7 | `Tomato___Late_blight` | 285 | 272 | 13 | 0.9645 | 0.9544 | 0.9594 |
| 8 | `Tomato___Leaf_Mold` | 143 | 137 | 6 | 0.9648 | 0.9580 | 0.9614 |
| 9 | `Tomato___Septoria_leaf_spot` | 265 | 256 | 9 | 0.9624 | 0.9660 | 0.9642 |
| 10 | `Tomato___Spider_mites Two-spotted_spider_mite` | 252 | 243 | 9 | 0.9492 | 0.9643 | 0.9567 |
| 11 | `Tomato___Target_Spot` | 210 | 194 | 16 | 0.9327 | 0.9238 | 0.9282 |
| 12 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 803 | 793 | 10 | 0.9950 | 0.9875 | 0.9912 |
| 13 | `Tomato___Tomato_mosaic_virus` | 56 | 56 | 0 | 0.9333 | 1.0000 | 0.9655 |
| 14 | `Tomato___healthy` | 237 | 234 | 3 | 0.9832 | 0.9873 | 0.9853 |

---

## 3. Challenging & Rare Class Analysis

### Highly Underrepresented Class Performance:
1. **Class 4 (`Potato___healthy`)** — Test Support: **23 images** (106 Train / 23 Val)
   * Precision: **`1.0000`**
   * Recall: **`1.0000`** (23 / 23 correct)
   * F1-Score: **`1.0000`**
   * Errors: 0 false negatives, 0 false positives.

2. **Class 13 (`Tomato___Tomato_mosaic_virus`)** — Test Support: **56 images** (261 Train / 56 Val)
   * Precision: **`0.9333`**
   * Recall: **`1.0000`** (56 / 56 correct)
   * F1-Score: **`0.9655`**

### Lowest-Performing Classes on Test Split:
- **Lowest Recall**:
  * `Tomato___Early_blight`: Recall = **`0.8600`**, F1 = **`0.8515`** (Support: 150)
  * `Tomato___Target_Spot`: Recall = **`0.9238`**, F1 = **`0.9282`** (Support: 210)
  * `Tomato___Late_blight`: Recall = **`0.9544`**, F1 = **`0.9594`** (Support: 285)
- **Lowest F1-Score**:
  * `Tomato___Early_blight`: F1 = **`0.8515`**, Recall = **`0.8600`** (Support: 150)
  * `Tomato___Target_Spot`: F1 = **`0.9282`**, Recall = **`0.9238`** (Support: 210)
  * `Tomato___Spider_mites Two-spotted_spider_mite`: F1 = **`0.9567`**, Recall = **`0.9643`** (Support: 252)

---

## 4. Test Set 15 × 15 Confusion Matrix

Axes definition:
- **Rows (Vertical)**: Ground Truth Label ($T_0$ to $T_{14}$)
- **Columns (Horizontal)**: Model Prediction ($0$ to $14$)

```text
Pred:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
--------------------------------------------------------------------------------
T 0 :  146    2    0    0    0    0    0    0    0    0    1    0    0    0    0
T 1 :    4  217    0    0    0    0    0    0    0    0    0    0    0    0    0
T 2 :    0    0  148    1    0    0    1    0    0    0    0    0    0    0    0
T 3 :    0    0    0  150    0    0    0    0    0    0    0    0    0    0    0
T 4 :    0    0    0    0   23    0    0    0    0    0    0    0    0    0    0
T 5 :    0    0    0    0    0  309    4    2    0    2    0    1    1    0    0
T 6 :    0    1    0    1    0    0  129    6    2    4    2    3    1    1    0
T 7 :    0    0    0    0    0    0   11  272    1    0    0    0    0    0    1
T 8 :    0    0    0    0    0    1    1    0  137    1    2    1    0    0    0
T 9 :    0    0    0    0    0    1    1    0    0  256    1    3    1    0    2
T10 :    0    0    0    0    0    0    0    1    1    0  243    4    1    2    0
T11 :    1    0    0    0    0    1    5    0    0    2    5  194    0    1    1
T12 :    0    0    0    0    0    5    1    0    1    1    1    1  793    0    0
T13 :    0    0    0    0    0    0    0    0    0    0    0    0    0   56    0
T14 :    0    0    0    0    0    0    0    1    0    0    1    1    0    0  234
```

---

## 5. Methodological Integrity & Dataset Audit Caveats

> [!IMPORTANT]
> **Dataset Audit Caveat Preserved**:  
> The Phase 2 dataset audit documented five cross-class near-duplicate pairs across split boundaries awaiting human review. Complete duplicate-free separation is not claimed.

> [!NOTE]
> **Strict Finality**:  
> This test evaluation was conducted exactly once on the pre-selected model without hyperparameter adjustments, threshold tuning, or test-driven iteration.

