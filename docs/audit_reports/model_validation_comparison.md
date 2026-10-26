# Validation Set Model Comparison: Unweighted vs. Balanced Class-Weighted

**Generated**: `2026-10-20T12:18:01.535063Z`  
**Split Evaluated**: Validation split only (`split == 'val'`)  
**Validation Sample Count**: **3,418** images  
**Test Set Isolation**: **3,413** test images strictly isolated (0 test images loaded or decoded)  

---

## 1. Overall Performance Comparison

| Metric | Unweighted (Model A) | Balanced Weighted (Model B) | Delta (B - A) | Superior Model |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | `0.963411` | `0.968587` | `+0.005175` | **Balanced Weighted** |
| **Accuracy** | `0.971913` | `0.975132` | `+0.003218` | **Balanced Weighted** |
| **Macro-Precision** | `0.963716` | `0.963556` | `-0.000160` | **Unweighted** |
| **Macro-Recall** | `0.963383` | `0.974257` | `+0.010874` | **Balanced Weighted** |
| **Weighted-F1** | `0.971825` | `0.975168` | `+0.003343` | **Balanced Weighted** |
| **Validation Loss** | `0.170589` | `0.174981` | `+0.004392` | **Unweighted** |

---

## 2. Checkpoint Reconciliation

| Experiment | Checkpoint Saved Macro-F1 | Re-evaluated Macro-F1 | Discrepancy (Δ) | Checkpoint Loss | Re-evaluated Loss | Discrepancy (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted** | `0.963411` | `0.963411` | `+0.00e+00` | `0.170589` | `0.170589` | `+0.00e+00` |
| **Class-Weighted** | `0.968587` | `0.968587` | `+0.00e+00` | `0.174981` | `0.174981` | `+0.00e+00` |

*Note: Any negligible floating-point discrepancies (~1e-7 or lower) are standard floating point accumulation variations during non-deterministic GPU/CPU matrix summation; the weights and predictions are 100% identical.*

---

## 3. Per-Class Performance Breakdown (All 15 Classes)

| ID | Class Name | Support | Unweighted Prec | Unweighted Rec | Unweighted F1 | Weighted Prec | Weighted Rec | Weighted F1 | F1 Delta (B - A) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | `Pepper,_bell___Bacterial_spot` | 150 | 0.9801 | 0.9867 | 0.9834 | 0.9735 | 0.9800 | 0.9767 | `-0.0066` |
| 1 | `Pepper,_bell___healthy` | 222 | 0.9910 | 0.9910 | 0.9910 | 0.9865 | 0.9910 | 0.9888 | `-0.0022` |
| 2 | `Potato___Early_blight` | 150 | 0.9803 | 0.9933 | 0.9868 | 0.9934 | 1.0000 | 0.9967 | `+0.0099` |
| 3 | `Potato___Late_blight` | 150 | 0.9667 | 0.9667 | 0.9667 | 0.9865 | 0.9733 | 0.9799 | `+0.0132` |
| 4 | `Potato___healthy` | 23 | 0.9130 | 0.9130 | 0.9130 | 0.8846 | 1.0000 | 0.9388 | `+0.0257` |
| 5 | `Tomato___Bacterial_spot` | 319 | 0.9812 | 0.9843 | 0.9828 | 0.9906 | 0.9875 | 0.9890 | `+0.0062` |
| 6 | `Tomato___Early_blight` | 150 | 0.9286 | 0.8667 | 0.8966 | 0.8954 | 0.9133 | 0.9043 | `+0.0077` |
| 7 | `Tomato___Late_blight` | 285 | 0.9645 | 0.9544 | 0.9594 | 0.9746 | 0.9439 | 0.9590 | `-0.0004` |
| 8 | `Tomato___Leaf_Mold` | 143 | 0.9645 | 0.9510 | 0.9577 | 0.9583 | 0.9650 | 0.9617 | `+0.0039` |
| 9 | `Tomato___Septoria_leaf_spot` | 266 | 0.9769 | 0.9549 | 0.9658 | 0.9846 | 0.9624 | 0.9734 | `+0.0076` |
| 10 | `Tomato___Spider_mites Two-spotted_spider_mite` | 251 | 0.9526 | 0.9602 | 0.9563 | 0.9563 | 0.9602 | 0.9583 | `+0.0019` |
| 11 | `Tomato___Target_Spot` | 211 | 0.9091 | 0.9479 | 0.9281 | 0.9439 | 0.9573 | 0.9506 | `+0.0225` |
| 12 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 804 | 0.9901 | 0.9975 | 0.9938 | 0.9925 | 0.9925 | 0.9925 | `-0.0013` |
| 13 | `Tomato___Tomato_mosaic_virus` | 56 | 0.9655 | 1.0000 | 0.9825 | 0.9492 | 1.0000 | 0.9739 | `-0.0085` |
| 14 | `Tomato___healthy` | 238 | 0.9915 | 0.9832 | 0.9873 | 0.9833 | 0.9874 | 0.9853 | `-0.0020` |

---

## 4. Underrepresented and Low-Performing Class Analysis

### Key Underrepresented Classes:
1. **Class 4 (`Potato___healthy`)** — Support: **23** validation samples (106 training samples, class weight: `10.026415`)
   * Unweighted      : Precision=0.9130, Recall=0.9130, F1=0.9130
   * Class-Weighted  : Precision=0.8846, Recall=1.0000, F1=0.9388
   * Impact of Weighting: Recall changed by `+0.0870`, F1 changed by `+0.0257`.

2. **Class 13 (`Tomato___Tomato_mosaic_virus`)** — Support: **56** validation samples (261 training samples, class weight: `4.072031`)
   * Unweighted      : Precision=0.9655, Recall=1.0000, F1=0.9825
   * Class-Weighted  : Precision=0.9492, Recall=1.0000, F1=0.9739
   * Impact of Weighting: Recall changed by `+0.0000`, F1 changed by `-0.0085`.

### Classes with Lowest Recall:
- **Unweighted Lowest Recall**: `Tomato___Early_blight` (0.8667), `Potato___healthy` (0.9130), `Tomato___Target_Spot` (0.9479)
- **Class-Weighted Lowest Recall**: `Tomato___Early_blight` (0.9133), `Tomato___Late_blight` (0.9439), `Tomato___Target_Spot` (0.9573)

---

## 5. Confusion Matrices (15 x 15)

Axes definition:
- **Rows (Vertical)**: Ground Truth Label (Classes 0 to 14)
- **Columns (Horizontal)**: Model Prediction (Classes 0 to 14)

### Unweighted Model Confusion Matrix
```text
Pred:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
--------------------------------------------------------------------------------
T 0 :  148    2    0    0    0    0    0    0    0    0    0    0    0    0    0
T 1 :    2  220    0    0    0    0    0    0    0    0    0    0    0    0    0
T 2 :    0    0  149    1    0    0    0    0    0    0    0    0    0    0    0
T 3 :    0    0    1  145    2    0    0    2    0    0    0    0    0    0    0
T 4 :    0    0    0    2   21    0    0    0    0    0    0    0    0    0    0
T 5 :    0    0    0    0    0  314    1    0    0    0    0    1    3    0    0
T 6 :    0    0    1    0    0    2  130    7    1    3    0    4    1    1    0
T 7 :    0    0    0    2    0    1    4  272    2    0    0    2    2    0    0
T 8 :    0    0    0    0    0    1    1    1  136    1    2    1    0    0    0
T 9 :    0    0    1    0    0    2    2    0    2  254    0    3    0    1    1
T10 :    0    0    0    0    0    0    0    0    0    0  241    7    2    0    1
T11 :    1    0    0    0    0    0    2    0    0    1    7  200    0    0    0
T12 :    0    0    0    0    0    0    0    0    0    0    2    0  802    0    0
T13 :    0    0    0    0    0    0    0    0    0    0    0    0    0   56    0
T14 :    0    0    0    0    0    0    0    0    0    1    1    2    0    0  234
```

### Balanced Class-Weighted Model Confusion Matrix
```text
Pred:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
--------------------------------------------------------------------------------
T 0 :  147    3    0    0    0    0    0    0    0    0    0    0    0    0    0
T 1 :    2  220    0    0    0    0    0    0    0    0    0    0    0    0    0
T 2 :    0    0  150    0    0    0    0    0    0    0    0    0    0    0    0
T 3 :    0    0    0  146    3    0    0    1    0    0    0    0    0    0    0
T 4 :    0    0    0    0   23    0    0    0    0    0    0    0    0    0    0
T 5 :    0    0    0    0    0  315    2    0    0    0    0    0    2    0    0
T 6 :    0    0    1    0    0    0  137    4    2    3    0    1    1    1    0
T 7 :    0    0    0    2    0    0   10  269    1    0    0    3    0    0    0
T 8 :    0    0    0    0    0    0    0    1  138    1    1    0    1    1    0
T 9 :    0    0    0    0    0    1    2    0    3  256    0    3    0    0    1
T10 :    0    0    0    0    0    0    0    0    0    0  241    5    2    1    2
T11 :    1    0    0    0    0    0    1    0    0    0    6  202    0    0    1
T12 :    0    0    0    0    0    2    1    0    0    0    3    0  798    0    0
T13 :    0    0    0    0    0    0    0    0    0    0    0    0    0   56    0
T14 :    1    0    0    0    0    0    0    1    0    0    1    0    0    0  235
```

---

## 6. Model Recommendation Rationale

**Recommended Candidate**: **Balanced Class-Weighted (Model B)**  
**Decision Basis**: Higher Validation Macro-F1 (0.968587 vs. 0.963411, delta: +0.005175).  

> [!NOTE]
> In strict adherence to experimental safety rules, test set images (3,413 images) have not been evaluated.

