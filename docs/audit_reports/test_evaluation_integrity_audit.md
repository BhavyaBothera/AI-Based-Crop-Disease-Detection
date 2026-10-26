# Final Test-Set Evaluation Report Integrity Audit

**Audit Timestamp**: `2026-10-20T12:50:53.645338Z`  
**Audit Scope**: Mathematical recalculation from 15 × 15 Confusion Matrix, numerical consistency check, and report cross-verification.  
**Audit Status**: ✅ **PASSED (100% Mathematical Parity, 0 Discrepancies)**  

---

## 1. Confusion Matrix Structural Verification

| Check | Expected Value | Recomputed Value | Status |
| :--- | :---: | :---: | :---: |
| **Matrix Shape** | `(15, 15)` | `(15, 15)` | ✅ Exact |
| **Total Matrix Sum (All Elements)** | `3,413` | `3413` | ✅ Exact |
| **Diagonal Sum (Trace / Correct Predictions)** | `3,307` | `3307` | ✅ Exact |
| **Off-Diagonal Sum (Misclassifications)** | `106` | `106` | ✅ Exact |
| **Row Sums == Class Supports** | 15 / 15 Match | 15 / 15 Match | ✅ Exact |

---

## 2. Overall Metrics Recomputation & Parity

| Metric | Formula | Recomputed (from CM) | Reported JSON | Reported MD | Discrepancy (Δ) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | $\frac{\sum \text{diag}}{\sum \text{CM}} = \frac{3307}{3413}$ | `0.96894228` | `0.96894228` | `0.968942` | `0.00e+00` |
| **Macro-Precision** | $\frac{1}{15} \sum P_i$ | `0.96287643` | `0.96287643` | `0.962876` | `0.00e+00` |
| **Macro-Recall** | $\frac{1}{15} \sum R_i$ | `0.96790229` | `0.96790229` | `0.967902` | `0.00e+00` |
| **Macro-F1 (Primary)** | $\frac{1}{15} \sum F1_i$ | `0.96528801` | `0.96528801` | `0.965288` | `0.00e+00` |
| **Weighted-F1** | $\frac{\sum (F1_i \times S_i)}{3413}$ | `0.96900550` | `0.96900550` | `0.969006` | `0.00e+00` |

---

## 3. Per-Class Recomputed Metrics vs. Reported Values

| ID | Class Name | Support ($S_i$) | TP | FN | FP | Recalc Prec | Rep MD Prec | Recalc Rec | Rep MD Rec | Recalc F1 | Rep MD F1 | Δ F1 |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :---: |
| 0 | `Pepper,_bell___Bacterial_spot` | 149 | 146 | 3 | 5 | 0.9669 | 0.9669 | 0.9799 | 0.9799 | 0.9733 | 0.9733 | `0.0000` |
| 1 | `Pepper,_bell___healthy` | 221 | 217 | 4 | 3 | 0.9864 | 0.9864 | 0.9819 | 0.9819 | 0.9841 | 0.9841 | `0.0000` |
| 2 | `Potato___Early_blight` | 150 | 148 | 2 | 0 | 1.0000 | 1.0000 | 0.9867 | 0.9867 | 0.9933 | 0.9933 | `0.0000` |
| 3 | `Potato___Late_blight` | 150 | 150 | 0 | 2 | 0.9868 | 0.9868 | 1.0000 | 1.0000 | 0.9934 | 0.9934 | `0.0000` |
| 4 | `Potato___healthy` | 23 | 23 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `0.0000` |
| 5 | `Tomato___Bacterial_spot` | 319 | 309 | 10 | 8 | 0.9748 | 0.9748 | 0.9687 | 0.9687 | 0.9717 | 0.9717 | `0.0000` |
| 6 | `Tomato___Early_blight` | 150 | 129 | 21 | 24 | 0.8431 | 0.8431 | 0.8600 | 0.8600 | 0.8515 | 0.8515 | `0.0000` |
| 7 | `Tomato___Late_blight` | 285 | 272 | 13 | 10 | 0.9645 | 0.9645 | 0.9544 | 0.9544 | 0.9594 | 0.9594 | `0.0000` |
| 8 | `Tomato___Leaf_Mold` | 143 | 137 | 6 | 5 | 0.9648 | 0.9648 | 0.9580 | 0.9580 | 0.9614 | 0.9614 | `0.0000` |
| 9 | `Tomato___Septoria_leaf_spot` | 265 | 256 | 9 | 10 | 0.9624 | 0.9624 | 0.9660 | 0.9660 | 0.9642 | 0.9642 | `0.0000` |
| 10 | `Tomato___Spider_mites Two-spotted_spider_mite` | 252 | 243 | 9 | 13 | 0.9492 | 0.9492 | 0.9643 | 0.9643 | 0.9567 | 0.9567 | `0.0000` |
| 11 | `Tomato___Target_Spot` | 210 | 194 | 16 | 14 | 0.9327 | 0.9327 | 0.9238 | 0.9238 | 0.9282 | 0.9282 | `0.0000` |
| 12 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 803 | 793 | 10 | 4 | 0.9950 | 0.9950 | 0.9875 | 0.9875 | 0.9912 | 0.9912 | `0.0000` |
| 13 | `Tomato___Tomato_mosaic_virus` | 56 | 56 | 0 | 4 | 0.9333 | 0.9333 | 1.0000 | 1.0000 | 0.9655 | 0.9655 | `0.0000` |
| 14 | `Tomato___healthy` | 237 | 234 | 3 | 4 | 0.9832 | 0.9832 | 0.9873 | 0.9873 | 0.9853 | 0.9853 | `0.0000` |

---

## 4. Metric Consistency Between Validation and Test Reports

| Comparison Anchor | Value in `model_validation_comparison.json` | Value in `final_test_evaluation_report.json` | Parity Status |
| :--- | :---: | :---: | :---: |
| **Model B Validation Macro-F1** | `0.96858673` | `0.96858673` | ✅ Exact (`0.00e+00` diff) |
| **Model B Validation Accuracy** | `0.97513166` | `0.97513166` | ✅ Exact (`0.00e+00` diff) |

Both reports employ identical statistical definitions:
- Unweighted arithmetic mean across all 15 classes for macro metrics.
- Support-weighted sum for weighted metrics.
- Standard zero-division protection (`zero_division=0`).

---

## 5. Audit Conclusion

- **Mathematical Rigor**: Every single cell of the 15 × 15 confusion matrix accurately reflects the reported metrics.
- **Rounding Integrity**: All numbers presented in [`final_test_evaluation_report.md`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/final_test_evaluation_report.md) correctly round the full-precision values stored in [`final_test_evaluation_report.json`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/final_test_evaluation_report.json) without truncations or offsets.
- **Audit Verdict**: **100% Verified Consistent and Error-Free.**

