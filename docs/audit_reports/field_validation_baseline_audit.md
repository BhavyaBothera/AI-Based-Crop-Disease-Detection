# PHYTOSIGNAL — Field Validation Baseline Audit Report

**Audit Date:** 2026-11-01  
**Auditor:** Senior ML Engineer & Evaluation Specialist  
**Project:** PHYTOSIGNAL (AI-Based Crop Disease Detection and Treatment Recommendation System)  
**Status:** Baseline Audited & Verified; Field Validation Framework Established (Empirical Field Data Pending)

---

## 1. Executive Summary

This audit establishes the empirical baseline and software requirements for **PHYTOSIGNAL 2.0 Phase 1: Real-World Field Validation & Generalization Audit**.

The primary objective is to evaluate how well the existing 15-class Solanaceae disease classification pipeline—trained exclusively on controlled laboratory imagery (PlantVillage unaugmented)—generalizes to realistic field photographs of tomato, potato, and bell pepper foliage under uncontrolled environmental conditions.

> [!IMPORTANT]
> **Definitive Finding**: As of this audit, **zero real field foliar photographs** have been evaluated in this repository. All reported test metrics (e.g., Macro-F1 = 0.9653 across 3,413 images) reflect the controlled PlantVillage held-out test split only. The system is **NOT** currently field-validated. A rigorous, reproducible field validation protocol and evaluation harness must be established before any real-world performance claims can be made.

---

## 2. Existing Repository Architecture & Model Artifacts

### 2.1 Model Artifact Used in Production Inference
* **Filesystem Location:** [`ml/models/experiment_weighted/best_model.keras`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/experiment_weighted/best_model.keras)
* **File Size:** 28,392,683 bytes (~27.08 MB)
* **Metadata & Config:** [`ml/models/experiment_weighted/config.json`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/experiment_weighted/config.json)
* **Architecture:** MobileNetV2 with pre-trained ImageNet weights.
  * Layer 1: Embedded `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` mapping raw RGB `[0.0, 255.0]` to `[-1.0, 1.0]`.
  * Feature Extractor: Frozen during Stage 1 (`lr=0.001`, 15 epochs); fine-tuned during Stage 2 (`lr=1e-5`, 15 epochs).
  * Classification Head: GlobalAveragePooling2D, Dropout(0.2), Dense(15, activation='softmax').
* **Class Weighting:** Balanced inverse frequency class weights applied during training (ranging from 0.2834 for dominant class `Tomato___Tomato_Yellow_Leaf_Curl_Virus` to 10.0264 for rare class `Potato___healthy`).
* **Selection Criterion:** Selected as the top checkpoint via highest Validation Macro-F1 (`0.968587`).

### 2.2 Controlled Baseline Performance (Held-Out Test Split)
Evaluated on **3,413 isolated test images** from [`ml/data/dataset_manifest.csv`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/data/dataset_manifest.csv) ([`docs/audit_reports/final_test_evaluation_report.md`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/final_test_evaluation_report.md)):

| Primary Metric | Held-Out Test Result | Validation Result | Generalization Delta (Test − Val) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | **0.965288** | 0.968587 | −0.003299 |
| **Top-1 Accuracy** | **0.968942** (3,307 / 3,413) | 0.975132 | −0.006189 |
| **Macro-Precision** | **0.962876** | 0.963556 | −0.000680 |
| **Macro-Recall** | **0.967902** | 0.974257 | −0.006355 |
| **Total Test Support** | **3,413 images** | 3,418 images | — |

* **Controlled Confusion & Failure Patterns**:
  * Lowest Recall: `Tomato___Early_blight` (0.8600, F1 = 0.8515, 150 support) — 11 instances misclassified as `Tomato___Late_blight`, 5 as `Tomato___Target_Spot`.
  * `Tomato___Target_Spot` (Recall = 0.9238, F1 = 0.9282, 210 support) — confused with Early Blight and Bacterial Spot.
  * `Tomato___Spider_mites` (Recall = 0.9643, F1 = 0.9567, 252 support) — confused with Target Spot and Mosaic Virus.

---

## 3. Authoritative Class Mapping

The disease classifier evaluates a closed-world distribution across **15 Solanaceae taxa classes**, indexed 0 through 14 alphabetically:

| Class ID | Target Class Identifier | Crop | Condition Name | Pathological Status |
| :-: | :--- | :--- | :--- | :--- |
| **0** | `Pepper,_bell___Bacterial_spot` | Bell Pepper | Bacterial Spot | Diseased |
| **1** | `Pepper,_bell___healthy` | Bell Pepper | Healthy | Healthy |
| **2** | `Potato___Early_blight` | Potato | Early Blight | Diseased |
| **3** | `Potato___Late_blight` | Potato | Late Blight | Diseased |
| **4** | `Potato___healthy` | Potato | Healthy | Healthy |
| **5** | `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | Diseased |
| **6** | `Tomato___Early_blight` | Tomato | Early Blight | Diseased |
| **7** | `Tomato___Late_blight` | Tomato | Late Blight | Diseased |
| **8** | `Tomato___Leaf_Mold` | Tomato | Leaf Mold | Diseased |
| **9** | `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | Diseased |
| **10** | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-spotted Spider Mite | Diseased |
| **11** | `Tomato___Target_Spot` | Tomato | Target Spot | Diseased |
| **12** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Yellow Leaf Curl Virus | Diseased |
| **13** | `Tomato___Tomato_mosaic_virus` | Tomato | Mosaic Virus | Diseased |
| **14** | `Tomato___healthy` | Tomato | Healthy | Healthy |

*Source of Truth:* [`backend/app/services/inference.py`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/backend/app/services/inference.py) and [`ml/scripts/train_model.py`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/scripts/train_model.py).

---

## 4. Preprocessing Pipeline & Consistency Analysis

| Processing Step | Training Pipeline (`ml/scripts/train_model.py`) | Production Inference (`backend/app/services/inference.py`) | Consistency Status |
| :--- | :--- | :--- | :--- |
| **Decoding** | `tf.io.decode_image(..., channels=3)` | Pillow `Image.open()`, `ImageOps.exif_transpose()`, `.convert("RGB")` | **Identical** (3-channel RGB) |
| **Pixel Dimension Limit** | Controlled dataset (all 256×256) | Enforces 20 MP ceiling to prevent decompression bombs | **Compliant** |
| **Spatial Resizing** | `tf.image.resize(..., [224, 224], method='bilinear')` | `img.resize((224, 224), Image.Resampling.BILINEAR)` | **Identical** (Bilinear 224×224) |
| **Pixel Value Range** | Raw float32 array in `[0.0, 255.0]` | Raw float32 array in `[0.0, 255.0]` | **Identical** |
| **Rescaling Layer** | Inside model: `layers.Rescaling(1/127.5, -1.0)` | Inside model: `layers.Rescaling(1/127.5, -1.0)` | **Identical** |

> [!WARNING]
> **Double Rescaling Hazard**: External preprocessing must NEVER divide pixel values by 255.0 or normalize to `[-1, 1]` prior to inference. The Keras model incorporates `Rescaling(scale=1.0/127.5, offset=-1.0)` as its first layer.

---

## 5. Input Safety Gate Architecture (Level A & Level B)

The inference pipeline executes a two-tier screening process before disease classification:

1. **Level A (Deterministic Physical Integrity)**:
   - Magic-byte sniffing (`image/jpeg`, `image/png`, `image/webp`).
   - Dimension validation: `width >= 64`, `height >= 64`, `width * height <= 20,971,520`.
   - Blank/uniform canvas rejection: ITU-R BT.601 luminance population standard deviation `sigma > 2.0`.
2. **Level B (Learned Content Gate)**:
   - Model: [`ml/models/input_gate/leaf_gate.keras`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/input_gate/leaf_gate.keras) (linear head on frozen ImageNet MobileNetV2).
   - Classes: `non_leaf`, `other_plant`, `supported_leaf`.
   - Operating Point: Confidence floor `LEAF_GATE_MIN_CONFIDENCE = 0.60`.
   - Fallback: HSV plant-pigmentation and spatial distribution screen in [`backend/app/services/input_gate.py`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/backend/app/services/input_gate.py).
   - *Known Baseline Limitations*: Blocks 100% of tested portraits and leaf-free backgrounds, but has an evaluated 24.3% false acceptance rate on held-out Strawberry leaves and 46.7% on held-out Soybean leaves ([`docs/PROJECT_SUMMARY.md`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/PROJECT_SUMMARY.md#L567-L569)).

---

## 6. Dataset Split & Leakage State

The controlled dataset split is formally verified in [`docs/audit_reports/split_verification.json`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/split_verification.json):
* **Train Split:** 15,942 images
* **Validation Split:** 3,418 images
* **Test Split:** 3,413 images (locked, isolated)
* **Excluded (Non-Target Crops):** 32,677 images
* **Grouping & Near-Duplicates:** Group-aware splitting was used on dHash perceptual hash clusters. However, **five cross-class near-duplicate pairs** were documented across split boundaries in [`docs/audit_reports/near_duplicate_review.md`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/near_duplicate_review.md) due to uniform studio background similarities.

---

## 7. What Exists vs. What Is Missing

| Component | Current State | Missing for Field Generalization |
| :--- | :--- | :--- |
| **Disease Classifier** | Trained MobileNetV2, 15 classes, `0.9653` test F1 | Empirical evaluation on non-studio field images |
| **Input Gate** | 3-class Level B gate (`leaf_gate.keras`) | Evaluation against real in-field soil, weed, and canopy clutter |
| **Data Collection Protocol** | None for field imagery | Formal, standardized protocol for in-situ field sampling |
| **Field Manifest Schema** | None | Manifest schema tracking capture conditions, device, and verification |
| **Data Quality Pipeline** | Manifest auditor for PlantVillage | Field data auditor checking metadata, corruptions, and duplicate leakage |
| **Field Evaluation Runner** | None | Standalone evaluation pipeline generating subgroup performance slices |
| **Empirical Field Dataset** | None | Real, verified field foliar images of Tomato, Potato, and Bell Pepper |

---

## 8. Baseline Audit Conclusion & Roadmap

1. **The current baseline model** ([`ml/models/experiment_weighted/best_model.keras`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/experiment_weighted/best_model.keras)) is robust on uniform studio backgrounds but has **unquantified domain shift** when applied to field photography.
2. **Phase 1 Implementation Tasks**:
   - Establish `docs/field_validation/field_data_collection_protocol.md` covering realistic field collection conditions, minimum power requirements, and privacy constraints.
   - Build `ml/evaluation/field_validation/` containing manifest schemas, data auditing tools, metrics calculation engines, and evaluation runners.
   - Implement comprehensive automated unit and integration tests with synthetic fixtures.
   - Document Phase 1 results in `docs/audit_reports/field_validation_phase1_report.md` stating clearly that empirical field collection remains pending.
