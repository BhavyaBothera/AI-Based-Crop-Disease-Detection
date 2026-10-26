# Dataset Preparation & Stratified Split Report
**Generated:** 2026-10-19T11:19:58.956583  
**Random Seed:** `42` (100% deterministic)  
**Target Proportions:** 70% Train / 15% Validation / 15% Test  

> [!IMPORTANT]
> **Zero Data Leakage Certified:**
> - Train, Validation, and Test sets have **0 SHA-256 byte overlaps**.
> - All confirmed exact duplicate and near-duplicate groups are **100% group-locked** within single splits.
> - No image augmentations have been saved to disk; augmentations will occur dynamically on training data only.

## 1. Dataset Reconciliation & Inclusion Summary
| Metric | Image Count | Description / Action |
| :--- | :--- | :--- |
| **Total Images in Source Archive** | 55,448 | Full PlantVillage unaugmented archive |
| **Non-Target Crop Images Excluded** | 32,661 | Preserved in place; marked `excluded` (non-target) |
| **Target Solanaceae Images** | 22,787 | Tomato, Potato, and Bell Pepper images |
| **Exact Byte Duplicates Excluded** | 14 | Redundant byte clones excluded |
| **Target Images Included for Modeling** | **22,773** | Clean, unique, leak-free image pool |
| **Reconciliation Status** | ✅ Verified Exact | Included + Excluded = Total Archive |

## 2. Overall Partition Allocations
| Split | Image Count | Realized % | Target % | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 15,942 | 70.0% | 70.0% | Model optimization & dynamic augmentation |
| **Validation** | 3,418 | 15.01% | 15.0% | Hyperparameter tuning & checkpoint selection |
| **Test** | 3,413 | 14.99% | 15.0% | Isolated academic performance benchmark |
| **Total Included** | **22,773** | **100.0%** | **100.0%** | |

## 3. Per-Class Split Allocation (15 Solanaceae Classes)
| Class Name | Crop | Total Unique | Train (70%) | Val (15%) | Test (15%) | % Train |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Pepper,_bell___Bacterial_spot` | Bell Pepper | 997 | 698 | 150 | 149 | 70.0% |
| `Pepper,_bell___healthy` | Bell Pepper | 1,478 | 1,035 | 222 | 221 | 70.0% |
| `Potato___Early_blight` | Potato | 1,000 | 700 | 150 | 150 | 70.0% |
| `Potato___Late_blight` | Potato | 1,000 | 700 | 150 | 150 | 70.0% |
| `Potato___healthy` | Potato | 152 | 106 | 23 | 23 | 69.7% |
| `Tomato___Bacterial_spot` | Tomato | 2,127 | 1,489 | 319 | 319 | 70.0% |
| `Tomato___Early_blight` | Tomato | 1,000 | 700 | 150 | 150 | 70.0% |
| `Tomato___Late_blight` | Tomato | 1,901 | 1,331 | 285 | 285 | 70.0% |
| `Tomato___Leaf_Mold` | Tomato | 952 | 666 | 143 | 143 | 70.0% |
| `Tomato___Septoria_leaf_spot` | Tomato | 1,771 | 1,240 | 266 | 265 | 70.0% |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | 1,676 | 1,173 | 251 | 252 | 70.0% |
| `Tomato___Target_Spot` | Tomato | 1,404 | 983 | 211 | 210 | 70.0% |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | 5,357 | 3,750 | 804 | 803 | 70.0% |
| `Tomato___Tomato_mosaic_virus` | Tomato | 373 | 261 | 56 | 56 | 70.0% |
| `Tomato___healthy` | Tomato | 1,585 | 1,110 | 238 | 237 | 70.0% |

## 4. Leakage Verification Results
- **Train ∩ Val SHA-256 Overlap:** 0 (Passed)
- **Train ∩ Test SHA-256 Overlap:** 0 (Passed)
- **Val ∩ Test SHA-256 Overlap:** 0 (Passed)
- **Duplicate Clusters Split Across Partitions:** 0 (Passed)
- **Zero Leakage Certification:** ✅ CERTIFIED CLEAN

## 5. Exact Duplicate Exclusions
14 duplicate groups (28 images) were detected in target crops. The first sorted path was kept as canonical; the redundant copy was excluded:

| Group ID | Class | Canonical File (Included) | Redundant Copy (Excluded) |
| :--- | :--- | :--- | :--- |
| `GRP_EXACT_001` | `Tomato___Late_blight` | `image (109).JPG` | `image (54).JPG` |
| `GRP_EXACT_002` | `Tomato___Late_blight` | `image (1188).jpg` | `image (1485).jpg` |
| `GRP_EXACT_003` | `Tomato___Late_blight` | `image (1400).jpg` | `image (590).jpg` |
| `GRP_EXACT_004` | `Tomato___Late_blight` | `image (1578).JPG` | `image (1627).JPG` |
| `GRP_EXACT_005` | `Tomato___Late_blight` | `image (1598).jpg` | `image (882).jpg` |
| `GRP_EXACT_006` | `Tomato___Late_blight` | `image (1704).JPG` | `image (275).JPG` |
| `GRP_EXACT_007` | `Tomato___Late_blight` | `image (1748).JPG` | `image (971).JPG` |
| `GRP_EXACT_008` | `Tomato___Late_blight` | `image (182).jpg` | `image (264).jpg` |
| `GRP_EXACT_009` | `Tomato___healthy` | `image (1027).JPG` | `image (682).JPG` |
| `GRP_EXACT_010` | `Tomato___healthy` | `image (1305).JPG` | `image (55).JPG` |
| `GRP_EXACT_011` | `Tomato___healthy` | `image (1311).JPG` | `image (984).JPG` |
| `GRP_EXACT_012` | `Tomato___healthy` | `image (1457).JPG` | `image (477).JPG` |
| `GRP_EXACT_013` | `Tomato___healthy` | `image (474).JPG` | `image (553).JPG` |
| `GRP_EXACT_014` | `Tomato___healthy` | `image (943).JPG` | `image (992).JPG` |
