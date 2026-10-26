# Dataset Audit Report: Crop Disease Collection
**Audit Generated:** 2026-10-19T02:39:10.645401  
**Dataset Directory:** `C:\Users\bhavy\OneDrive\Desktop\College\Sem 5\Project\ml\data\raw_dataset\Plant_leave_diseases_dataset_without_augmentation`  
**Audit Execution Time:** 2524.55 seconds  
**Image Analysis Engine:** Pillow (PIL) Active

> [!IMPORTANT]
> **Data Leakage Notice for Academic Evaluation:**
> This audit is the first stage of leakage prevention. Any exact duplicate groups or confirmed near-duplicate
> pairs identified below must **not** be split across training, validation, and test sets.
> Image augmentations must be performed dynamically on the training split only.

## 1. Executive Summary
| Metric | Value | Status / Description |
| :--- | :--- | :--- |
| **Total Files Audited** | 55448 | All files in directory hierarchy |
| **Valid Images** | 55448 | Verified readable image binaries |
| **Corrupted / Unreadable Images** | 0 | ✅ None |
| **Non-Image Files Found** | 0 | ✅ None |
| **Total Classes Detected** | 39 | Extracted from folder hierarchy |
| **Pre-existing Partitions** | No (Single Directory) | Layout detection |
| **Majority-Class Baseline** | **9.93%** | Minimum performance floor to beat |
| **Class Imbalance Ratio** | **36.23 : 1** | Ratio of max class to min class |

## 2. Solanaceae Crop Coverage Verification
Transparent rule matching against confirmed target crops (Tomato, Potato, Bell Pepper):

| Target Crop | Classes Found | Total Images | Status |
| :--- | :--- | :--- | :--- |
| **Tomato** | 10 classes | 18160 images | ✅ Supported |
| **Potato** | 3 classes | 2152 images | ✅ Supported |
| **Bell Pepper** | 2 classes | 2475 images | ✅ Supported |
| *Other / Non-Solanaceae* | 24 classes | 32661 images | Filter candidate |

## 3. Detected Class Breakdown
| Class Directory Name | Crop | Health Status | Images | % Representation |
| :--- | :--- | :--- | :--- | :--- |
| `Orange___Haunglongbing_(Citrus_greening)` | Other / Unmatched | Diseased | 5507 | 9.93% |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Diseased | 5357 | 9.66% |
| `Soybean___healthy` | Other / Unmatched | Healthy | 5090 | 9.18% |
| `Peach___Bacterial_spot` | Other / Unmatched | Diseased | 2297 | 4.14% |
| `Tomato___Bacterial_spot` | Tomato | Diseased | 2127 | 3.84% |
| `Tomato___Late_blight` | Tomato | Diseased | 1909 | 3.44% |
| `Squash___Powdery_mildew` | Other / Unmatched | Diseased | 1835 | 3.31% |
| `Tomato___Septoria_leaf_spot` | Tomato | Diseased | 1771 | 3.19% |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Diseased | 1676 | 3.02% |
| `Apple___healthy` | Other / Unmatched | Healthy | 1645 | 2.97% |
| `Tomato___healthy` | Tomato | Healthy | 1591 | 2.87% |
| `Blueberry___healthy` | Other / Unmatched | Healthy | 1502 | 2.71% |
| `Pepper,_bell___healthy` | Bell Pepper | Healthy | 1478 | 2.67% |
| `Tomato___Target_Spot` | Tomato | Diseased | 1404 | 2.53% |
| `Grape___Esca_(Black_Measles)` | Other / Unmatched | Diseased | 1383 | 2.49% |
| `Corn___Common_rust` | Other / Unmatched | Diseased | 1192 | 2.15% |
| `Grape___Black_rot` | Other / Unmatched | Diseased | 1180 | 2.13% |
| `Corn___healthy` | Other / Unmatched | Healthy | 1162 | 2.1% |
| `Background_without_leaves` | Other / Unmatched | Diseased | 1143 | 2.06% |
| `Strawberry___Leaf_scorch` | Other / Unmatched | Diseased | 1109 | 2.0% |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Other / Unmatched | Diseased | 1076 | 1.94% |
| `Cherry___Powdery_mildew` | Other / Unmatched | Diseased | 1052 | 1.9% |
| `Potato___Early_blight` | Potato | Diseased | 1000 | 1.8% |
| `Potato___Late_blight` | Potato | Diseased | 1000 | 1.8% |
| `Tomato___Early_blight` | Tomato | Diseased | 1000 | 1.8% |
| `Pepper,_bell___Bacterial_spot` | Bell Pepper | Diseased | 997 | 1.8% |
| `Corn___Northern_Leaf_Blight` | Other / Unmatched | Diseased | 985 | 1.78% |
| `Tomato___Leaf_Mold` | Tomato | Diseased | 952 | 1.72% |
| `Cherry___healthy` | Other / Unmatched | Healthy | 854 | 1.54% |
| `Apple___Apple_scab` | Other / Unmatched | Diseased | 630 | 1.14% |
| `Apple___Black_rot` | Other / Unmatched | Diseased | 621 | 1.12% |
| `Corn___Cercospora_leaf_spot Gray_leaf_spot` | Other / Unmatched | Diseased | 513 | 0.93% |
| `Strawberry___healthy` | Other / Unmatched | Healthy | 456 | 0.82% |
| `Grape___healthy` | Other / Unmatched | Healthy | 423 | 0.76% |
| `Tomato___Tomato_mosaic_virus` | Tomato | Diseased | 373 | 0.67% |
| `Raspberry___healthy` | Other / Unmatched | Healthy | 371 | 0.67% |
| `Peach___healthy` | Other / Unmatched | Healthy | 360 | 0.65% |
| `Apple___Cedar_apple_rust` | Other / Unmatched | Diseased | 275 | 0.5% |
| `Potato___healthy` | Potato | Healthy | 152 | 0.27% |

## 4. Image Quality & Dimension Profile
- **Resolution Range:** Min: `256x192` px | Max: `256x256` px
- **Mean Resolution:** `256.0x254.7` px
- **Square Aspect Ratio (1:1):** 54305 / 55448 images
- **Images Below Threshold (<128px):** 0
- **Image Formats Encountered:** JPEG (55447), PNG (1)

## 5. Duplicate Detection & Data Leakage Review
- **Exact Byte Duplicates (SHA-256):** 22 groups (23 redundant duplicate files).
- **Perceptual Near-Duplicate Candidates (dHash Hamming distance <= 2):** 100 candidate pairs flagged.
- **Cross-Partition Leakage Clusters:** 0 clusters cross train/val/test boundaries.

### Sample Exact Duplicate Groups
**Group 1 (SHA-256: `6b242d89b134...`):**
  - `Apple___healthy\image (1120).JPG`
  - `Apple___healthy\image (163).JPG`
**Group 2 (SHA-256: `5d486b62abe5...`):**
  - `Apple___healthy\image (1272).JPG`
  - `Apple___healthy\image (908).JPG`
**Group 3 (SHA-256: `d4b74359138a...`):**
  - `Apple___healthy\image (1335).JPG`
  - `Apple___healthy\image (1447).JPG`
**Group 4 (SHA-256: `262198126fd6...`):**
  - `Apple___healthy\image (1552).JPG`
  - `Apple___healthy\image (1631).JPG`
**Group 5 (SHA-256: `767bbeee0c31...`):**
  - `Apple___healthy\image (375).JPG`
  - `Apple___healthy\image (846).JPG`

### Sample Near-Duplicate Candidates (Requires Review)
| Image A | Image B | Class A | Class B | Distance | Cross-Split Leak? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Apple___Apple_scab\image (123).JPG` | `Apple___Apple_scab\image (411).JPG` | Apple___Apple_scab | Apple___Apple_scab | 2 | No |
| `Apple___Apple_scab\image (279).JPG` | `Apple___Apple_scab\image (90).JPG` | Apple___Apple_scab | Apple___Apple_scab | 2 | No |
| `Apple___Cedar_apple_rust\image (272).JPG` | `Blueberry___healthy\image (205).JPG` | Apple___Cedar_apple_rust | Blueberry___healthy | 2 | No |
| `Apple___healthy\image (1120).JPG` | `Apple___healthy\image (163).JPG` | Apple___healthy | Apple___healthy | 0 | No |
| `Apple___healthy\image (1131).JPG` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus\image (2880).JPG` | Apple___healthy | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 2 | No |
| `Apple___healthy\image (1272).JPG` | `Apple___healthy\image (908).JPG` | Apple___healthy | Apple___healthy | 0 | No |
| `Apple___healthy\image (1335).JPG` | `Apple___healthy\image (1447).JPG` | Apple___healthy | Apple___healthy | 0 | No |
| `Apple___healthy\image (1552).JPG` | `Apple___healthy\image (1631).JPG` | Apple___healthy | Apple___healthy | 0 | No |
| `Apple___healthy\image (16).JPG` | `Potato___Late_blight\image (898).JPG` | Apple___healthy | Potato___Late_blight | 2 | No |
| `Apple___healthy\image (1600).JPG` | `Apple___healthy\image (250).JPG` | Apple___healthy | Apple___healthy | 2 | No |

## 6. Recommendations for Phase 2 (ML Pipeline)
1. **Crops Approved:** Tomato, Potato, and Bell Pepper are all present with viable representation.
2. **Baseline Target:** Model evaluation must significantly exceed the **9.93%** majority-class baseline.
3. **Macro-F1 Optimization:** Due to class imbalance, monitor Macro-F1 rather than raw accuracy.
4. **Split Sanitization:** Deduplicate identical copies before creating stratified splits.
