# Near-Duplicate Candidate Review Log

**Audit Scope:** 5 cross-class candidate pairs with Difference Hash (dHash) distance = 2 crossing train/validation/test partitions.  
**Review Date:** 2026-10-19  
**Visual Contact Sheet:** [`docs/audit_reports/near_duplicates_contact_sheet.png`](near_duplicates_contact_sheet.png)  
**Dataset Version:** PlantVillage Unaugmented (Seed 42 Splits)  

---

## 1. Candidate Review Summary

| Pair | Image A (Split) | Image B (Split) | Class A vs Class B | dHash Dist | Human-Review Status | Automated Finding & Disposition Notes |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `image (1699).JPG` [**train**] | `image (836).JPG` [**val**] | `Tomato___Bacterial_spot` vs `Tomato___Yellow_Leaf_Curl_Virus` | 2 | `pending` | Different disease pathologies; avg RGB diff 40.3/255. Background illumination similarity on studio backdrop. |
| **2** | `image (722).JPG` [**val**] | `image (3986).JPG` [**train**] | `Tomato___Early_blight` vs `Tomato___Yellow_Leaf_Curl_Virus` | 2 | `pending` | Different lesion morphology (concentric rings vs curling/chlorosis); avg RGB diff 28.1/255. |
| **3** | `image (1605).JPG` [**test**] | `image (745).JPG` [**train**] | `Tomato___Late_blight` vs `Tomato___healthy` | 2 | `pending` | Necrotic water-soaked lesion vs symptomless healthy leaf; avg RGB diff 19.3/255. Lowest pixel difference, requires human inspection. |
| **4** | `image (756).JPG` [**train**] | `image (1604).JPG` [**test**] | `Tomato___Leaf_Mold` vs `Tomato___Yellow_Leaf_Curl_Virus` | 2 | `pending` | Yellowish upper-leaf spots vs upward leaf curling; avg RGB diff 28.0/255. Background alignment artifact. |
| **5** | `image (975).JPG` [**test**] | `image (1139).JPG` [**train**] | `Tomato___Spider_mites` vs `Tomato___Target_Spot` | 2 | `pending` | Fine white speckling/stippling vs dark target-like lesions; avg RGB diff 36.0/255. Distinct visual symptoms. |

---

## 2. Detailed Pair-by-Pair Records

### Pair 1
* **Image A:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Bacterial_spot/image (1699).JPG`
  * Class: `Tomato___Bacterial_spot` (ID: 5)
  * Split: **`train`**
* **Image B:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (836).JPG`
  * Class: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (ID: 12)
  * Split: **`val`**
* **Metrics:** dHash Hamming Distance: `2` | Mean Absolute Pixel Difference (MAE): `40.3 / 255.0`
* **Human-Review Status:** `pending`
* **Review Disposition & Rationale:** Both images feature dark studio backgrounds, which caused low-frequency 8×8 gradient hashing to match at distance 2. Pixel differences are large (40.3), and the symptoms represent two biologically distinct conditions (bacterial spotting vs viral leaf curl). Retained as separate samples pending manual sign-off.

---

### Pair 2
* **Image A:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Early_blight/image (722).JPG`
  * Class: `Tomato___Early_blight` (ID: 6)
  * Split: **`val`**
* **Image B:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (3986).JPG`
  * Class: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (ID: 12)
  * Split: **`train`**
* **Metrics:** dHash Hamming Distance: `2` | Mean Absolute Pixel Difference (MAE): `28.1 / 255.0`
* **Human-Review Status:** `pending`
* **Review Disposition & Rationale:** Early blight shows fungal concentric necrotic lesions, whereas Yellow Leaf Curl shows severe marginal leaf curl and vein yellowing. Retained across splits pending manual sign-off.

---

### Pair 3
* **Image A:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Late_blight/image (1605).JPG`
  * Class: `Tomato___Late_blight` (ID: 7)
  * Split: **`test`**
* **Image B:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___healthy/image (745).JPG`
  * Class: `Tomato___healthy` (ID: 14)
  * Split: **`train`**
* **Metrics:** dHash Hamming Distance: `2` | Mean Absolute Pixel Difference (MAE): `19.3 / 255.0`
* **Human-Review Status:** `pending`
* **Review Disposition & Rationale:** One image depicts an uninfected healthy leaflet while the other exhibits late blight pathogen spots. With an average pixel difference of 19.3, this pair has the highest visual similarity among the five and is flagged for human review on the contact sheet.

---

### Pair 4
* **Image A:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Leaf_Mold/image (756).JPG`
  * Class: `Tomato___Leaf_Mold` (ID: 8)
  * Split: **`train`**
* **Image B:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Tomato_Yellow_Leaf_Curl_Virus/image (1604).JPG`
  * Class: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (ID: 12)
  * Split: **`test`**
* **Metrics:** dHash Hamming Distance: `2` | Mean Absolute Pixel Difference (MAE): `28.0 / 255.0`
* **Human-Review Status:** `pending`
* **Review Disposition & Rationale:** Distinct fungal leaf mold patches vs viral curling. Centered leaf placement on the standard black sheet created coarse gradient similarity. Retained pending human sign-off.

---

### Pair 5
* **Image A:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Spider_mites Two-spotted_spider_mite/image (975).JPG`
  * Class: `Tomato___Spider_mites Two-spotted_spider_mite` (ID: 10)
  * Split: **`test`**
* **Image B:** `ml/data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Tomato___Target_Spot/image (1139).JPG`
  * Class: `Tomato___Target_Spot` (ID: 11)
  * Split: **`train`**
* **Metrics:** dHash Hamming Distance: `2` | Mean Absolute Pixel Difference (MAE): `36.0 / 255.0`
* **Human-Review Status:** `pending`
* **Review Disposition & Rationale:** Spider mite chlorotic micro-speckling vs discrete circular fungal target spots. Distinct pathologies. Retained pending human sign-off.

---

## 3. Contact Sheet Artifact
A side-by-side visual comparison contact sheet has been generated and saved at:  
[`docs/audit_reports/near_duplicates_contact_sheet.png`](near_duplicates_contact_sheet.png)

It displays Image A on the left and Image B on the right for all 5 pairs, labeled with their respective classes, filenames, and split assignments (`TRAIN`, `VAL`, `TEST`).

---

## 4. Remaining Limitations & Scientific Assessment
1. **Preservation of Raw Data:** Neither candidate images nor manifest splits have been altered based solely on perceptual-hash distance.
2. **Coarse Hash Artifacts:** 64-bit dHash computes 8×8 horizontal gradient transitions. In laboratory plant pathology datasets like PlantVillage where leaves are photographed on identical flat black background paper, unrelated leaves with similar silhouette bounding boxes naturally produce small dHash Hamming distances ($\le 2$).
3. **Leakage Risk Impact:** Because all 5 pairs belong to **different disease classes** with distinct pathological symptoms, they do not represent identical clone leakage. Even if a model learned one leaf, it would not receive an unfair shortcut on the opposite class in validation or test.
