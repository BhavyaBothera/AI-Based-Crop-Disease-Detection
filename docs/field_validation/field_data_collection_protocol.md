# PHYTOSIGNAL — Field Data Collection & Annotation Protocol

**Document Version:** 1.0.0  
**Status:** Approved Standard Operating Procedure (SOP)  
**Scope:** Real-world in-situ foliar image collection and annotation for Solanaceae crop disease validation  
**Target Crops:** Tomato (*Solanum lycopersicum*), Potato (*Solanum tuberosum*), Bell Pepper (*Capsicum annuum*)  
**Target Taxonomy:** All 15 established PHYTOSIGNAL disease and healthy classes  

---

## 1. Objective & Purpose

This protocol establishes a standardized, scientifically rigorous procedure for acquiring, annotating, and verifying authentic in-field foliar photographs. 

The primary goal is to evaluate the real-world generalization bounds of PHYTOSIGNAL's MobileNetV2 disease classification pipeline and two-tier leaf acceptance gate under authentic agronomic conditions—including variable sunlight, soil clutter, canopy background, disease severity, optical aberrations, and mobile hardware diversity.

> [!IMPORTANT]
> **Integrity Mandate**: This document defines the protocol for upcoming field collection. No field data collection has been completed yet. No claims of field performance shall be asserted until field images collected under this protocol have been validated and evaluated.

---

## 2. In-Field Sampling Dimensions & Environmental Factors

Field photographs must intentionally capture real-world variance across the following twelve operational axes:

### 2.1 Target Crops & Classes
All 15 target classes must be represented across active agricultural fields, research plots, and greenhouses:
- **Tomato (10 classes):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy.
- **Potato (3 classes):** Early Blight, Late Blight, Healthy.
- **Bell Pepper (2 classes):** Bacterial Spot, Healthy.

### 2.2 Disease Severity Stratification
Where disease symptoms are present, images should be stratified across three developmental stages:
1. **Early / Mild (1–10% foliar area affected):** Incipient micro-lesions, pinprick chlorotic specks, early water-soaking. (Critical for early intervention in commercial IPM).
2. **Moderate (10–35% foliar area affected):** Coalescing lesions, distinct concentric rings, noticeable chlorosis, localized leaf curling.
3. **Severe (>35% foliar area affected):** Extensive necrosis, blighting, significant defoliation, advanced systemic stunting or curling.

### 2.3 Natural Illumination & Ambient Lighting
Field scouts must capture images across distinct natural daylight profiles:
- **Direct Midday Sunlight:** High illuminance (>50,000 lux), harsh shadows, specular reflections on waxy cuticles.
- **Diffuse Overcast:** Soft, uniform sky illumination without distinct shadow borders (ideal baseline).
- **Early Morning / Golden Hour:** Low solar angle, warm color temperature, long cast shadows.
- **Dappled Canopy Shadow:** High dynamic range scenes where leaves are partially shaded by upper canopy leaves.
- **Backlighting:** Transmitted sunlight revealing vein chlorosis and lesion translucency.

### 2.4 Realistic Background Clutter
Unlike laboratory datasets (e.g. PlantVillage black/grey sheets), field images must feature genuine agricultural surfaces:
- **Bare Soil:** Dry loam, damp earth, rocky or sandy substrate.
- **Agricultural Mulch:** Black polyethylene film, silver reflective film, straw or organic compost.
- **Weed & Non-Target Understory:** Mixed grasses, broadleaf weeds, fallen detritus.
- **Dense Plant Canopy:** Overlapping healthy and diseased leaves from the same plant or adjacent plants.
- **Greenhouse / Trellis Hardware:** Twine, tomato cages, bamboo stakes, drip irrigation tubing.
- **Human Hand Framing:** Clean grower or scout hand gently supporting the leaf (common scouting posture).

### 2.5 Camera Hardware Diversity
To ensure model robustness across varying Camera Sensor ISPs and optical qualities:
- **Budget / Entry Smartphones:** Small sensor format, aggressive noise reduction, lower dynamic range (e.g., $100–$200 Android handsets).
- **Mid-Tier Smartphones:** Standard smartphone cameras representing typical agricultural extension devices.
- **Flagship Smartphones:** Multi-camera systems with aggressive HDR, computational bokeh, and color saturation (e.g., iPhone, Galaxy, Pixel).
- **DSLR / Mirrorless Reference:** Calibrated optical ground truth with neutral picture profile and minimal computational sharpening.

### 2.6 Working Distance, Framing & Orientation
- **Macro Close-Up:** High-magnification capture of an individual lesion or pustule group.
- **Standard Framed Leaf (Recommended):** Single target leaf occupying 60% to 85% of the image frame.
- **Nadir (Top-Down):** Normal perpendicular angle to the leaf surface.
- **Oblique / Angled (30°–60°):** Perspective angle reflecting handheld scouting without flattening the leaf.
- **Adaxial vs. Abaxial Surfaces:** Both upper (adaxial) and lower (abaxial) foliar surfaces must be sampled (vital for Leaf Mold and Spider Mite detection).

### 2.7 Natural Occlusion & Surface Artifacts
- Natural partial occlusion by petioles, stems, or overlapping leaves (≤25% of the target leaf occluded).
- Surface moisture: Morning dew droplets, rain mist, sprinkler irrigation residue.
- Agronomic residues: Insoluble copper fungicide deposits, sulfur powder, dust, dried soil splatter.

### 2.8 Diagnostic Look-Alikes (Differential Pairs)
Scouts must actively prioritize capturing difficult diagnostic pairs to stress-test model discrimination:
- `Potato___Early_blight` vs. `Tomato___Early_blight` vs. `Tomato___Target_Spot`.
- `Tomato___Late_blight` vs. `Tomato___Bacterial_spot` (early water-soaked stage).
- `Tomato___Leaf_Mold` vs. `Tomato___Spider_mites Two-spotted_spider_mite`.
- Physiological disorders (nutrient deficiencies, herbicide drift, heat stress) for leaf gate stress-testing.

### 2.9 Image Quality & Practical Field Flaws
Include controlled proportions of realistic field photography imperfections:
- Minor motion blur from wind-blown canopy foliage (≤10% of dataset).
- Slight defocus / shallow depth-of-field (where leaf tip is sharp and petiole is soft).
- Variable native resolutions (from 1 MP compressed phone uploads up to 20 MP raw captures; all prevalidated to ≥64×64 px).

---

## 3. Sampling Matrix & Target Cohort Sizes

### 3.1 Two-Stage Evaluation Plan

To balance statistical significance with logistical feasibility, field evaluation is structured into two sequential stages:

```
[ Phase 1: Initial Field Pilot ] ─────────► [ Phase 2: Fully Powered Evaluation ]
  Target: ~450–600 images                     Target: ~2,250–3,000 images
  Goal: Detect catastrophic domain shift      Goal: Formal statistical benchmarking
  Focus: Top-5 critical classes               Focus: All 15 classes (≥150 per class)
```

#### Stage A: Initial Field Pilot
- **Objective:** Detect immediate failure modes, evaluate leaf gate rejection behavior on field clutter, and calibrate preprocessing pipelines.
- **Minimum Target:** **450–600 images** (30–40 images per class across 15 classes).
- **Minimum Requirement for Reporting:** Minimum 20 distinct physical plants per class across at least 2 independent geographic farms or collection dates.

#### Stage B: Fully Powered Field Evaluation
- **Objective:** Compute statistically reliable per-class Macro-F1 scores with tight 95% confidence intervals (±3% margin of error).
- **Target Size:** **2,250–3,000 images** (150–200 images per class).
- **Power Justification:** For an expected F1 of 0.85 with an acceptable margin of error $E = 0.05$ at $\alpha = 0.05$, a minimum sample size of $n \ge \frac{z^2 \cdot p(1-p)}{E^2} = \frac{1.96^2 \cdot 0.85 \cdot 0.15}{0.0025} \approx 196$ independent observations per class is required.

### 3.2 Proposed Pilot Sampling Matrix

| Crop | Class Name | Pilot Target | Severity Stratification | Background Variety |
| :--- | :--- | :---: | :--- | :--- |
| **Bell Pepper** | `Pepper,_bell___Bacterial_spot` | 35 | Mild (10), Mod (15), Sev (10) | Soil, Canopy, Black plastic |
| **Bell Pepper** | `Pepper,_bell___healthy` | 35 | Asymptomatic (35) | Soil, Canopy, Greenhouse |
| **Potato** | `Potato___Early_blight` | 35 | Mild (10), Mod (15), Sev (10) | Soil, Mulch, Canopy |
| **Potato** | `Potato___Late_blight` | 35 | Mild (10), Mod (15), Sev (10) | Moist soil, Wet foliage |
| **Potato** | `Potato___healthy` | 35 | Asymptomatic (35) | Field soil, Canopy |
| **Tomato** | `Tomato___Bacterial_spot` | 35 | Mild (10), Mod (15), Sev (10) | Soil, Trellis, Hand-held |
| **Tomato** | `Tomato___Early_blight` | 35 | Mild (10), Mod (15), Sev (10) | Soil, Lower canopy |
| **Tomato** | `Tomato___Late_blight` | 35 | Mild (10), Mod (15), Sev (10) | Field canopy, Stem blighting |
| **Tomato** | `Tomato___Leaf_Mold` | 35 | Mild (10), Mod (15), Sev (10) | High-humidity greenhouse, Abaxial |
| **Tomato** | `Tomato___Septoria_leaf_spot` | 35 | Mild (10), Mod (15), Sev (10) | Lower foliage, Splattered soil |
| **Tomato** | `Tomato___Spider_mites` | 35 | Mild (10), Mod (15), Sev (10) | Dry canopy, Abaxial webbing |
| **Tomato** | `Tomato___Target_Spot` | 35 | Mild (10), Mod (15), Sev (10) | Warm humid canopy |
| **Tomato** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 35 | Early curl (10), Advanced (25) | Upper shoot canopy |
| **Tomato** | `Tomato___Tomato_mosaic_virus` | 35 | Mottling (15), Distortion (20) | Young leaves, Stunted canopy |
| **Tomato** | `Tomato___healthy` | 35 | Asymptomatic (35) | Field, Greenhouse, Trellis |
| **Total** | **All 15 Target Classes** | **525** | — | — |

---

## 4. Metadata Schema & Privacy Protection

### 4.1 Schema Definition
Every field image must be cataloged in a standardized UTF-8 CSV/JSON manifest containing the following 16 fields:

| Field | Type | Description | Permitted Values / Format |
| :--- | :--- | :--- | :--- |
| `image_id` | String | Unique specimen identifier | `FIELD_000001` format |
| `file_path` | String | Relative POSIX path within dataset | `data/field_dataset/tomato/...` |
| `crop` | String | Botanical crop name | `Tomato`, `Potato`, `Bell Pepper` |
| `ground_truth_class` | String | Exact matching class identifier | One of the 15 `TARGET_CLASS_MAPPING` keys |
| `collection_source` | String | Institutional or partner source | e.g. `Agronomy_Extension_Plot_A` |
| `capture_device` | String | Camera or mobile device model | e.g. `Samsung_Galaxy_A14`, `iPhone_13` |
| `capture_date` | String | Date of photograph capture | `YYYY-MM-DD` |
| `collection_location` | String | Privacy-preserving regional code | Region or District level (e.g. `US_CA_CentralValley`) |
| `lighting_condition` | String | Ambient illumination category | `direct_sun`, `diffuse_overcast`, `dappled_shade`, `golden_hour`, `backlit` |
| `background_type` | String | Primary backdrop behind leaf | `bare_soil`, `canopy`, `mulch_plastic`, `weed_understory`, `greenhouse`, `hand` |
| `disease_severity` | String | Pathological severity category | `mild`, `moderate`, `severe`, `asymptomatic` |
| `annotator_status` | String | Qualifications of first annotator | `field_agronomist`, `pathology_researcher`, `trained_scout` |
| `verification_status` | String | Diagnostic verification level | `visually_suspected`, `expert_verified`, `laboratory_confirmed` |
| `split` | String | Evaluation partition assignment | `field_dev` (development) or `field_test` (locked test) |
| `duplicate_group_id` | String | Plant/session grouping identifier | `PLANT_042_SESSION_01` (prevents split leakage) |
| `notes` | String | Optional pathological remarks | Contextual observations, e.g. look-alikes or spray residue |

### 4.2 Farmer & Location Privacy Rules
To protect farmers, landowners, and agricultural producers:
1. **Zero GPS Coordinates:** Precise latitude and longitude coordinates must **NEVER** be collected or committed to the repository.
2. **Coarse Regional Aggregation:** Geographic metadata is restricted to state, province, or agricultural climatic zone level (e.g., `IN_MH_NashikDistrict`, `US_FL_Hillsborough`).
3. **No Farmer PII:** Farmer names, business names, land parcel numbers, and identifiable landmarks must be excluded from all filenames, manifests, and image backgrounds.
4. **No Face or PII in Frame:** If a human hand or arm appears while holding a leaf, no identifying jewelry, clothing logos, or human faces may be visible. Any accidental faces are rejected by Level B gate.

---

## 5. Ground Truth & Annotation Rules

A deep learning classifier must never be validated against ground truth derived from its own predictions.

### 5.1 Verification Hierarchy
All annotations must declare one of three strict verification levels:
1. **`visually_suspected`:** Classified in the field based solely on gross visible symptoms by a trained agricultural scout. (Suitable for pilot exploration; excluded from primary formal test benchmark).
2. **`expert_verified`:** Confirmed by a qualified plant pathologist, certified crop advisor (CCA), or university extension specialist who reviewed physical specimens or high-resolution macro diagnostic imagery.
3. **`laboratory_confirmed`:** Verified via authoritative laboratory diagnostics:
   - Fungal / Oomycete sporulation examination under compound microscopy (e.g. *Phytophthora infestans*, *Alternaria solani*).
   - Bacterial streaming assay or PCR for *Xanthomonas euvesicatoria*.
   - ELISA or RT-PCR test strips for *Tomato Yellow Leaf Curl Virus* or *Tobacco/Tomato Mosaic Virus*.

### 5.2 Adjudication of Ambiguous Specimens
- If two qualified annotators disagree on a visual diagnosis (e.g. Early Blight vs. Target Spot), the sample must not be force-labeled.
- The sample must either be escalated to laboratory confirmation or assigned `verification_status = ambiguous_unresolved` and excluded from the primary supervised evaluation metric.

---

## 6. Minimum Evidence Requirements for Performance Claims

No claim that PHYTOSIGNAL is "field-ready", "field-tested", or "field-validated" may be published in portfolio documentation, technical reports, or marketing communications unless all of the following criteria are met:

1. **Dataset Integrity:** Evaluation was performed on an approved field dataset of at least **450 images** spanning all 15 classes.
2. **Independent Verification:** At least 80% of test images must be `expert_verified` or `laboratory_confirmed`.
3. **Leakage Protection:** No plant, session, or capture cluster appears in both development and test partitions.
4. **Reproducible Pipeline:** Evaluation is generated using the standalone runner in [`ml/evaluation/field_validation/`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/evaluation/field_validation/) with exact checkpoint verification and fixed seeds.
5. **Full Disclosure:** Both overall accuracy and per-class Macro-F1 are reported, alongside explicit error rates for difficult classes and input gate performance.
