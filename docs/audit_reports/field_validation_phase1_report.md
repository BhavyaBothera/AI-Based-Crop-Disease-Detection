# PHYTOSIGNAL 2.0 — Phase 1: Field Validation & Generalization Audit Report

**Product:** PHYTOSIGNAL (AI-Based Crop Disease Detection and Treatment Recommendation System)  
**Evaluation Phase:** Phase 1 — Field Validation Infrastructure & Generalization Audit  
**Author:** Senior Machine Learning Engineer, CV Researcher & ML Evaluation Specialist  
**Date:** November 2026  
**Status:** Verification Framework Complete; Empirical Field Collection Pending  

---

## Executive Summary

Phase 1 of **PHYTOSIGNAL 2.0** establishes a rigorous, scientifically credible evaluation framework to measure how well our existing 15-class Solanaceae crop disease classification pipeline generalizes from controlled studio photography (PlantVillage) to realistic in-field agricultural conditions across **Tomato**, **Potato**, and **Bell Pepper**.

Prior to this work, the project evaluated the classifier exclusively on a 3,413-image held-out test split of controlled studio specimens (achieving an overall accuracy of 96.89% and Macro-F1 of 0.9653). However, **no real-world field images existed or had been evaluated in this repository**. In accordance with our core scientific integrity principle:

> [!IMPORTANT]
> **Zero Field Validation Claim:**  
> The system is **NOT** claimed to be field-validated at this stage. Phase 1 provides the complete auditing engine, data collection protocol, manifest schema, leakage detectors, leaf gate evaluator, and standalone evaluation pipeline. Actual empirical generalization benchmarks will be published only after verified field images are collected under the approved protocol.

---

## Section A: Verified Existing Baseline Facts

An exhaustive audit of the existing repository was performed before introducing any new code. The following technical facts represent the ground truth of the production system:

### 1. Production Model Checkpoint & Architecture
- **Active Production Artifact:** `ml/models/experiment_weighted/best_model.keras`
- **File Integrity SHA-256:** `779b8431beb8eacc1cc31c1055e252d88900e00b26612fd1450ed3ef16eba534`
- **Architecture:** MobileNetV2 feature extractor initialized from ImageNet, topped with GlobalAveragePooling2D, Dropout(0.20), and a Dense(15, activation='softmax') classification head.
- **Embedded Rescaling:** The first layer of the Keras graph is `layers.Rescaling(scale=1.0/127.5, offset=-1.0)`.
- **CRITICAL Preprocessing Invariant:** Because the model internally rescales inputs from $[0, 255]$ to $[-1, 1]$, inputs fed to `model.predict()` must strictly be raw `float32` RGB arrays in the $[0.0, 255.0]$ range. Pre-normalizing to $[0, 1]$ or $[-1, 1]$ results in double-rescaling and catastrophic prediction failures.

### 2. Authoritative Class Mapping (15 Solanaceae Classes)
The class indices (0 to 14) are strictly contiguous and identical across training (`ml/scripts/train_model.py`), dataset preparation (`ml/scripts/prepare_dataset.py`), backend inference (`backend/app/services/inference.py`), and the frontend API contract:
- **Bell Pepper (2 classes):** `Pepper,_bell___Bacterial_spot` (ID: 0), `Pepper,_bell___healthy` (ID: 1)
- **Potato (3 classes):** `Potato___Early_blight` (ID: 2), `Potato___Late_blight` (ID: 3), `Potato___healthy` (ID: 4)
- **Tomato (10 classes):** `Tomato___Bacterial_spot` (ID: 5), `Tomato___Early_blight` (ID: 6), `Tomato___Late_blight` (ID: 7), `Tomato___Leaf_Mold` (ID: 8), `Tomato___Septoria_leaf_spot` (ID: 9), `Tomato___Spider_mites Two-spotted_spider_mite` (ID: 10), `Tomato___Target_Spot` (ID: 11), `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (ID: 12), `Tomato___Tomato_mosaic_virus` (ID: 13), `Tomato___healthy` (ID: 14)

### 3. Input Screening & Leaf Acceptance Gate
- **Two-Tier Screening:**
  - **Level A (Operational & Security):** Magic-byte format verification (JPEG, PNG, WebP), Pillow decompression-bomb protection (20 MP ceiling), EXIF orientation transposition, 64×64 px minimum floor, and luminance population standard deviation check ($\sigma > 3.0$) to reject blank/uniform canvases.
  - **Level B (Content Suitability):** MobileNetV2 linear head (`ml/models/input_gate/leaf_gate.keras`) outputting 3 classes: `non_leaf`, `other_plant`, `supported_leaf`. Threshold: confidence $\ge 0.60$ for acceptance. Deterministic plant pigmentation screen acts as fallback if the model file is disabled.

### 4. Controlled Benchmark Reference Baseline
Evaluated on the 3,413 controlled test images from `ml/data/dataset_manifest.csv`:
- **Top-1 Accuracy:** 96.8942% (3,307 / 3,413 correct)
- **Macro-F1 Score:** 0.965288
- **Top-3 Accuracy:** 99.7070% (3,403 / 3,413)
- **Macro-Precision:** 0.966459 | **Macro-Recall:** 0.964953
- **Weakest Controlled Classes:**
  - `Tomato___Early_blight`: Recall 86.00%, Precision 84.31%, F1 0.8515 (confused with `Tomato___Target_Spot`)
  - `Tomato___Target_Spot`: Recall 92.38%, Precision 93.27%, F1 0.9282

---

## Section B: New Implementation

To prepare PHYTOSIGNAL for field deployment, we designed and built a modular evaluation subsystem in `ml/evaluation/field_validation/`:

```
ml/evaluation/field_validation/
├── __init__.py               # Clean package export interface
├── manifest_schema.py        # 16-attribute metadata schema, validation & privacy rules
├── data_auditor.py           # Integrity, corruptions, duplicate hashes & split leakage detector
├── evaluator.py              # Exact-inference model evaluator with error ledgers & uncalibrated scores
├── gate_evaluator.py         # Leaf acceptance gate evaluator (FRR, FAR & challenge robustness)
├── metrics.py                # Wilson score & bootstrap CIs, per-crop/class metrics & subgroup slicing
└── runner.py                 # Non-destructive CLI orchestrator & report generator
```

### Module Descriptions & Capabilities

1. **Manifest Schema & Privacy Protection (`manifest_schema.py`)**
   - Implements `FieldSampleRecord` supporting 16 standardized metadata fields: `image_id`, `file_path`, `crop`, `ground_truth_class`, `collection_source`, `capture_device`, `capture_date`, `collection_location`, `lighting_condition`, `background_type`, `disease_severity`, `annotator_status`, `verification_status`, `split`, `duplicate_group_id`, `notes`.
   - **Privacy Rule:** Strictly rejects precise GPS coordinates (e.g. `37.7749, -122.4194`) and farm-level PII, requiring regional/district codes.
   - **Crop-Class Alignment:** Validates that declared crop strictly matches the disease target (e.g. flags any record claiming `Potato` crop with `Tomato___Early_blight`).
   - **Split Discipline:** Enforces disjoint partition labels (`field_dev` vs `field_test`).

2. **Data Quality & Leakage Auditor (`data_auditor.py`)**
   - **Integrity Validation:** Detects truncated bytes, malformed containers, and dimensions outside the $[64\text{ px}, 20\text{ MP}]$ operational window.
   - **Exact Duplicates:** Computes SHA-256 byte hashes and flags identical images.
   - **Perceptual Near-Duplicates:** Implements 64-bit difference hashing (`dHash`) with pairwise Hamming distance evaluation (threshold $\le 6$).
   - **Group-Aware Split Leakage:** Flags any instance where photographs from the same plant, capture session, or scene (`duplicate_group_id`) appear in both `field_dev` and `field_test`.
   - **Controlled Baseline Leakage:** Cross-checks field image hashes against all 18,160 controlled PlantVillage images to guarantee total independence.

3. **Standalone Model Evaluator (`evaluator.py`)**
   - Applies the exact production inference pipeline: `ImageOps.exif_transpose`, RGB conversion, bilinear resize to $224 \times 224$, and raw float32 $[0.0, 255.0]$ arrays.
   - Records top-1 and top-3 candidates, error ledgers, and raw classifier scores explicitly marked as uncalibrated raw softmax scores.
   - Generates immutable timestamped JSON outputs (`field_disease_eval_YYYYMMDD_HHMMSS.json`) and Markdown summaries. Never overwrites prior runs.

4. **Leaf Gate Evaluator (`gate_evaluator.py`)**
   - Evaluates Level A + Level B input screening against field reality.
   - Computes False Rejection Rate (FRR) on legitimate supported leaves and False Acceptance Rate (FAR) on other plant species and non-leaf objects.
   - Assesses resilience under challenge conditions: foliage clutter, partial leaves, multi-leaf scenes, motion blur, and background-only frames.

5. **Statistical Metrics Engine (`metrics.py`)**
   - Computes Wilson score 95% confidence intervals for Top-1, Top-3, and subgroup accuracies.
   - Implements 500-sample percentile bootstrap confidence intervals for Macro-F1.
   - Computes full $15 \times 15$ confusion matrices, per-crop metrics, and per-class precision/recall/F1.
   - Evaluates environmental subgroups (lighting, background, severity, device) with automatic `[LOW SAMPLE < 10]` disclosure flags.
   - Computes plant-level majority-vote accuracy for grouped observations.

6. **CLI Runner (`runner.py`)**
   - Provides a single command-line interface supporting `--manifest`, `--dataset-root`, `--model-path`, `--gate-path`, `--output-dir`, `--split`, `--audit-only`, `--eval-gate`, and `--baseline-comparison`.

---

## Section C: Software Test Results

To verify software correctness and mathematical reproducibility without external dependencies or data contamination, an automated test suite was constructed: `ml/tests/test_field_validation.py`.

### Test Execution Command
```powershell
.venv/Scripts/python.exe -m unittest ml/tests/test_field_validation.py
```

### Test Outcomes
```
................
----------------------------------------------------------------------
Ran 16 tests in 0.844s

OK
```

All 16 test suites passed:
1. `TestClassMappingConsistency.test_class_mapping_matches_backend`: Verified 15-class match with backend.
2. `TestClassMappingConsistency.test_class_ids_contiguous_0_to_14`: Verified IDs $0 \dots 14$.
3. `TestClassMappingConsistency.test_crop_class_distribution`: Verified 10 Tomato, 3 Potato, 2 Pepper classes.
4. `TestManifestSchemaAndValidation.test_valid_manifest_record`: Validated complete 16-field record parsing.
5. `TestManifestSchemaAndValidation.test_reject_crop_class_mismatch`: Successfully caught crop-disease mismatch.
6. `TestManifestSchemaAndValidation.test_reject_gps_coordinates_for_privacy`: Rejected GPS coordinates for privacy.
7. `TestManifestSchemaAndValidation.test_reject_invalid_split`: Rejected illegal split names.
8. `TestDataQualityAndLeakageAuditor.test_corrupt_and_missing_images_detection`: Isolated corrupt and missing files into error ledgers.
9. `TestDataQualityAndLeakageAuditor.test_exact_and_perceptual_duplicate_detection`: Detected SHA-256 and dHash near-duplicates.
10. `TestDataQualityAndLeakageAuditor.test_cross_partition_group_leakage`: Flagged shared plant IDs across `field_dev` and `field_test`.
11. `TestPreprocessingParity.test_preprocessing_output_specs`: Verified $(1, 224, 224, 3)$ float32 in $[0.0, 255.0]$.
12. `TestPreprocessingParity.test_uniform_canvas_rejection`: Rejected blank canvas ($\sigma \le 3.0$).
13. `TestPreprocessingParity.test_undersized_image_rejection`: Rejected sub-$64\text{ px}$ image.
14. `TestMetricCalculations.test_wilson_score_interval`: Validated binomial Wilson 95% CIs.
15. `TestMetricCalculations.test_field_metrics_calculation`: Validated Top-1, Top-3, Macro-F1, per-crop, and subgroup warnings.
16. `TestLeafGateEvaluator.test_gate_metrics_aggregation`: Validated FRR and FAR calculations.

### Regression Verification
Existing test suites were executed to verify zero regression across the system:
- **ML Training Pipeline Suite:** `ml/tests/test_train_pipeline.py` (22 tests, all PASSED).
- **Backend API & Service Suite:** `backend/tests/` (98 tests, all PASSED).

---

## Section D: Actual Empirical Results

### Current Repository Status: Empirical Field Evaluation Pending
A physical scan of the repository confirms that **zero verified field photographs of Solanaceae crops currently exist on disk**.

The repository contains:
- Controlled PlantVillage images (`ml/data/raw/`): Studio photography with uniform gray/black backing boards.
- Non-leaf OOD images (`ml/data/input_gate_corpus/`): Objects, hands, papers, and non-crop foliage used to train the gate.

Therefore, **no empirical field benchmark numbers are claimed**.

### End-to-End System Demonstration (Synthetic Fixture)
To verify that the evaluation pipeline generates valid reports end-to-end, a 15-sample synthetic fixture was processed through `runner.py`:
- **Audit Tool:** Verified manifest parsing, image dimensions, and absence of controlled leakage.
- **Disease Evaluator:** Successfully executed inference using `ml/models/experiment_weighted/best_model.keras` (SHA-256: `779b8431beb8eacc...`), recorded top-1/top-3 predictions and raw softmax scores, and saved `field_disease_eval_*.json` and Markdown summary.
- **Leaf Gate Evaluator:** Verified `ml/models/input_gate/leaf_gate.keras` predictions and saved `field_gate_eval_*.json`.
- **Generalization Comparison:** Produced comparison report against controlled baseline (`controlled_vs_field_*.md`).

---

## Section E: Pending Data Collection & Sampling Plan

A complete data collection protocol has been formalized in [`docs/field_validation/field_data_collection_protocol.md`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/field_validation/field_data_collection_protocol.md).

### 1. Two-Stage Field Sampling Strategy

| Parameter | Stage 1: Pilot Feasibility | Stage 2: Full Statistical Audit |
| :--- | :--- | :--- |
| **Primary Objective** | Validate protocol, test pipeline in real fields, audit annotator agreement. | Compute statistically powered per-class benchmarks (95% CI margin $\le \pm 5\%$). |
| **Target Sample Size** | **525 images** (35 per class $\times$ 15 classes). | **2,250 – 3,000 images** (150 – 200 per class $\times$ 15 classes). |
| **Crops Covered** | Tomato (350), Potato (105), Bell Pepper (70). | Tomato (1,500–2,000), Potato (450–600), Bell Pepper (300–400). |
| **Collection Sites** | 3 distinct agricultural zones / farm blocks. | $\ge 10$ independent commercial farms and research stations. |
| **Partition Allocation** | 100% assigned to `field_dev` for threshold calibration and failure discovery. | 30% `field_dev`, 70% locked `field_test`. |

### 2. Environmental Axes Covered
The protocol systematically samples across 12 operational dimensions:
1. All 15 existing classes (12 diseased, 3 healthy).
2. Four severity tiers (asymptomatic, mild $\le 10\%$, moderate $11\text{--}35\%$, severe $> 35\%$).
3. Five lighting regimes (direct sun, diffuse overcast, dappled shade, golden hour, backlighting).
4. Four background types (bare soil/mulch, dense canopy, greenhouse hardware, complex understory).
5. Sensor diversity ($\ge 5$ smartphone models across flagship, mid-tier, and entry-level).
6. Framing variations (macro leaflet framing, multi-leaf cluster, angled orientation).
7. Realistic optical challenges (partial leaf clipping, motion blur, wind shake).
8. Pathological look-alikes (e.g. early blight vs target spot, septoria vs bacterial spot).

### 3. Diagnostic Verification Hierarchy
Annotations must follow strict evidential tiers:
1. **Tier 1 (Laboratory Confirmed):** PCR, ELISA, or microscopic confirmation (gold standard).
2. **Tier 2 (Expert Verified):** Plant pathologist or certified agronomist inspection.
3. **Tier 3 (Trained Scout):** Standard field scouts; requires dual-blind consensus.
4. **Disallowed:** Farmer self-reports without photos or model-assigned pseudo-labels.

---

## Section F: Risks and Limitations

1. **Closed-World Diagnostic Risk:**
   The 15-class classifier assumes the input belongs to one of its 15 trained categories. In the field, non-Solanaceae weeds, abiotic nutrient deficiencies (nitrogen, magnesium), herbicide drift, and insect damage will be presented. Without the Level B input gate, the model will output a high-confidence false diagnosis.
2. **Pathology Distribution Shift:**
   PlantVillage images feature isolated leaves picked from plants and photographed on neutral boards. Real field leaves are attached to living stems, often showing compound symptoms (e.g. insect feeding plus fungal blight), variable senescence, and soil splash.
3. **Severe Small-Sample Uncertainty:**
   Until $\ge 150$ verified field images per class are evaluated, individual class F1-scores will have wide confidence intervals ($\pm 15\text{--}25\%$). Low support classes must never be disguised behind overall accuracy.
4. **Grouped Observation Bias:**
   Evaluating 5 photographs of different leaflets on the same plant does not represent 5 independent statistical observations. If split across partitions, this causes data leakage; if treated as independent, it inflates confidence.

---

## Section G: Recommended Phase 2 Roadmap

1. **Execute Field Pilot Collection (Target: 525 Images):**
   - Deploy mobile protocol to 3 partner farms / agricultural stations in Solanaceae growing regions.
   - Collect and catalog 35 images per class into `field_dev`.
2. **Calibrate Rejection Thresholds on `field_dev`:**
   - Measure actual distribution of raw softmax entropy and maximum score on field inputs.
   - Tune leaf acceptance gate operating point to optimize the trade-off between False Rejection Rate (FRR) on real leaves and False Acceptance Rate (FAR) on field weeds.
3. **Lock `field_test` Partition:**
   - Assemble Stage 2 full dataset (2,250+ images).
   - Freeze the test manifest and run the final benchmark audit without any further hyperparameter tuning.
4. **Publish Empirical Generalization Report:**
   - Run `ml.evaluation.field_validation.runner` on the locked `field_test` partition.
   - Document real-world degradation delta, specific confusion pairs, and operational recommendations for farm deployment.
