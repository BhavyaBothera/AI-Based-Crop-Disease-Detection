# AI-Based Crop Disease Detection and Treatment Recommendation System

### Project Summary Report

| | |
|:--|:--|
| **Project title** | AI-Based Crop Disease Detection and Treatment Recommendation System |
| **Domain** | Applied machine learning · agricultural informatics · web service engineering |
| **Student names** | Bhavya, Atharv, Dheeraj, Ganpat |
| **Semester** | V |
| **Institution** | NIIT University |
| **Academic year** |2026-27|

---

## Abstract

Smallholder and horticultural growers lose a significant share of their Solanaceae crop
(tomato, potato, bell pepper) to foliar disease, and the usual first line of defence is visual
inspection of a leaf. That inspection is slow, requires experience, and is easily confused
between look-alike conditions. This project builds an end-to-end system that takes a single
photograph of a leaf and returns two things: a ranked diagnosis over **15 Solanaceae
conditions**, and the validated agronomic guidance for whichever condition is identified —
symptoms, favourable conditions, cultural prevention, integrated pest management (IPM)
principles, escalation triggers, and cited extension sources.

The classifier is a fine-tuned **MobileNetV2** convolutional network trained with two-stage
transfer learning on a deduplicated, leak-audited subset of the **PlantVillage** dataset
(22,773 unique images). The held-out test set was evaluated exactly once, after model
selection was frozen on validation data, and the final model achieves **macro-F1 0.9653** and
**accuracy 96.89 %** (3,307 of 3,413 images). Around that model sits a **FastAPI** service
with a deterministic input-safety ladder, bounded concurrency, and a startup-validated
agronomic knowledge base, plus a no-build HTML/CSS/JavaScript front end. The system is
verified by **99 automated tests** across the ML pipeline, the backend, and the front-end API
client.

The distinguishing work of this project is not the classifier alone but the rigour around it:
leakage was measured rather than assumed, the test set was touched once, an out-of-distribution
rejection gate was **evaluated and then deliberately rejected on evidence**, and every safety
threshold in the shipped code carries an empirical justification over the full validation set.

---

## 1. Problem Statement

Crop disease diagnosis from a photograph is a well-studied supervised classification problem,
but a deployable system must answer four questions that a notebook does not:

1. **Is the reported accuracy real?** PlantVillage images are near-duplicates of one another.
   If copies of the same leaf straddle the train/test boundary, test accuracy measures
   memorisation, not generalisation.
2. **What happens to a bad upload?** Real users submit a blank screen, a 20-megapixel photo, a
   GIF, a corrupt file, or a screenshot. A model that silently answers anyway produces a
   confident, useless diagnosis.
3. **What does the label mean to the grower?** A class name such as
   `Tomato___Tomato_Yellow_Leaf_Curl_Virus` is not advice. It has to be turned into symptoms,
   prevention, escalation, and sources.
4. **How should the output be worded?** A 15-class softmax is a relative distribution over a
   closed world of 15 classes. Presenting it as diagnostic certainty would be a false claim
   with real agricultural consequences.

This project treats all four as first-class engineering requirements.

---

## 2. Objectives

1. Train a 15-class Solanaceae foliar disease classifier with strong generalisation, using
   transfer learning on a modest CPU-only budget.
2. Certify the dataset before training: audit quality, remove duplicates, and construct a
   **leak-free stratified split** by content hash and near-duplicate group.
3. Select the model on validation only, and evaluate the test set **once**, with an integrity
   audit proving that discipline was followed.
4. Serve the model behind a REST API with a deterministic input-validation ladder, bounded
   concurrency, and a specific HTTP status code for every failure mode.
5. Build a curated agronomic knowledge base whose class IDs are **1:1 with model output**,
   validated at startup, and serve it through the same API.
6. Provide a working browser front end that renders diagnoses and guidance safely.
7. Verify the whole system with automated tests across all three layers.
8. State limitations honestly, including one intervention that was measured and rejected.

---

## 3. Scope and Delimitations

**In scope:** 15 Solanaceae classes across three crops; single-leaf, close-up photography;
library-style diagnosis with educational guidance; local deployment.

**Explicitly out of scope:** field-condition robustness; recommendation of specific pesticide
products or doses; yield or economic modelling; user accounts; multi-crop generalisation
beyond the three trained crops; and cloud deployment hardening.

---

## 4. Background

**Transfer learning.** MobileNetV2 (Sandler et al., 2018) is a depthwise-separable
convolutional architecture designed for efficient inference. Its ImageNet features transfer
well to leaf imagery, and its small footprint keeps CPU inference practical — the served
checkpoint is ~28 MB on disk.

**PlantVillage.** The dataset (Hughes & Salathé, 2015) contains laboratory-captured leaf
images on uniform backgrounds across 38 disease and health classes. It is the standard
benchmark for this task and also the source of its best-known weakness: controlled capture
conditions, and near-duplicate images that can leak across splits.

**Agronomic sourcing.** Guidance in this project is drawn from public agricultural extension
material — principally the Tamil Nadu Agricultural University (TNAU) Agritech Portal and the
University of California Statewide IPM Program — rather than commercial product literature.

---

## 5. System Architecture

```text
        Browser (frontend/)
        ┌──────────────────────────────────────────────┐
        │ index.html + styles.css                      │
        │ app.js  — upload, preview, render results    │
        │ api.js  — fetch client, timeouts, error maps │
        └───────────────────┬──────────────────────────┘
                            │  HTTP/JSON  (CORS allow-list)
                            ▼
        FastAPI service (backend/app/)
        ┌──────────────────────────────────────────────┐
        │ main.py      lifespan: load model + KB once  │
        │ api/         health · predict · knowledge    │
        │ schemas/     Pydantic contracts              │
        │ services/    inference · knowledge           │
        └───────┬───────────────────────┬──────────────┘
                │                       │
                ▼                       ▼
   ml/models/experiment_weighted/   backend/app/data/
   best_model.keras  (28 MB)        disease_knowledge_base.json (72 KB)
   MobileNetV2, 15-class softmax    15 validated agronomic records
```

**Request lifecycle for one diagnosis:**

1. Browser validates type and size locally, then POSTs `multipart/form-data` to `/api/predict`.
2. The endpoint checks upload presence and model readiness, then enforces the declared MIME type.
3. The body is streamed in 64 KB chunks with a hard 10 MB ceiling (HTTP 413 when exceeded).
4. The declared MIME type is checked against the file's magic bytes (415 on mismatch).
5. Decoding is admitted through a queue (10 waiters) and a semaphore (2 concurrent decodes), then
   runs in a worker thread so the event loop never blocks.
6. Preprocessing validates format, pixel ceiling, minimum dimension, converts to RGB, resizes to
   224×224 bilinear, and rejects blank canvases by luminance σ.
7. Inference runs under a separate semaphore of 1 (serialised), then the response is built from
   the Pydantic contract with the top-3 candidates and all 15 scores.
8. The front end renders the primary diagnosis and, for whichever class the user selects, fetches
   `/api/knowledge/{class_id}` and renders the agronomic profile.

**Startup behaviour.** The model and knowledge base load once in the FastAPI lifespan handler.
If either fails, the process still starts, `/health` stays `200`, and `/ready` returns `503`
naming which component is unavailable — so an orchestrator can distinguish "process alive" from
"cannot serve".

---

## 6. Dataset Engineering

### 6.1 Audit before use

All **55,448** images were inspected with Pillow before any split existed:

| Check | Result |
|:--|:--|
| Files audited / readable images | 55,448 / 55,448 |
| Corrupted or unreadable | 0 |
| Non-image files | 0 |
| Classes detected | 39 |
| Majority-class baseline (dataset-wide) | 9.93 % |
| Class imbalance ratio (max : min) | 36.23 : 1 |
| Resolution range | 256×192 to 256×256 px (mean 256.0×254.7) |
| Square (1:1) images | 54,305 / 55,448 |
| Formats | JPEG 55,447 · PNG 1 |
| Exact byte duplicates (SHA-256) | 22 groups |
| Perceptual near-duplicates (dHash ≤ 2) | 100 candidate pairs |

The majority-class baseline is the number any model must beat to be meaningful: 9.93 %.

### 6.2 Reconciliation to the target problem

| Step | Count |
|:--|--:|
| Total images in archive | 55,448 |
| Non-target crops excluded (preserved, marked `excluded`) | −32,661 |
| Target Solanaceae images (Tomato 18,160 · Potato 2,152 · Bell Pepper 2,475) | 22,787 |
| Exact duplicate copies excluded | −14 |
| **Unique images used for modelling** | **22,773** |

### 6.3 Leak-free stratified split

Split by **content hash group**, not by file, so that duplicate and near-duplicate clusters can
never straddle a boundary. Seed 42, target 70/15/15:

| Split | Images | Realised | Purpose |
|:--|--:|--:|:--|
| Train | 15,942 | 70.0 % | optimisation + dynamic augmentation |
| Validation | 3,418 | 15.01 % | checkpoint selection |
| Test | 3,413 | 14.99 % | one-shot benchmark, sealed until the end |

Verification results: **0 SHA-256 overlaps** across all three intersections, **0 duplicate
clusters split across partitions**. Augmentation is applied dynamically in memory on the
training split only — never written to disk.

### 6.4 Residual imperfection, declared

Five cross-class near-duplicate pairs (dHash = 2) remained after grouping and were reviewed
individually against a contact sheet. Complete duplicate-free separation is **not** claimed;
this caveat travels with the test results rather than being buried.

---

## 7. Model Design and Training

### 7.1 Architecture

```text
Input (224, 224, 3), raw [0, 255]
  └─ Rescaling(1/127.5, offset = -1)      ← embedded in the checkpoint
  └─ MobileNetV2 backbone, ImageNet weights
  └─ GlobalAveragePooling2D
  └─ BatchNormalization
  └─ Dense(256, ReLU, L2 1e-4)
  └─ Dropout(0.35)
  └─ Dense(15, softmax)
```

Because normalisation lives inside the model, the API deliberately feeds **raw 0–255 pixels**.
This removes an entire class of train/serve skew: there is no separate preprocessing constant to
get out of sync.

### 7.2 Two-stage transfer learning

| | Stage 1 — feature extraction | Stage 2 — fine-tuning |
|:--|:--|:--|
| Backbone | frozen | layers 100–153 unfrozen |
| BatchNorm layers | frozen | **remain frozen** (`training=False`) |
| Optimizer / LR | Adam, 1e-3 | Adam, 1e-5 |
| Epochs configured | 15 | 15 |
| Batch size | 32 | 32 |
| Seed | 42 | 42 |

Batch normalisation statistics stay frozen in both stages so the pretrained feature
distribution is not disturbed by the small fine-tuning learning budget.

Callbacks: `EarlyStopping(patience=5)` and `ReduceLROnPlateau(factor=0.2, patience=2)`, both
monitoring `val_macro_f1`. The Macro-F1 callback runs first each epoch so the schedulers observe
a populated metric.

Training completed **28 epochs** (15 + 13; stage 2 stopped early). The selected checkpoint is
stage 2, epoch 8 — global epoch 23 — at validation macro-F1 **0.9685867**.

### 7.3 Class imbalance

Balanced class weights were computed from training counts. The two extremes show the range:

| Class | Train images | Weight |
|:--|--:|--:|
| 4 · `Potato___healthy` | 106 | 10.026 |
| 13 · `Tomato___Tomato_mosaic_virus` | 261 | 4.072 |
| 12 · `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 3,750 | 0.283 |

### 7.4 Selection rule

The checkpoint with the highest **full-validation macro-F1** wins — computed over the entire
validation set after each epoch, never averaged over mini-batches. Differences within `1e-4` are
treated as ties and broken by lower `val_loss`. Macro-F1 is the selection metric because with a
36:1 imbalance, accuracy rewards ignoring the small classes; macro-F1 weights all 15 equally.

An unweighted control model was trained for comparison and evaluated on validation only:

| Metric (validation, n = 3,418) | Unweighted | Balanced weighted | Δ |
|:--|--:|--:|--:|
| **Macro-F1** | 0.963411 | **0.968587** | **+0.005175** |
| Accuracy | 0.971913 | 0.975132 | +0.003218 |
| Macro-recall | 0.963383 | 0.974257 | +0.010874 |
| Macro-precision | 0.963716 | 0.963556 | −0.000160 |
| Weighted-F1 | 0.971825 | 0.975168 | +0.003343 |
| Validation loss | 0.170589 | 0.174981 | +0.004392 |

The weighted model was selected on validation macro-F1 (+0.0052). Its most visible gain is on
the rare classes: `Potato___healthy` recall rose from 0.9130 to 1.0000. The test set was not
touched during this comparison.

---

## 8. Results

### 8.1 Held-out test set — evaluated once

n = **3,413** images, evaluation time 19.04 s, selected model
`ml/models/experiment_weighted/best_model.keras`.

| Metric | Test result | Validation | Generalisation gap |
|:--|--:|--:|--:|
| **Macro-F1 (primary)** | **0.965288** | 0.968587 | −0.003299 |
| **Accuracy** | **0.968942** (3,307 / 3,413) | 0.975132 | −0.006189 |
| Macro-precision | 0.962876 | 0.963556 | −0.000680 |
| Macro-recall | 0.967902 | 0.974257 | −0.006355 |
| Weighted-F1 | 0.969006 | 0.975168 | −0.006163 |
| Compiled loss | 0.187019 | 0.174981 | +0.012037 |

A validation-to-test drop of 0.0033 macro-F1 and 0.62 pp accuracy indicates the split held: the
model was not tuned against the test data, and there is no evidence of leakage-driven optimism.

### 8.2 Best and worst classes

| | Class | Support | F1 |
|:--|:--|--:|--:|
| Best | 4 · `Potato___healthy` | 23 | **1.0000** (23/23) |
| Best | 3 · `Potato___Late_blight` | 150 | 0.9934 |
| Best | 12 · `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 803 | 0.9912 |
| Worst | 6 · `Tomato___Early_blight` | 150 | **0.8515** (recall 0.8600) |
| Worst | 11 · `Tomato___Target_Spot` | 210 | 0.9282 (recall 0.9238) |
| Worst | 10 · `Tomato___Spider_mites` | 252 | 0.9567 |

`Tomato___Early_blight` is the hardest class, and its errors are not random: 21 of its 150 test
images were missed, and the three largest destinations — Late Blight (6), Septoria leaf spot (4)
and Target Spot (3) — are exactly the three look-alike conditions listed in its own
knowledge-base record. The model's mistakes concentrate where a human's would, which is a
useful signal that the class mapping is sound rather than permuted.

**106 misclassifications in total** (3.11 %). Beyond the hard classes above, the residue is
spread thinly across conditions that genuinely overlap visually, not piled onto one label —
which is what a permuted or broken class mapping would look like.

### 8.3 Integrity audit of the evaluation

The reported metrics were independently recomputed from the 15×15 confusion matrix, entirely
from the matrix rather than from the model:

| Check | Expected | Recomputed | Status |
|:--|--:|--:|:--|
| Matrix shape | 15×15 | 15×15 | exact |
| Matrix sum | 3,413 | 3,413 | exact |
| Diagonal (correct) | 3,307 | 3,307 | exact |
| Off-diagonal (errors) | 106 | 106 | exact |
| Row sums vs class supports | 15/15 | 15/15 | exact |
| Accuracy / macro-P / macro-R / macro-F1 / weighted-F1 | — | all | 0.00e+00 discrepancy |

The audit also confirms the validation numbers quoted in the test report byte-match the
validation-comparison report, and that no test-time selection occurred.

---

## 9. Backend Service

**Stack:** FastAPI 0.152.0 · Uvicorn 0.54.0 · Pydantic 2.14.1 · TensorFlow 2.21.0 ·
Pillow 13.0.0 · NumPy 2.4.6 · Python 3.11. The service is **stateless and database-free**.

### 9.1 Endpoints

| Method | Path | Purpose | Codes |
|:--|:--|:--|:--|
| GET | `/` | Service identity and URL map | 200 |
| GET | `/health` | Process liveness | 200 |
| GET | `/ready` | Model **and** knowledge base loaded | 200 / 503 |
| POST | `/api/predict` | Single leaf image → diagnosis | 200 / 400 / 413 / 415 / 422 / 429 / 503 |
| GET | `/api/knowledge/{class_id}` | Agronomic record for class 0–14 | 200 / 404 / 422 / 503 |

### 9.2 Input-safety ladder

Each gate is deterministic, ordered, and cheap-first:

| Order | Gate | Rejection |
|--:|:--|:--|
| 1 | Upload field present | 400 |
| 2 | Model loaded | 503 |
| 3 | Declared MIME ∈ {JPEG, PNG, WEBP} | 415 |
| 4 | Streamed body ≤ 10 MB (64 KB chunks) | 413 |
| 5 | Body non-empty | 400 |
| 6 | Declared MIME agrees with magic bytes | 415 |
| 7 | Decode queue not saturated (≤ 10 waiters) | 429 |
| 8 | Container format ∈ allowed set, `verify()` passes | 415 / 400 |
| 9 | Pixels ≤ 20 MP (before raster decompression) | 422 |
| 10 | min(width, height) ≥ 64 px | 422 |
| 11 | Luminance σ > 2.0 on the 224×224 result | 422 |

Two design points are worth defending in a viva:

- **The 10 MB limit is applied while streaming**, so an oversized upload is abandoned during
  ingestion instead of being buffered and then measured. A gate that reads first and checks
  later is not a memory limit.
- **The blank-canvas threshold is empirically grounded, not guessed.** The exact production
  pipeline was replayed over all **3,418 validation images**: the minimum observed σ was
  **11.3144** (a severe necrotic late-blight lesion), with 0 false rejections at thresholds of
  1.0, 2.0, 3.0, 5.0 and even 10.0. The shipped threshold σ ≤ 2.0 therefore sits **5.65×**
  below the most uniform real leaf in the dataset — it rejects blank canvases without rejecting
  any legitimate sample.

Bilinear downsampling acts as a low-pass filter, which is why the check is computed *after*
resize: a synthetic 1-pixel checkerboard collapses from σ = 127.49 to 0.50 (correctly rejected),
while leaf veins and lesions survive (σ = 2.75 for a 1-pixel line, 11.02 for a 4-pixel pattern).

### 9.3 Concurrency

| Primitive | Value | Purpose |
|:--|--:|:--|
| `MAX_DECODE_CONCURRENCY` | 2 | bounds simultaneous CPU decodes and transient RAM |
| `MAX_DECODE_QUEUE_SIZE` | 10 | waiters beyond this get HTTP 429 instead of unbounded queueing |
| Inference semaphore | 1 | serialises model calls |
| Execution | `asyncio.to_thread` | keeps the event loop responsive during decode and inference |

Measured cost of the worst realistic case: a 20 MP WebP decodes to **80.16 MB peak RAM** and
takes 258.78 ms of CPU, so two concurrent decodes are bounded at roughly 160 MB. A 20 MP
*JPEG* is cheap by comparison — 0.44–0.94 MB on disk, 77–150 ms end to end. A 20 MP natural
PNG, by contrast, is a **60.03 MB file** and is stopped at the 10 MB stream gate with HTTP 413
before any decoding happens; the largest natural PNG that can pass the limit is about 3 MP.

The honest position, documented in the audit: the semaphore bounds *concurrent decodes*, not
total process memory. Upload buffers, allocator retention, and multi-worker multiplication all
sit outside its guarantee, which is why a reverse proxy body limit is called out as the correct
outer control.

### 9.4 Error contract

Every failure returns JSON with a human-readable `detail` string, so the front end can display
the server's own explanation without hard-coding status-code guesses. Internal paths and
tracebacks are logged server-side and never returned.

---

## 10. Agronomic Knowledge Base

`backend/app/data/disease_knowledge_base.json` (72 KB) holds **15 records**, one per model
class, with **13 top-level fields** each: identity (`class_id`, `raw_class`, `crop`,
`condition`, `health_status`, `causal_agent`), `symptoms`, `favorable_conditions`,
`prevention_and_cultural`, `treatment`, `extension_consultation`, `sources`, and
`safety_and_uncertainty_notes`.

**Sourcing:** 30 citations in total (2 per record) from TNAU Agritech (12), UC IPM (11), Ohio
State Extension (2), UF/IFAS EDIS (2), UMass Extension (1), Clemson HGIC (1), and Purdue
Vegetable Crops Hotline (1). Each source names the specific claims it substantiates and carries
both a live URL and a stable publication fallback.

**Startup validation** — the service refuses to serve a record it cannot trust:

- exactly 15 records, class IDs `0`–`14` unique and complete;
- `raw_class` / `crop` / `condition` / `health_status` must match the model metadata **1:1**;
- healthy classes (`1`, `4`, `14`) must have `causal_agent: null` and must **not** recommend a
  chemical spray;
- diseased classes must have a non-empty `causal_agent`;
- `crop_rotation` is enforced as a string; all list elements non-empty strings;
- every source must carry an HTTP(S) URL and non-empty metadata (validated syntactically,
  offline — a rotted link still passes, which is stated as a maintenance task, not a guarantee).

Records are returned as `copy.deepcopy`, so a caller cannot mutate cached server state.

This matters because class order is the classic silent failure of this kind of project: Keras
directory ordering is alphabetical and easy to permute. A permuted mapping would yield a model
that scores 96 % and gives every user the wrong disease. The parity assertion is what converts
that from a latent bug into a startup crash.

---

## 11. Front End

No build step, no framework, no dependency: `index.html` + `styles.css` + two JavaScript modules.

- **`api.js`** — `fetch` client with request timeouts, structured error mapping, and readiness
  probing.
- **`app.js`** — UI state machine: idle → file selected → preview → analysing → result.

Features: drag-and-drop or browse; client-side type/size pre-checks; preview with filename,
dimensions and size; a live backend-readiness pill; dismissible alert banner; primary diagnosis
with confidence and a health badge; top-3 candidates that are individually selectable to read
*their* reference entry (labelled as reference material, not a second diagnosis); and a rendered
agronomic profile with sections for symptoms, prevention, treatment, escalation and sources.

**Safety in rendering:** all server-provided text is inserted with `textContent`, never
`innerHTML`, and a citation only becomes a clickable link when `isValidHttpUrl()` passes.
Object URLs for previews are revoked on reset to avoid leaks. The result view states explicitly
that scores are not calibrated probabilities.

---

## 12. Testing and Verification

**99 automated tests**, all passing:

| Layer | Tests | What is covered |
|:--|--:|:--|
| Backend unit | 49 | health/readiness (7), prediction + every input-safety path (19), knowledge base validation and API contract (21), real-model integration (2) |
| ML pipeline | 22 | class mapping, split isolation, class weights, tie-breaking, architecture specification |
| Front-end API client | 28 | success paths, error mapping, timeout behaviour, 3 live integration tests (skipped when no server is listening) |

```bash
# Backend
.venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_*.py"

# ML pipeline
.venv/Scripts/python.exe -m unittest ml.tests.test_train_pipeline

# Front end
node --test frontend/test_api.js
```

Beyond unit tests, the system was verified end to end: a cold start on a fresh port was
confirmed to load both artifacts from disk (`/ready` → `model_loaded: true`,
`knowledge_loaded: true`), `/api/knowledge/0` returned the expected Bell Pepper record, and a
real upload of a `Tomato___healthy` image returned class 14 with score 0.99999. All nine UI
error paths and the server-side race conditions (double submit, mid-flight cancel, navigation
during request) were driven in a browser and behaved correctly.

### Claims → evidence

| Claim | Evidence |
|:--|:--|
| Dataset audited before use | `docs/audit_reports/plantvillage/dataset_audit_report.md` (+ `.json`) |
| Split is leak-free | `docs/audit_reports/split_verification.json` — 0 SHA-256 overlaps |
| Model chosen on validation only | `docs/audit_reports/model_validation_comparison.md` (+ `.json`) |
| 0.9653 test macro-F1 | `docs/audit_reports/final_test_evaluation_report.md` (+ `.json`) |
| Metrics are arithmetically consistent | `docs/audit_reports/test_evaluation_integrity_audit.md` (+ `.json`) |
| Safety thresholds reject no real leaf | `docs/audit_reports/phase5b_level_a_final_validation_audit.md` |
| OOD gate was considered and rejected | `docs/audit_reports/phase5b_ood_policy_verification.md` |
| KB schema and sourcing | `docs/KNOWLEDGE_BASE.md` |

---

## 13. Limitations

1. **Controlled-condition data.** PlantVillage images have uniform backgrounds. Field
   photography with soil, shadow, occlusion and clutter is out of distribution and will degrade
   accuracy in ways this project did not measure.
2. **No unsupported-crop rejection.** The model maps *any* leaf into one of its 15 classes. An
   embedding-centroid OOD gate was implemented and audited as an experiment, then rejected:
   it produced a **56 % false-accept rate** on unsupported crops while also falsely rejecting
   **5.8–7.6 %** of valid diseased leaves. Shipping it would have made the system confidently
   wrong in a new way, so the limitation is documented instead of papered over.
3. **Scores are not calibrated probabilities.** Softmax output is a relative distribution over
   15 closed-world classes. The API and UI both say so.
4. **Not a prescription.** Guidance is educational; pesticide decisions require label
   compliance and confirmation by a qualified agronomist or extension service.
5. **Five reviewed near-duplicate pairs remain.** Declared, not hidden.
6. **Retrospective safety measurements.** The Phase 5B harnesses (σ sweep, aliasing fixtures,
   upload benchmark, OOD experiment) were throwaway scripts and are no longer in the tree; their
   methodology and full result tables survive in the reports, so the conclusions are auditable
   by inspection but not re-runnable as-is.

---

## 14. Future Work

1. **Field-data evaluation** on a separately collected dataset with natural backgrounds, to
   measure the real degradation this report only predicts.
2. **Unsupported-crop rejection done properly** — e.g. an open-set head or energy-based score
   trained with explicit negative classes, evaluated against the false-accept/false-reject
   trade-off the rejected gate failed.
3. **Score calibration** (temperature scaling) on the validation split, so the UI can show
   intervals rather than raw softmax.
4. **Multi-leaf / whole-plant capture** support instead of a single close-up.
5. **Deployment hardening** — containerisation, a reverse-proxy body limit, and multi-worker
   memory measurement, which the concurrency audit identifies as the real outer bound.
6. **Localisation** of the knowledge-base text for regional growers.

---

## 15. Conclusion

The project delivers a complete, tested path from a leaf photograph to actionable agronomic
guidance: a MobileNetV2 classifier at **96.89 % accuracy / 0.9653 macro-F1** on a leak-audited
held-out test set, wrapped in a FastAPI service with an empirically justified input-safety
ladder and a startup-validated 15-record knowledge base, presented through a dependency-free
browser interface and verified by 99 automated tests.

Its main contribution is methodological rather than architectural: the dataset was certified
before training, model selection was confined to validation, the test set was evaluated once and
then arithmetically audited, the safety thresholds were derived from all 3,418 validation images
rather than chosen by intuition, and one attractive feature — out-of-distribution rejection —
was measured, found to be worse than the problem it solved, and rejected on the evidence. The
remaining limitations are stated in the same place as the results.

---

## 16. Viva Preparation

**Q1. Why MobileNetV2 rather than a larger network such as ResNet-50 or EfficientNet?**
The task has 15 classes, ~16k training images, and must run on CPU in a student environment. A
28 MB depthwise-separable model fine-tunes in 28 epochs to 0.9653 macro-F1, and inference is fast
enough for interactive use. A larger backbone would cost several times the training and inference
budget for a marginal gain this dataset's 256 px images cannot support.

**Q2. Why 224×224?**
It is MobileNetV2's native input size, so the pretrained filters apply without resampling, and
the source images are already ~256 px, so almost no detail is discarded. Upsampling to 320 or
384 would add compute without adding information.

**Q3. Why macro-F1 as the primary metric instead of accuracy?**
The dataset has a 36:1 imbalance, and the majority-class baseline is only 9.93 %. Accuracy is
dominated by the large classes — `Tomato___Tomato_Yellow_Leaf_Curl_Virus` alone is 803 of 3,413
test images. Macro-F1 averages the 15 classes equally, so a model cannot buy accuracy by
sacrificing the rare ones. `Potato___healthy` has just 23 test images; that is exactly the class
accuracy hides.

**Q4. How do you know the test result is not inflated by leakage?**
Three independent controls. Deduplication happened before the split, and grouping was by content
hash so duplicate clusters could not straddle boundaries; verification reported 0 SHA-256
overlaps across all three intersections and 0 split clusters. Model selection and the
weighted-vs-unweighted comparison used validation only. The test set was evaluated once, after
selection, and then the reported metrics were recomputed from the confusion matrix with 0.00e+00
discrepancy. The validation→test gap is only −0.0033 macro-F1, consistent with an honest split.

**Q5. What do the class weights do, and did they help?**
They scale the loss contribution inversely to training frequency — `Potato___healthy` gets 10.03,
`Tomato___Tomato_Yellow_Leaf_Curl_Virus` gets 0.28. On validation, macro-F1 improved by +0.0052
and macro-recall by +0.0109, with the clearest gain being rare-class recall (`Potato___healthy`
0.913 → 1.000). It cost 0.0002 macro-precision and 0.0044 validation loss, which was an
acceptable trade.

**Q6. Your model still answers when I upload a cucumber leaf. Why?**
Because it is a closed-world 15-class classifier, and no gate ships that reliably fixes this. We
built and audited an embedding-centroid OOD gate; it accepted 56 % of unsupported crops and
rejected 5.8–7.6 % of genuine diseased leaves. That is worse than the disease it treats, so it
was rejected on evidence and the limitation is documented. The honest fix is an open-set head
trained with negatives, which is listed as future work.

**Q7. How was the blank-image threshold of σ ≤ 2.0 chosen?**
Not by intuition. The exact production pipeline — EXIF transposition, dimension checks, RGB
conversion, bilinear resize to 224×224, ITU-R BT.601 luminance, population standard deviation —
was replayed over all 3,418 validation images. The minimum σ among real leaves was 11.3144, and
zero images fell at or below 1.0, 2.0, 3.0, 5.0 or even 10.0. The threshold sits 5.65× below the
most uniform real leaf, so it rejects blank canvases with zero false rejections.

**Q8. Why check σ after resizing instead of on the original?**
Because the model sees the resized image, so the check must describe the model's input. Bilinear
downsampling is a low-pass filter: it collapses a synthetic 1-pixel checkerboard from σ = 127.49
to 0.50 while preserving leaf scale features. Testing the original would let high-frequency
sensor noise mask a blank frame.

**Q9. What stops a 20-megapixel decompression bomb from exhausting memory?**
Layered controls: a 10 MB hard cap enforced *during* the 64 KB streaming read (so oversized
bodies are abandoned, not buffered), a 20 MP check evaluated before raster decompression, Pillow
decompression-bomb warnings escalated to errors, a decode semaphore of 2, a queue cap of 10
returning 429, and inference serialised behind its own semaphore. The audit is explicit that the
semaphore bounds concurrent decodes, not total process memory, so a reverse-proxy body limit is
the correct outer control.

**Q10. Why does a failed model load still return 200 on `/health`?**
Separation of concerns. `/health` answers "is the process alive", which an orchestrator uses to
avoid restart loops. `/ready` answers "can it serve", returning 503 with the specific component
that failed. Conflating them means a slow model load looks like a crashed process.

**Q11. How do you guarantee the knowledge base never shows the wrong disease?**
Class IDs 0–14 are the single key shared by the training script, the serving metadata, and the
JSON record. At startup the service asserts 1:1 parity on `raw_class`, `crop`, `condition` and
`health_status`, plus exactly 15 records with no gaps or duplicates, healthy classes having a
null causal agent and no chemical recommendation, and `crop_rotation` being a string. A mismatch
is a hard startup failure and `/api/knowledge/*` returns 503 — because a silently wrong
agronomic recommendation is worse than an outage.

**Q12. What would you do differently with more time?**
Three things: evaluate on genuinely field-captured images to quantify the distribution shift;
implement rejection with an open-set formulation rather than an unsupervised distance threshold;
and calibrate the softmax on validation so the UI can report uncertainty honestly instead of a
raw relative score.

**Q13. What is the practical takeaway for a grower?**
A fast, free second opinion on 15 common Solanaceae conditions, backed by cited extension
material and an explicit instruction to confirm with a local specialist before spending money on
chemicals. It is decision support, not a prescription.

---

## 17. Five-Minute Demonstration Script

```bash
# 1. Backend (loads model + knowledge base once, validates both)
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 2. Front end (serve statically — do not open via file://)
.venv/Scripts/python.exe -m http.server 5500 --directory frontend
```

1. Open <http://127.0.0.1:5500> — point out the readiness pill showing the backend is live.
2. Upload a leaf photo; show filename, dimensions, size, and the Analyse button.
3. Analyse; show the primary diagnosis with confidence, the health badge, and the top-3
   candidates. Switch to a runner-up to demonstrate that its reference entry loads on demand and
   is labelled as reference, not diagnosis.
4. Show the agronomic profile: symptoms, favourable conditions, prevention, IPM principles,
   escalation triggers, citations.
5. Demonstrate three rejections: a blank white image (422 blank canvas), a renamed `.png` that is
   really a JPEG (415 signature mismatch), and an oversized file (413 at the stream gate).
6. Show `/docs` for the contract and `/ready` for readiness.

---

## 18. References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2:
   Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
2. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to
   enable the development of mobile disease diagnostics.* arXiv:1511.08060. (PlantVillage)
3. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A
   Large-Scale Hierarchical Image Database.* CVPR 2009. (pretraining source)
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.*
   CVPR 2016. (residual/depthwise design context)
5. Tamil Nadu Agricultural University. *Agritech Portal — Crop Protection.*
   <https://agritech.tnau.ac.in>
6. University of California Statewide IPM Program. *Agriculture: Pest Management Guidelines.*
   <https://ipm.ucanr.edu>
7. FastAPI documentation. <https://fastapi.tiangolo.com>
8. TensorFlow / Keras documentation. <https://www.tensorflow.org/api_docs>

*Full per-disease citation set (30 sources) is recorded in
`backend/app/data/disease_knowledge_base.json` and summarised in `docs/KNOWLEDGE_BASE.md`.*

---

## Appendix A — Repository Map

```text
Project/
├── backend/
│   ├── app/
│   │   ├── main.py             app factory, lifespan loading, CORS
│   │   ├── config.py           paths, limits, thresholds, feature flags
│   │   ├── api/                health · predict · knowledge
│   │   ├── schemas/            Pydantic contracts
│   │   ├── services/           inference · knowledge
│   │   └── data/               disease_knowledge_base.json (15 records)
│   ├── tests/                  49 tests
│   └── README.md               full API contract
├── frontend/
│   ├── index.html · css/styles.css
│   ├── js/api.js · js/app.js
│   └── test_api.js             28 tests
├── ml/
│   ├── scripts/                audit_dataset · prepare_dataset · train_model · evaluate_*
│   ├── tests/                  22 tests
│   ├── data/                   manifest + raw dataset (gitignored)
│   └── models/                 experiment_weighted (served) · experiment_unweighted (baseline)
├── docs/
│   ├── PROJECT_SUMMARY.md      this document
│   ├── README.md               evidence index
│   ├── KNOWLEDGE_BASE.md       data specification
│   └── audit_reports/          9 reports + machine-readable JSON
└── README.md                   system overview and quick start
```

## Appendix B — Key Configuration

| Setting | Default | Meaning |
|:--|--:|:--|
| Max upload size | 10 MB | enforced during streaming |
| Max decoded pixels | 20,971,520 | ~20 MP, checked before decompression |
| Min image dimension | 64 px | width and height |
| Blank-canvas σ threshold | 2.0 | luminance population std on 224×224 |
| Max concurrent decodes | 2 | semaphore |
| Decode queue size | 10 | beyond this → HTTP 429 |
| Target image size | 224×224×3 | raw 0–255 pixels; rescaling is inside the model |
| Model path | `ml/models/experiment_weighted/best_model.keras` | override with `MODEL_PATH` |
| Knowledge base | `backend/app/data/disease_knowledge_base.json` | override with `KNOWLEDGE_BASE_PATH` |
| CORS origins | localhost/127.0.0.1 on 3000, 8000, 5500 | comma list via `CORS_ORIGINS` |
