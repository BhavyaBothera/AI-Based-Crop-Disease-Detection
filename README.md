# AI-Based Crop Disease Detection and Treatment Recommendation System

An end-to-end agricultural diagnostic application. A user uploads a leaf photograph; a
fine-tuned MobileNetV2 model classifies it into one of **15 Solanaceae conditions**
(Tomato, Potato, Bell Pepper), and the app retrieves validated agronomic guidance for that
condition: diagnostic symptoms, favorable conditions, cultural prevention, IPM treatment
principles, extension contacts, and cited sources.

**Stack:** FastAPI (Python 3.11) · TensorFlow 2.21 / Keras 3.15 · Vanilla HTML/CSS/JS ·
Pydantic · Pillow. The agronomic knowledge base is a versioned JSON file - there is no
database, no authentication, and no external service dependency. The backend is stateless.

---

## Supported Crops and Classes

Class IDs `0`-`14` are authoritative and shared by the training pipeline, the serving
backend, and the knowledge base.

| ID | Raw Class | Crop | Condition | Health |
|:--:|:----------|:-----|:----------|:-------|
| 0 | `Pepper,_bell___Bacterial_spot` | Bell Pepper | Bacterial Spot | Diseased |
| 1 | `Pepper,_bell___healthy` | Bell Pepper | Healthy | Healthy |
| 2 | `Potato___Early_blight` | Potato | Early Blight | Diseased |
| 3 | `Potato___Late_blight` | Potato | Late Blight | Diseased |
| 4 | `Potato___healthy` | Potato | Healthy | Healthy |
| 5 | `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | Diseased |
| 6 | `Tomato___Early_blight` | Tomato | Early Blight | Diseased |
| 7 | `Tomato___Late_blight` | Tomato | Late Blight | Diseased |
| 8 | `Tomato___Leaf_Mold` | Tomato | Leaf Mold | Diseased |
| 9 | `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | Diseased |
| 10 | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-spotted Spider Mite | Diseased |
| 11 | `Tomato___Target_Spot` | Tomato | Target Spot | Diseased |
| 12 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Yellow Leaf Curl Virus | Diseased |
| 13 | `Tomato___Tomato_mosaic_virus` | Tomato | Mosaic Virus | Diseased |
| 14 | `Tomato___healthy` | Tomato | Healthy | Healthy |

---

## Model and Recorded Results

- **Architecture:** MobileNetV2 (ImageNet weights) → GlobalAveragePooling → BatchNorm →
  Dense(256, L2) → Dropout(0.35) → Dense(15, softmax). Normalization is embedded in the
  model as a `Rescaling(1/127.5, -1)` layer, so the API feeds raw `[0, 255]` pixels.
- **Training:** two-stage transfer learning (frozen backbone, then layers 100-153
  unfrozen), balanced class weights, checkpoint selected by full-validation macro-F1.
- **Data:** PlantVillage, deduplicated and split by content hash group so near-duplicates
  never straddle splits. 15,942 train / 3,418 validation / 3,413 test images.
- **Held-out test set (evaluated once):** **macro-F1 0.9653**, **accuracy 96.89 %**
  (3,307 / 3,413). Validation macro-F1 was 0.9686.
- **Serving checkpoint:** `ml/models/experiment_weighted/best_model.keras`.

Full per-class metrics, confusion matrices, and experimental comparisons are in
[`docs/audit_reports/`](docs/audit_reports/); see [`docs/README.md`](docs/README.md) for an
index of what each report establishes and how to reproduce it.

---

## Project Structure

```text
Project/
├── backend/                    # FastAPI service
│   ├── app/
│   │   ├── main.py             # App factory, lifespan model/knowledge loading, CORS
│   │   ├── config.py           # Paths, limits, thresholds, feature flags
│   │   ├── api/                # /health, /ready, /api/predict, /api/knowledge/{id}
│   │   ├── schemas/            # Pydantic request/response contracts
│   │   ├── services/           # Model lifecycle + preprocessing; knowledge validation
│   │   └── data/               # 15-record disease knowledge base (JSON)
│   └── tests/                  # 75 backend tests
├── frontend/                   # No-build vanilla HTML/CSS/JS single page
│   ├── index.html
│   ├── css/styles.css
│   ├── js/api.js               # API client (timeouts, error mapping)
│   ├── js/app.js               # UI controller and safe DOM rendering
│   ├── test_api.js             # Node test-runner suite
│   └── test_browser.py         # Headless Chrome E2E (desktop + mobile)
├── ml/
│   ├── scripts/                # audit_dataset, prepare_dataset, train_model, evaluate_*
│   ├── tests/                  # 22 pipeline tests
│   ├── data/                   # dataset + manifest (gitignored)
│   └── models/                 # checkpoints (gitignored)
├── docs/
│   ├── PROJECT_SUMMARY.md      # Consolidated report for submission and viva
│   ├── README.md               # Index of the evidence, and how to reproduce it
│   ├── KNOWLEDGE_BASE.md       # Agronomic record schema and sourcing standard
│   └── audit_reports/          # Dataset, evaluation, safety and backend audit reports
├── .venv/                      # Local Python 3.11 environment (gitignored)
├── .gitignore
└── README.md
```

Large artifacts - the dataset, the raw archive, and `.keras` checkpoints - are excluded by
`.gitignore` and must exist locally to train or serve.

---

## Running Locally

### 1. Backend

```bash
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

On startup the model and the knowledge base are loaded once and validated. If either fails,
the process still starts, `/health` stays green, and `/ready` reports `503` with the reason.

- Swagger UI: <http://127.0.0.1:8000/docs>
- Readiness: `GET /ready` must return `model_loaded: true, knowledge_loaded: true`

### 2. Frontend

Serve the folder statically (do not open `index.html` via `file://`):

```bash
.venv/Scripts/python.exe -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>. The page targets `http://127.0.0.1:8000` by default; set
`window.API_BASE_URL` before the scripts load to point elsewhere. That origin is already in
the backend's CORS allow-list.

### 3. Using it

Drop a leaf photo (JPEG/PNG/WEBP, ≤ 10 MB, ≤ 20 MP, ≥ 64×64) and press **Analyze**. The
result shows the primary diagnosis, the top-3 candidates with scores, and the agronomic
reference profile. Any top-3 candidate can be selected to read *its* reference entry - the
UI labels that clearly as reference material, not a diagnosis.

If the image cannot be verified as plant material, the app shows a dedicated rejection state
instead: no diagnosis, no candidates, no treatment guidance and no advisory export. It
explains "We couldn't verify that this image contains a supported crop leaf" and offers
**Upload another image**.

---

## Testing

```bash
# Backend: 98 tests (health, prediction + input safety gate, knowledge base, real model)
.venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_*.py"

# ML pipeline: 22 tests (class mapping, split isolation, class weights, architecture)
.venv/Scripts/python.exe -m unittest ml.tests.test_train_pipeline

# Frontend API client: 36 tests
node --test frontend/test_api.js

# Leaf-gate training and held-out evaluation (see ml/README.md)
.venv/Scripts/python.exe ml/scripts/train_input_gate.py
.venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py

# Browser end-to-end: starts uvicorn + a static server + headless Chrome, desktop and mobile
.venv/Scripts/python.exe frontend/test_browser.py --static-port 5510
```

The frontend API suite includes three live integration tests against a running backend; they
report as **skipped** when nothing is listening on the configured base URL. The browser suite
requires Chrome and the trained model, and writes screenshots to
`docs/audit_reports/non_leaf_safety_screenshots/`.

---

## Input Safety

Input safety is layered.

**Layer A - physical integrity (unchanged).** Streamed 10 MB upload cap, JPEG/PNG/WEBP
signature verification, declared-MIME vs. detected-signature agreement, 20 MP decompression
ceiling, 64 px minimum dimension, and blank-canvas rejection based on luminance sigma
(`σ ≤ 2.0`). The σ threshold was validated over all 3,418 validation images: the minimum
observed σ for a real leaf was 11.31. Decoding runs in a worker thread under a concurrency
cap of 2 with a bounded queue, and inference is serialized.

**Layer B - leaf acceptance (trained model).** A dedicated 3-class model
(`supported_leaf` / `other_plant` / `non_leaf`) runs on the same 224×224 RGB frame the disease
model would see, before inference. It is a linear head on frozen ImageNet MobileNetV2 features,
trained by `ml/scripts/train_input_gate.py` and shipped as one self-contained artefact
(`ml/models/input_gate/leaf_gate.keras`) - no second download, no new serving dependency.
Rejected and abstained inputs never reach the disease model; they return a structured `422`
whose `detail` carries `status` (`rejected`/`abstained`), `code`, `category`, a user message and
the measured signals - and no diagnosis fields. Successful `200` responses keep the existing
contract. A deterministic plant-pigmentation screen remains in the same module as a fallback
for the case where the model file is missing or `ENABLE_LEAF_GATE=false`.

Measured on a disjoint held-out split (never trained on, disease `test` split never opened):

| Input | n | Blocked (no diagnosis) |
|:--|--:|--:|
| Valid tomato/potato/pepper leaves | 1,698 | **1.7%** (1.1% hard rejection) |
| Leaf-free field photographs | 193 | **100%** |
| Real portraits, animals, buildings, documents, objects, graphics | 70 | **100%** |
| Unsupported crop leaves (seen species) | 1,200 | **97.8%** |
| Unsupported crop leaves (Strawberry, Soybean - held out of training) | 600 | **75.7% / 53.3%** |

Validation accuracy of the head is 0.987 on 4,293 held-out embeddings, and the exported Keras
graph reproduces the trained sklearn decision (max |Δ| 3e-7). Gate cost is ~58 ms mean CPU per
image (p95 84 ms), serialized like disease inference.

**This is not a fully general leaf detector.** It rejects the demonstrated failure mode
(people, animals, buildings, documents, objects) reliably and refuses most unsupported species,
but foliage-heavy scenes that resemble foliage close-ups and visually unusual supported leaves
still slip through or get declined; softmax confidence is deliberately not used as an
acceptance criterion. Root cause, architecture, evaluation methodology, provenance and
remaining risks are in
[`docs/audit_reports/non_leaf_input_safety_report.md`](docs/audit_reports/non_leaf_input_safety_report.md).

---

## Known Limitations

- **Controlled-condition data.** PlantVillage images were captured on uniform backgrounds.
  Field photography with soil, shadows, and clutter is out of distribution.
- **Unsupported-crop rejection is good, not complete.** The gate declines 97.8 % of
  unsupported crop leaves from species it saw in training, but only 75.7 % of Strawberry and
  53.3 % of Soybean leaves (species held out entirely). Anything that passes is still forced
  into one of the 15 Solanaceae classes.
- **The gate is trained on a small negative corpus.** 70 real non-leaf photographs
  (licence-cleared Wikimedia Commons, 6 categories) are split between training and evaluation,
  so the 100 % block rate on them has wide uncertainty. Field photography with foliage in
  frame, hands, soil, heavy blur or green-tinted graphics is under-represented, and a small
  number of valid leaves (1.7 %) are declined.
- **Procedurally drawn or stylised leaves are rejected** as non-plant; the gate learned from
  photographs, not illustrations.
- **Scores are not calibrated probabilities.** Softmax outputs are a relative distribution
  over 15 closed-world classes, not diagnostic certainty.
- **Not a treatment prescription.** Guidance is educational; pesticide decisions require
  label compliance and confirmation by a qualified agronomist or extension service.
