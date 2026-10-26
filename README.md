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
│   └── tests/                  # 49 backend tests
├── frontend/                   # No-build vanilla HTML/CSS/JS single page
│   ├── index.html
│   ├── css/styles.css
│   ├── js/api.js               # API client (timeouts, error mapping)
│   ├── js/app.js               # UI controller and safe DOM rendering
│   └── test_api.js             # Node test-runner suite
├── ml/
│   ├── scripts/                # audit_dataset, prepare_dataset, train_model, evaluate_*
│   ├── tests/                  # 22 pipeline tests
│   ├── data/                   # dataset + manifest (gitignored)
│   └── models/                 # checkpoints (gitignored)
├── docs/
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

---

## Testing

```bash
# Backend: 49 tests (health, prediction + input safety, knowledge base, real-model integration)
.venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_*.py"

# ML pipeline: 22 tests (class mapping, split isolation, class weights, architecture)
.venv/Scripts/python.exe -m unittest ml.tests.test_train_pipeline

# Frontend API client: 28 tests
node --test frontend/test_api.js
```

The frontend suite includes three live integration tests against a running backend; they
report as **skipped** when nothing is listening on the configured base URL.

---

## Input Safety

Before an image reaches the model, the backend enforces: streamed 10 MB upload cap,
JPEG/PNG/WEBP signature verification, declared-MIME vs. detected-signature agreement,
20 MP decompression ceiling, 64 px minimum dimension, and a blank-canvas rejection based
on luminance sigma (`σ ≤ 2.0`). Decoding runs in a worker thread under a concurrency cap of
2 with a bounded queue, and inference is serialized. Rejections return specific
`400/413/415/422/429` codes with user-facing detail strings.

The σ threshold was validated over all 3,418 validation images: the minimum observed σ for
a real leaf was 11.31, so the threshold rejects no legitimate samples. See
[`docs/audit_reports/phase5b_level_a_final_validation_audit.md`](docs/audit_reports/phase5b_level_a_final_validation_audit.md).

---

## Known Limitations

- **Controlled-condition data.** PlantVillage images were captured on uniform backgrounds.
  Field photography with soil, shadows, and clutter is out of distribution.
- **No unsupported-crop rejection.** The model will force any leaf into one of its 15
  classes. An embedding-centroid OOD gate was audited and rejected (56 % false-accept rate
  on unsupported crops, 5.8-7.6 % false rejection of valid diseased leaves).
- **Scores are not calibrated probabilities.** Softmax outputs are a relative distribution
  over 15 closed-world classes, not diagnostic certainty.
- **Not a treatment prescription.** Guidance is educational; pesticide decisions require
  label compliance and confirmation by a qualified agronomist or extension service.
