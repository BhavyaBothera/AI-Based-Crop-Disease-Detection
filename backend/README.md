# Crop Disease Detection - FastAPI Backend

This service provides an inference API and agronomic knowledge base for the AI-Based Crop Disease Detection and Treatment Recommendation System. It serves predictions for 15 Solanaceae crop conditions using a fine-tuned MobileNetV2 deep learning model and serves validated, evidence-backed agronomic guidance, cultural prevention, IPM principles, and extension escalation pathways for each condition.

---

## 1. Architecture Overview

- **Framework**: FastAPI (Asynchronous Python 3.11)
- **Inference Runtime**: TensorFlow 2.21 / Keras 3.15
- **Selected Checkpoint**: `ml/models/experiment_weighted/best_model.keras`
- **Knowledge Base**: Curated, evidence-backed knowledge base at `backend/app/data/disease_knowledge_base.json`
- **Class Space**: 15 Solanaceae classes across Tomato, Potato, and Bell Pepper
- **Service Lifecycles**:
  - Model service: Loaded once during FastAPI application startup via lifespan context manager.
  - Knowledge base service: Loaded, validated, and cached in-memory once during application startup. Defensive copies are served to guarantee immutability.

---

## 2. Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI entrypoint, lifespan loader, CORS middleware
│   ├── config.py          # Environment, paths, limits, and CORS settings
│   ├── data/
│   │   └── disease_knowledge_base.json # Curated 15-class agronomic knowledge base
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── prediction.py  # Pydantic schemas for health, readiness, and predictions
│   │   └── knowledge.py   # Pydantic schemas for disease knowledge records
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference.py   # Model service, Pillow preprocessing, 15-class mapping
│   │   └── knowledge.py   # Knowledge service, startup validation, defensive access
│   └── api/
│       ├── __init__.py
│       ├── health.py      # GET /health, GET /ready
│       ├── predict.py     # POST /api/predict
│       └── knowledge.py   # GET /api/knowledge/{class_id}
├── tests/
│   ├── __init__.py
│   ├── test_health.py     # Unit tests for health and readiness endpoints
│   ├── test_predict.py    # Unit tests for prediction, safeguards, and preprocessing
│   ├── test_knowledge.py  # Unit & integration tests for knowledge service and API
│   └── test_model_integration.py # Integration test using actual weights
├── requirements.txt       # Backend dependencies
└── README.md              # Documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11
- Virtual environment at `.venv/`

### Install Dependencies
From the project root:
```powershell
.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

---

## 4. Configuration

All configuration settings are defined in [`backend/app/config.py`](app/config.py) and can be overridden via environment variables:

| Environment Variable | Default Value | Description |
|----------------------|---------------|-------------|
| `MODEL_PATH` | `ml/models/experiment_weighted/best_model.keras` | Absolute or relative path to the trained `.keras` model |
| `KNOWLEDGE_BASE_PATH` | `backend/app/data/disease_knowledge_base.json` | Path to the curated disease knowledge base JSON file |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500` | Comma-separated list of allowed frontend origins (no wildcards with credentials) |
| `MAX_IMAGE_PIXELS` | `20971520` (20 MP) | Maximum decoded pixel limit to prevent decompression bombs |
| `ENABLE_LEVEL_A_PREVALIDATION` | `true` | Enables the deterministic minimum-dimension and blank-canvas checks (rollback switch) |
| `MIN_IMAGE_DIMENSION` | `64` | Minimum accepted width and height in pixels |
| `UNIFORM_CANVAS_STD_THRESHOLD` | `2.0` | Luminance sigma at or below which an image is rejected as a blank canvas |
| `MAX_DECODE_CONCURRENCY` | `2` | Maximum concurrent image decodes (bounds transient CPU and RAM use) |
| `MAX_DECODE_QUEUE_SIZE` | `10` | Maximum queued decode requests before overload rejection (HTTP 429) |

---

## 5. Starting the Backend Server Locally

To run the local development server with auto-reload:
```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 6. Endpoints & API Contract

### `GET /health`
Returns process operational status.
- **Status Code**: `200 OK`
- **Response**:
```json
{
  "status": "ok",
  "app": "AI-Based Crop Disease Detection API",
  "version": "1.0.0"
}
```

### `GET /ready`
Reports whether the ML model **and** the agronomic knowledge base are loaded and ready to serve requests.
- **Status Code**: `200 OK` (both loaded) or `503 Service Unavailable` (either uninitialized)
- **Response (Ready)**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "knowledge_loaded": true,
  "model_path": "ml/models/experiment_weighted/best_model.keras"
}
```

### `POST /api/predict`
Accepts a single leaf image file upload via multipart/form-data.

- **Request Fields**:
  - `file` (or `image`): Binary image data (`image/jpeg`, `image/png`, or `image/webp`).
- **File Constraints**:
  - Maximum upload size: **10 MB** (enforced in bounded streaming chunks; larger uploads aborted immediately).
  - Maximum decoded image resolution: **20 Megapixels** (default `20,971,520` pixels, configurable via `MAX_IMAGE_PIXELS`).
  - Minimum image dimension: **64 x 64 pixels** (`MIN_IMAGE_DIMENSION`).
  - Blank or solid-colour canvases (luminance sigma <= `UNIFORM_CANVAS_STD_THRESHOLD`) are rejected before inference.
  - Supported formats: **JPEG**, **PNG**, **WEBP**.
- **Response Codes**: `200` success, `400` missing/empty/corrupt upload, `413` oversized upload, `415` unsupported or mismatched media type, `422` invalid dimensions or blank canvas, `429` peak load, `503` model unavailable.
- **Response Format**: `200 OK`
```json
{
  "predicted_class_id": 2,
  "raw_class": "Potato___Early_blight",
  "crop": "Potato",
  "condition": "Early Blight",
  "health": "Diseased",
  "score": 0.9854,
  "top_predictions": [
    {
      "rank": 1,
      "class_id": 2,
      "raw_class": "Potato___Early_blight",
      "crop": "Potato",
      "condition": "Early Blight",
      "health": "Diseased",
      "score": 0.9854
    },
    {
      "rank": 2,
      "class_id": 3,
      "raw_class": "Potato___Late_blight",
      "crop": "Potato",
      "condition": "Late Blight",
      "health": "Diseased",
      "score": 0.0112
    },
    {
      "rank": 3,
      "class_id": 6,
      "raw_class": "Tomato___Early_blight",
      "crop": "Tomato",
      "condition": "Early Blight",
      "health": "Diseased",
      "score": 0.0021
    }
  ],
  "all_scores": {
    "Pepper,_bell___Bacterial_spot": 0.0001,
    "Pepper,_bell___healthy": 0.0000,
    "Potato___Early_blight": 0.9854,
    "Tomato___healthy": 0.0001
  },
  "disclaimer": "This prediction is generated by an automated machine learning model for informational and educational purposes only. Softmax scores represent the model's relative output distribution across its 15 trained classes and should not be treated as calibrated probabilities or guaranteed diagnostic certainty. Always seek confirmation from a qualified agronomic expert or local agricultural extension service before undertaking crop treatment."
}
```

### `GET /api/knowledge/{class_id}`
Retrieves validated agronomic knowledge, diagnostic symptoms, prevention, IPM principles, extension escalation guidelines, and source references for a specific model class ID (0-14).

- **Path Parameters**:
  - `class_id` (`integer`, required): Authoritative class ID from `0` through `14`.
- **Response Status Codes**:
  - `200 OK`: Successful retrieval of knowledge record.
  - `404 Not Found`: Unknown class ID (outside `0` to `14`).
  - `422 Unprocessable Entity`: Non-integer class ID format (e.g. string).
  - `503 Service Unavailable`: Knowledge base service is not loaded or unavailable.
- **Example Request**:
```http
GET /api/knowledge/2 HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```
- **Example Response (`200 OK`)**:
```json
{
  "class_id": 2,
  "raw_class": "Potato___Early_blight",
  "crop": "Potato",
  "condition": "Early Blight",
  "health_status": "Diseased",
  "causal_agent": "Alternaria solani (Ascomycete fungus)",
  "symptoms": {
    "summary": "Characteristic circular to angular dark brown target spots with concentric rings on older lower leaves, surrounded by yellow chlorotic halos.",
    "foliar_symptoms": [
      "Small 1-2 mm dark brown spots on older lower leaves",
      "Lesions enlarge up to 10 mm developing distinctive concentric target-ring ridges",
      "Yellow chlorotic halos surround lesions",
      "Extensive blighting and premature senescence of lower foliage"
    ],
    "stem_and_fruit_symptoms": [
      "Elongated dark brown to black sunken lesions on stems",
      "Dark, sunken, dry, leathery circular lesions on tuber skin"
    ],
    "look_alike_conditions": [
      "Late blight (water-soaked irregular lesions with white sporulation on undersides)",
      "Septoria leaf spot (smaller spots with grey centers and black pycnidia)",
      "Brown spot (Alternaria alternata)"
    ]
  },
  "favorable_conditions": {
    "environmental_factors": "Alternating wet and dry periods; warm temperatures between 24°C and 29°C; extended leaf wetness from dew or sprinkler irrigation",
    "transmission_routes": [
      "Overwinters in solanaceous crop debris and volunteer potato plants",
      "Conidia spread by wind, water splashing, and field machinery"
    ]
  },
  "prevention_and_cultural": {
    "sanitation": [
      "Destroy or deeply incorporate infected potato crop residue immediately after harvest",
      "Eradicate volunteer potato plants and solanaceous weeds (nightshades)",
      "Plant only certified disease-free seed tubers"
    ],
    "crop_rotation": "Maintain a 2- to 3-year crop rotation with non-solanaceous crops (cereals, legumes, corn, grasses)",
    "irrigation_and_airflow": [
      "Use drip or furrow irrigation to minimize canopy moisture",
      "Time sprinkler irrigation for morning hours to allow rapid leaf drying",
      "Avoid planting in low-lying, poorly ventilated fields"
    ],
    "resistant_cultivars": [
      "Select medium to late-maturing cultivars with documented partial resistance to early blight"
    ]
  },
  "treatment": {
    "urgency": "Action Required: Initiate protective management when lower leaf target spots are first detected.",
    "non_chemical_controls": [
      "Maintain adequate plant nutrition (avoid nitrogen deficiency which increases foliage susceptibility)",
      "Remove heavily infected lower leaves in smallholder garden settings"
    ],
    "ipm_chemical_principles": [
      "Preventive protectant fungicides (mancozeb, chlorothalonil) applied prior to extensive infection",
      "Rotate FRAC fungicide groups (e.g. FRAC 7 SDHI, FRAC 11 QoI) to mitigate Alternaria resistance",
      "Monitor disease forecasting models (e.g. TomCast / P-Days) to optimize spray timing"
    ],
    "safety_and_regulatory_constraints": [
      "Observe statutory Pre-Harvest Intervals (PHI) and Re-Entry Intervals (REI)",
      "Adhere strictly to local pesticide registration and label requirements",
      "Do not apply systemic fungicides on an emergency basis without verified diagnosis"
    ]
  },
  "extension_consultation": {
    "escalation_triggers": [
      "Rapid upward spread of target lesions before tuber bulking",
      "Failure of protectant sprays suggesting QoI (FRAC 11) fungicide resistance"
    ],
    "local_specialist": "District Agricultural Extension Officer, Krishi Vigyan Kendra (KVK) Plant Pathologist, or University Extension Specialist"
  },
  "sources": [
    {
      "organization": "University of Idaho Extension",
      "title": "Potato Early Blight (Alternaria solani)",
      "url": "https://www.uidaho.edu/extension/ipm/diseases/early-blight-potato",
      "claims_supported": "Causal agent, concentric target lesions, environmental drivers, tuber symptoms, and resistance management",
      "reference_url_or_publication": "https://www.uidaho.edu/extension/ipm/diseases/early-blight-potato"
    },
    {
      "organization": "University of California Statewide IPM Program (UC IPM)",
      "title": "Potato: Early Blight (Alternaria solani)",
      "url": "https://ipm.ucanr.edu/agriculture/potato/early-blight/",
      "claims_supported": "Fungicide timing, FRAC group rotation, irrigation management, sanitation, and look-alikes",
      "reference_url_or_publication": "https://ipm.ucanr.edu/agriculture/potato/early-blight/"
    }
  ],
  "safety_and_uncertainty_notes": "Early blight target lesions can be visually confused with brown spot or ozone injury. Physical field verification or lab sporulation testing is strongly advised before intensive chemical intervention."
}
```

---

## 7. Knowledge Base Loading & Validation Architecture

The Knowledge Base Service ([`backend/app/services/knowledge.py`](app/services/knowledge.py)) enforces strict startup validation:

1. **Deterministic Path Resolution**: Resolves `backend/app/data/disease_knowledge_base.json` relative to the project root, independent of current working directory.
2. **Exact Class Parity**: Validates all 15 records against the authoritative `TARGET_CLASS_MAPPING` from `ml/scripts/train_model.py` and `backend/app/services/inference.py`:
   - Checks that class IDs are unique integers `0` through `14`.
   - Validates that `raw_class`, `crop`, `condition`, and `health_status` match model metadata exactly.
3. **Healthy-Class Safety Contract**:
   - Classes 1, 4, 14 (`Pepper,_bell___healthy`, `Potato___healthy`, `Tomato___healthy`) must have `causal_agent: null`.
   - Explicitly checks that `ipm_chemical_principles` does not recommend any fungicide or pesticide application for healthy crops.
4. **Field Type & Structure Enforcement**:
   - Strictly enforces `prevention_and_cultural.crop_rotation` as a string (`str`), preventing list/array serialization errors.
   - Validates all nested symptoms, environmental factors, cultural controls, and extension escalation fields as non-empty.
5. **Offline Source Verification**:
   - Validates that each record has at least one source citation with valid HTTP/HTTPS URI scheme, organization, title, publication reference, and substantiated claims.
   - Performed completely in-memory without making runtime network requests.
6. **Defensive Immutability**:
   - Records are parsed once at startup and stored in memory.
   - `knowledge_service.get_disease_record(class_id)` returns a deep defensive copy (`copy.deepcopy`), preventing any caller or downstream service from mutating cached state.

---

## 8. Error Handling & Safeguards

| Error Condition | HTTP Code | Error Response Detail |
|-----------------|-----------|-----------------------|
| Missing file upload | `400 Bad Request` | `"No image uploaded. Provide a file in multipart form field 'file' or 'image'."` |
| Empty file (0 bytes) | `400 Bad Request` | `"Uploaded file is empty (0 bytes)."` |
| Corrupt / truncated image body | `400 Bad Request` | `"Corrupt or unreadable image file: ..."` |
| File bytes match no supported signature | `415 Unsupported Media Type` | `"Unsupported image format. Supported formats: JPEG, PNG, WEBP."` |
| Declared MIME type not supported (e.g. `image/gif`) | `415 Unsupported Media Type` | `"Unsupported media type 'image/gif'. Supported MIME types: ..."` |
| Declared MIME type contradicts file signature | `415 Unsupported Media Type` | `"Declared MIME type 'image/jpeg' does not match detected file signature 'PNG'."` |
| Dimensions below 64 x 64 or above 20 MP | `422 Unprocessable Entity` | `"Image resolution (32x32) is too small. Minimum required resolution is 64x64 pixels."` / `"... exceeds maximum permitted limit of 20,971,520 pixels."` |
| Blank or solid-colour canvas (sigma <= 2.0) | `422 Unprocessable Entity` | `"Uploaded image appears to be a blank or solid-color canvas. Please provide a clear photograph of a crop leaf."` |
| Decompression bomb warning / error | `422 Unprocessable Entity` | `"Image exceeds safe decompression limits: ..."` |
| Oversized upload (> 10 MB) | `413 Content Too Large` | `"Uploaded file exceeds maximum permitted size of 10 MB."` |
| Peak request volume (decode queue full) | `429 Too Many Requests` | `"Server is currently handling peak request volume. Please retry your request shortly."` |
| Unknown knowledge class ID (`< 0` or `> 14`) | `404 Not Found` | `"Disease knowledge record not found for class_id {class_id}."` |
| Non-integer knowledge class parameter | `422 Unprocessable Entity` | Standard FastAPI parameter validation error |
| Model not loaded / unready | `503 Service Unavailable` | `"Model inference service is currently unavailable."` (no internal paths or tracebacks leaked) |
| Knowledge base not loaded / unready | `503 Service Unavailable` | `"Knowledge base service is currently unavailable."` (no internal paths or tracebacks leaked) |

---

## 9. Preprocessing Pipeline Specification

The inference pipeline performs the following steps on incoming images
(executed in a worker thread via `asyncio.to_thread` behind `MAX_DECODE_CONCURRENCY`):
1. **Signature Sniffing**: Checks JPEG / PNG / WEBP magic bytes and rejects anything else (`415`).
2. **Header Verification**: Confirms the Pillow-detected container format matches the sniffed signature, then calls `Image.verify()`.
3. **Dimension Ceiling**: Rejects images above `MAX_IMAGE_PIXELS` *before* raster decompression (`422`), after EXIF transposition.
4. **Minimum Dimension**: Rejects images smaller than `MIN_IMAGE_DIMENSION` x `MIN_IMAGE_DIMENSION` (`422`).
5. **RGB Normalization**: Converts non-RGB images (RGBA, LA, palette, grayscale, CMYK) to 3-channel RGB (`Image.convert("RGB")`), then explicitly loads the raster to catch truncated scanlines.
6. **Resizing**: Resizes image to `(224, 224)` using bilinear interpolation (`Image.Resampling.BILINEAR`), matching the `tf.image.resize` training preprocessing.
7. **Blank-Canvas Guard**: Computes population standard deviation of ITU-R BT.601 luminance on the 224x224 result and rejects sigma <= `UNIFORM_CANVAS_STD_THRESHOLD` (`422`).
8. **Input Range & Dimension**: Converts image to float32 NumPy array with shape `(1, 224, 224, 3)` in the raw `[0.0, 255.0]` range.
   > **Note on Rescaling**: The trained `.keras` checkpoint contains an embedded `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` as Layer 1. Preprocessing must NOT normalize pixel values to `[0, 1]` or `[-1, 1]` prior to model ingestion.

---

## 10. Authoritative 15-Class Mapping

Class indices strictly adhere to `ml/scripts/train_model.py`:

| ID | Raw Class Identifier | Crop | Condition | Health |
|:--:|:---------------------|:-----|:----------|:-------|
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

## 11. Automated Testing

### Run Lightweight Unit Tests (Fast, mocked model, full knowledge base validation):
```powershell
.venv\Scripts\python.exe -m unittest backend/tests/test_health.py backend/tests/test_predict.py backend/tests/test_knowledge.py
```

### Run Model Integration Test (Requires TensorFlow & trained model weights):
```powershell
.venv\Scripts\python.exe -m unittest backend/tests/test_model_integration.py
```

### Run All Backend Tests:
```powershell
.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

---

## 12. Informational Disclaimer

> **Important**: This automated machine learning model is intended solely for educational and informational purposes. Softmax scores represent relative model output distribution across the 15 supported training classes and are not calibrated probabilities or diagnostic guarantees. Automated predictions must be confirmed by an agricultural extension specialist or agronomist before taking crop treatment or pesticide action.
