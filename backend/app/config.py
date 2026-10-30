"""
Application configuration for the Crop Disease Detection backend.
"""
from pathlib import Path
import os
from typing import List, Set

# Base Paths (Resolved relative to project root, independent of working directory)
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model Configuration
DEFAULT_MODEL_REL_PATH = "ml/models/experiment_weighted/best_model.keras"
DEFAULT_MODEL_PATH = PROJECT_ROOT / DEFAULT_MODEL_REL_PATH
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Knowledge Base Configuration
DEFAULT_KNOWLEDGE_BASE_REL_PATH = "backend/app/data/disease_knowledge_base.json"
DEFAULT_KNOWLEDGE_BASE_PATH = PROJECT_ROOT / DEFAULT_KNOWLEDGE_BASE_REL_PATH
KNOWLEDGE_BASE_PATH = Path(os.getenv("KNOWLEDGE_BASE_PATH", str(DEFAULT_KNOWLEDGE_BASE_PATH)))

# Image Preprocessing & Decompression Safeguards
TARGET_IMAGE_SIZE = (224, 224)
TARGET_CHANNELS = 3

# Decompression & Pixel Limit Configuration
# Configurable maximum decoded image pixels (default: 20 Megapixels).
# Accommodates modern high-resolution leaf photos while preventing
# pixel-flood / decompression-bomb Denial-of-Service memory exhaustion.
DEFAULT_MAX_IMAGE_PIXELS = 20 * 1024 * 1024  # 20,971,520 pixels (~20 MP)
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS)))

# Upload Constraints (10 MB hard limit, chunked streaming reads)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE_BYTES = 64 * 1024  # 64 KB read buffer


# Helper functions for robust, fail-safe environment variable parsing
def _parse_bool_env(var_name: str, default: bool) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _parse_int_env(var_name: str, default: int, min_val: int = 1) -> int:
    val = os.getenv(var_name)
    if val is None:
        return default
    try:
        parsed = int(val.strip())
        return parsed if parsed >= min_val else default
    except ValueError:
        return default


def _parse_float_env(var_name: str, default: float, min_val: float = 0.0) -> float:
    val = os.getenv(var_name)
    if val is None:
        return default
    try:
        parsed = float(val.strip())
        return parsed if parsed >= min_val else default
    except ValueError:
        return default


# Level A Deterministic Physical-Integrity Validation Configuration
# Enable/disable deterministic Level A prevalidation checks (for safe rollback if needed)
ENABLE_LEVEL_A_PREVALIDATION: bool = _parse_bool_env("ENABLE_LEVEL_A_PREVALIDATION", True)

# Minimum acceptable image dimension in pixels (width >= 64 and height >= 64)
MIN_IMAGE_DIMENSION: int = _parse_int_env("MIN_IMAGE_DIMENSION", 64, min_val=1)

# Population standard deviation threshold on 224x224 luminance (sigma <= threshold rejects uniform/blank canvas)
UNIFORM_CANVAS_STD_THRESHOLD: float = _parse_float_env("UNIFORM_CANVAS_STD_THRESHOLD", 2.0, min_val=0.0)

# Content-Level Input Suitability Gate (Level B)
# Deterministic plant-tissue evidence and image-quality gate. This runs AFTER physical
# integrity checks and BEFORE model inference. It is a *necessary-condition* screen:
# it can establish that an image contains no plant pigmentation, but it does NOT verify
# that the subject is a supported Solanum/Capsicum leaf. Those limits are documented in
# docs/audit_reports/non_leaf_input_safety_report.md.
# Thresholds were calibrated on non-test data (see ml/scripts/evaluate_input_gate.py).
ENABLE_INPUT_GATE: bool = _parse_bool_env("ENABLE_INPUT_GATE", True)

# Broad plant-pigmentation mask: HSV hue 40-170 deg, saturation >= 0.12, value >= 0.08.
# Fraction of image pixels in this mask. Calibration (validation split, 3,418 leaves):
#   minimum 0.0606, 1st percentile 0.1488, 0 images below 0.02.
# Real non-leaf photographs (Background_without_leaves, 1,143 images): median 0.137,
#   with 15.8% below 0.01, 22.0% below 0.02 and 43.7% below 0.08.
INPUT_GATE_REJECT_BELOW_PLANT_EVIDENCE: float = _parse_float_env(
    "INPUT_GATE_REJECT_BELOW_PLANT_EVIDENCE", 0.01, min_val=0.0
)
INPUT_GATE_ACCEPT_AT_PLANT_EVIDENCE: float = _parse_float_env(
    "INPUT_GATE_ACCEPT_AT_PLANT_EVIDENCE", 0.08, min_val=0.0
)

# Dominant contiguous plant region and spread. A scene whose plant-coloured pixels are
# scattered across several small patches is treated as ambiguous context photography
# rather than a framed specimen.
INPUT_GATE_MIN_DOMINANT_REGION: float = _parse_float_env(
    "INPUT_GATE_MIN_DOMINANT_REGION", 0.02, min_val=0.0
)
INPUT_GATE_SCATTERED_REGION_COUNT: int = _parse_int_env(
    "INPUT_GATE_SCATTERED_REGION_COUNT", 3, min_val=1
)
INPUT_GATE_REGION_AREA_FLOOR: float = _parse_float_env(
    "INPUT_GATE_REGION_AREA_FLOOR", 0.01, min_val=0.0
)

# Exposure and detail floors for the "cannot assess" (abstain) path.
# Validation split minima: mean luminance 0.0561, Laplacian variance 0.0002.
INPUT_GATE_MIN_MEAN_LUMINANCE: float = _parse_float_env(
    "INPUT_GATE_MIN_MEAN_LUMINANCE", 0.05, min_val=0.0
)
INPUT_GATE_MIN_DETAIL_VARIANCE: float = _parse_float_env(
    "INPUT_GATE_MIN_DETAIL_VARIANCE", 0.0001, min_val=0.0
)

# Trained Leaf-Acceptance Gate (Level B content classifier)
# A 3-class model (supported_leaf / other_plant / non_leaf) built on frozen ImageNet
# MobileNetV2 features and trained by ml/scripts/train_input_gate.py. It is the content
# authority when it is loaded; the deterministic screen below is the fallback. Its
# validation-only operating point and measured error rates are recorded in
# docs/audit_reports/non_leaf_input_safety_report.md.
DEFAULT_LEAF_GATE_REL_PATH = "ml/models/input_gate/leaf_gate.keras"
DEFAULT_LEAF_GATE_PATH = PROJECT_ROOT / DEFAULT_LEAF_GATE_REL_PATH
LEAF_GATE_MODEL_PATH = Path(os.getenv("LEAF_GATE_MODEL_PATH", str(DEFAULT_LEAF_GATE_PATH)))
ENABLE_LEAF_GATE: bool = _parse_bool_env("ENABLE_LEAF_GATE", True)

# Minimum top-class probability required to act on a gate decision. Below it the gate
# abstains instead of asserting "leaf" or "not a leaf". Selected on the gate validation
# split subject to >= 99% valid-leaf coverage; never tuned on the evaluation split.
LEAF_GATE_MIN_CONFIDENCE: float = _parse_float_env("LEAF_GATE_MIN_CONFIDENCE", 0.60, min_val=0.0)

# Concurrency & Admission Control Configuration
# Bounded number of concurrent CPU-intensive image decodes to limit transient RAM usage
MAX_DECODE_CONCURRENCY: int = _parse_int_env("MAX_DECODE_CONCURRENCY", 2, min_val=1)

# Maximum queue size for waiting decode requests before overload rejection (HTTP 429)
MAX_DECODE_QUEUE_SIZE: int = _parse_int_env("MAX_DECODE_QUEUE_SIZE", 10, min_val=1)

# Format Validation
ALLOWED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
ALLOWED_PIL_FORMATS: Set[str] = {
    "JPEG",
    "PNG",
    "WEBP",
}

# CORS Configuration
_default_cors = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:8000,"
    "http://127.0.0.1:8000,"
    "http://localhost:5500,"
    "http://127.0.0.1:5500"
)
CORS_ORIGINS: List[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _default_cors).split(",")
    if origin.strip()
]

# Informational Disclaimer
DISCLAIMER_TEXT = (
    "This prediction is generated by an automated machine learning model for "
    "informational and educational purposes only. Softmax scores represent the "
    "model's relative output distribution across its 15 trained classes and should "
    "not be treated as calibrated probabilities or guaranteed diagnostic certainty. "
    "Always seek confirmation from a qualified agronomic expert or local agricultural "
    "extension service before undertaking crop treatment."
)

# API Metadata
API_TITLE = "AI-Based Crop Disease Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "FastAPI backend for Solanaceae crop disease classification using a "
    "balanced class-weighted MobileNetV2 deep learning model."
)
