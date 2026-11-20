"""
PHYTOSIGNAL Field Validation & Generalization Audit Package
===========================================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.
"""

from .data_auditor import AuditReport, DataQualityAuditor
from .evaluator import (
    FieldEvaluator,
    InferenceAccounting,
    ModelOutputValidationError,
    SamplePredictionRecord,
    preprocess_field_image,
    validate_model_output,
)
from .gate_evaluator import (
    GateIntegrityError,
    GatePredictionResult,
    GateSampleRecord,
    LeafGateEvaluator,
)
from .manifest_schema import (
    CROP_TO_CLASSES,
    NAME_TO_CLASS_ID,
    TARGET_CLASS_MAPPING,
    FieldSampleRecord,
    ManifestValidationError,
    load_and_validate_manifest,
)
from .metrics import FieldEvaluationSummary, calculate_field_metrics
from .path_resolver import (
    InvalidImageFileError,
    PathSecurityError,
    resolve_safe_image_path,
)

__all__ = [
    "TARGET_CLASS_MAPPING",
    "CROP_TO_CLASSES",
    "NAME_TO_CLASS_ID",
    "FieldSampleRecord",
    "ManifestValidationError",
    "load_and_validate_manifest",
    "DataQualityAuditor",
    "AuditReport",
    "FieldEvaluator",
    "ModelOutputValidationError",
    "validate_model_output",
    "InferenceAccounting",
    "SamplePredictionRecord",
    "preprocess_field_image",
    "LeafGateEvaluator",
    "GateIntegrityError",
    "GateSampleRecord",
    "GatePredictionResult",
    "FieldEvaluationSummary",
    "calculate_field_metrics",
    "resolve_safe_image_path",
    "PathSecurityError",
    "InvalidImageFileError",
]
