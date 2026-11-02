"""
PHYTOSIGNAL Field Validation & Generalization Audit Package
===========================================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.
"""

from .data_auditor import AuditReport, DataQualityAuditor
from .evaluator import FieldEvaluator, SamplePredictionRecord, preprocess_field_image
from .gate_evaluator import GatePredictionResult, GateSampleRecord, LeafGateEvaluator
from .manifest_schema import (
    CROP_TO_CLASSES,
    NAME_TO_CLASS_ID,
    TARGET_CLASS_MAPPING,
    FieldSampleRecord,
    ManifestValidationError,
    load_and_validate_manifest,
)
from .metrics import FieldEvaluationSummary, calculate_field_metrics

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
    "SamplePredictionRecord",
    "preprocess_field_image",
    "LeafGateEvaluator",
    "GateSampleRecord",
    "GatePredictionResult",
    "FieldEvaluationSummary",
    "calculate_field_metrics",
]
