"""
Field Evaluation Pipeline - Standalone Model Evaluator
======================================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Runs deterministic, reproducible evaluation of the existing 15-class Solanaceae
disease classifier against realistic field imagery without modifying model weights
or backend code.

Key Guarantees:
1. Exact Preprocessing Parity: Applies EXIF transpose, RGB normalization, bilinear
   resizing to 224x224, and outputs float32 values in [0.0, 255.0] matching
   `backend/app/services/inference.py`.
2. Provenance & Reproducibility: Captures model file hash (SHA-256), class mapping,
   and evaluation timestamp. Never overwrites prior evaluation records.
3. Transparent Scoring: Emits uncalibrated raw softmax scores explicitly, never
   claiming they are calibrated Bayesian posterior probabilities.
4. Error Isolation: Preprocessing and corrupt image failures are isolated into
   separate error ledgers rather than silently skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from backend.app.config import (
    ALLOWED_PIL_FORMATS,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_DIMENSION,
    TARGET_IMAGE_SIZE,
    UNIFORM_CANVAS_STD_THRESHOLD,
)
from backend.app.services.inference import (
    ImageValidationError,
    preprocess_image,
)
from .manifest_schema import (
    NAME_TO_CLASS_ID,
    TARGET_CLASS_MAPPING,
    load_and_validate_manifest,
)
from .metrics import calculate_field_metrics
from .path_resolver import (
    InvalidImageFileError,
    PathSecurityError,
    resolve_safe_image_path,
)

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "2.1-authoritative-inference-parity"


class ModelOutputValidationError(ValueError):
    """Raised when model predictions violate mathematical or structural probability distribution bounds."""
    pass


def validate_model_output(
    scores: Any,
    expected_classes: int = 15,
    tolerance: float = 0.01,
) -> np.ndarray:
    """
    Validates that raw disease model output is a valid probability distribution (Phase 5).
    Requirements:
    - 1D shape (expected_classes,) or 2D (1, expected_classes)
    - Numeric floating-point dtype
    - Finite values (no NaN, +Inf, -Inf)
    - Valid probability values in [-tolerance, 1.0 + tolerance]
    - Sum is approximately 1.0 within tolerance (1e-2)
    Raises ModelOutputValidationError on failure.
    """
    if scores is None:
        raise ModelOutputValidationError("Model output is None.")

    try:
        arr = np.asarray(scores)
    except Exception as exc:
        raise ModelOutputValidationError(f"Cannot convert model output to numpy array: {exc}") from exc

    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        raise ModelOutputValidationError(f"Model output dtype must be numeric, got {arr.dtype}.")

    arr = arr.astype(np.float64)

    # Check shape: allow (1, expected_classes) or (expected_classes,)
    if arr.ndim == 2:
        if arr.shape[0] != 1 or arr.shape[1] != expected_classes:
            raise ModelOutputValidationError(
                f"Model output shape {arr.shape} invalid. Expected (1, {expected_classes}) or ({expected_classes},)."
            )
        arr = arr[0]
    elif arr.ndim == 1:
        if arr.shape[0] != expected_classes:
            raise ModelOutputValidationError(
                f"Model output shape {arr.shape} invalid. Expected ({expected_classes},)."
            )
    else:
        raise ModelOutputValidationError(
            f"Model output has invalid dimension {arr.ndim}. Expected 1D or 2D batch of size 1."
        )

    # Check finite
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        raise ModelOutputValidationError(
            f"Model output contains non-finite values (NaN: {nan_count}, Inf: {inf_count})."
        )

    # Check probability range [-tolerance, 1.0 + tolerance]
    if np.any(arr < -tolerance) or np.any(arr > 1.0 + tolerance):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        raise ModelOutputValidationError(
            f"Model output probabilities out of bounds [0.0, 1.0]: min={min_val}, max={max_val}."
        )

    # Check sum
    prob_sum = float(np.sum(arr))
    if abs(prob_sum - 1.0) > tolerance:
        raise ModelOutputValidationError(
            f"Model output probabilities must sum to 1.0 within tolerance {tolerance}: sum={prob_sum}."
        )

    return arr


@dataclass
class InferenceAccounting:
    total_records: int
    valid_records: int
    inference_attempts: int
    successful_inferences: int
    failed_inferences: int
    failure_rate: float
    failure_adjusted_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TopPrediction:
    class_id: int
    class_name: str
    crop: str
    condition: str
    health: str
    raw_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "crop": self.crop,
            "condition": self.condition,
            "health": self.health,
            "raw_score": round(self.raw_score, 6),
        }


@dataclass
class SamplePredictionRecord:
    image_id: str
    file_path: str
    split: str
    crop: str
    ground_truth_class: str
    ground_truth_id: int
    predicted_class: str
    predicted_id: int
    is_correct: bool
    top_3_predictions: List[Dict[str, Any]]
    raw_scores: Dict[str, float]
    # Stratification metadata
    lighting_condition: str
    background_type: str
    disease_severity: str
    capture_device: str
    annotator_status: str
    verification_status: str
    duplicate_group_id: str
    notes: str
    # Provenance
    model_sha256: str
    preprocessing_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "split": self.split,
            "crop": self.crop,
            "ground_truth_class": self.ground_truth_class,
            "ground_truth_id": self.ground_truth_id,
            "predicted_class": self.predicted_class,
            "predicted_id": self.predicted_id,
            "is_correct": self.is_correct,
            "top_3_predictions": self.top_3_predictions,
            "raw_scores": {k: round(v, 6) for k, v in self.raw_scores.items()},
            "lighting_condition": self.lighting_condition,
            "background_type": self.background_type,
            "disease_severity": self.disease_severity,
            "capture_device": self.capture_device,
            "annotator_status": self.annotator_status,
            "verification_status": self.verification_status,
            "duplicate_group_id": self.duplicate_group_id,
            "notes": self.notes,
            "model_sha256": self.model_sha256,
            "preprocessing_version": self.preprocessing_version,
        }


@dataclass
class EvaluationErrorRecord:
    image_id: str
    file_path: str
    error_type: str
    error_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for audit provenance."""
    sha = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def preprocess_field_image(
    image_input: Union[Path, str, bytes, Image.Image],
    enforce_level_a: bool = True,
) -> np.ndarray:
    """
    Decodes an image, handles EXIF orientation, converts to RGB,
    resizes to 224x224, and outputs float32 values in [0.0, 255.0].

    Delegates directly to `backend.app.services.inference.preprocess_image` to
    guarantee 100% exact parity with production preprocessing, security limits,
    decompression-bomb safeguards, and validation thresholds.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        raw_bytes = p.read_bytes()
    elif isinstance(image_input, bytes):
        raw_bytes = image_input
    elif isinstance(image_input, Image.Image):
        import io
        buf = io.BytesIO()
        fmt = image_input.format or "PNG"
        image_input.save(buf, format=fmt)
        raw_bytes = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    return preprocess_image(raw_bytes, enforce_level_a=enforce_level_a)


class FieldEvaluator:
    """
    Evaluates the existing MobileNetV2 disease classifier against field manifests.
    """

    def __init__(self, model_path: Union[Path, str]):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        self.model_sha256 = compute_file_sha256(self.model_path)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads and verifies the Keras model checkpoint."""
        import keras

        logger.info("Loading model checkpoint from: %s", self.model_path)
        self._model = keras.models.load_model(str(self.model_path), compile=False)

        # Verify output dimension
        output_shape = getattr(self._model, "output_shape", None)
        if output_shape and output_shape[-1] != len(TARGET_CLASS_MAPPING):
            raise ValueError(
                f"Model output dimension ({output_shape[-1]}) does not match expected {len(TARGET_CLASS_MAPPING)} classes."
            )
        logger.info(
            "Model verified successfully. Checkpoint SHA-256: %s...",
            self.model_sha256[:16],
        )

    def predict_single(self, preprocessed_frame: np.ndarray) -> Tuple[str, int, List[TopPrediction], Dict[str, float]]:
        """
        Runs inference on a single preprocessed frame of shape (1, 224, 224, 3).
        Validates model output probability bounds (Phase 5).
        Returns:
            (top_1_class_name, top_1_class_id, top_3_candidates, raw_softmax_scores_dict)
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        raw_output = self._model.predict(preprocessed_frame, verbose=0)
        scores = validate_model_output(raw_output, expected_classes=len(TARGET_CLASS_MAPPING), tolerance=0.01)

        # Map to class IDs
        # id to class name
        id_to_name = {v["id"]: k for k, v in TARGET_CLASS_MAPPING.items()}
        raw_scores_dict = {id_to_name[i]: float(scores[i]) for i in range(len(scores))}

        sorted_indices = np.argsort(scores)[::-1]
        top_1_id = int(sorted_indices[0])
        top_1_name = id_to_name[top_1_id]

        top_3: List[TopPrediction] = []
        for rank in range(min(3, len(sorted_indices))):
            idx = int(sorted_indices[rank])
            name = id_to_name[idx]
            meta = TARGET_CLASS_MAPPING[name]
            top_3.append(
                TopPrediction(
                    class_id=idx,
                    class_name=name,
                    crop=meta["crop"],
                    condition=meta["condition"],
                    health=meta["health"],
                    raw_score=float(scores[idx]),
                )
            )

        return top_1_name, top_1_id, top_3, raw_scores_dict

    def evaluate_manifest(
        self,
        manifest_path: Union[Path, str],
        dataset_root: Optional[Union[Path, str]] = None,
        split_filter: Optional[str] = None,
        enforce_level_a: bool = True,
        exclude_unverified_from_primary: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation on the records within a field manifest.
        Separates disease classification from gate-only tasks (Bug 5).
        Excludes visually suspected / ambiguous samples from primary metrics by default (Bug 4).
        Enforces safe image path resolution (Phase 2) and model output validation (Phase 5).
        Tracks full failed inference accounting (Phase 6.4).
        """
        manifest_path = Path(manifest_path).resolve()
        if dataset_root is None:
            dataset_root = manifest_path.parent
        else:
            dataset_root = Path(dataset_root).resolve()

        records, validation_errors = load_and_validate_manifest(manifest_path)
        if validation_errors:
            logger.warning("Manifest schema validation returned %d issues.", len(validation_errors))

        # Filter by split if requested
        if split_filter:
            records = [r for r in records if r.split == split_filter]

        # Separate disease-classification tasks from gate-only tasks (Bug 5)
        disease_records = [r for r in records if r.evaluation_task in ("disease", "both")]
        gate_only_records = [r for r in records if r.evaluation_task == "gate"]

        logger.info(
            "Manifest records: %d total (%d disease-applicable, %d gate-only bypassed)",
            len(records),
            len(disease_records),
            len(gate_only_records),
        )

        evaluation_records: List[SamplePredictionRecord] = []
        error_records: List[EvaluationErrorRecord] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for idx, rec in enumerate(disease_records, start=1):
            # Resolve image file path safely with directory traversal protection (Phase 2)
            try:
                img_path = resolve_safe_image_path(rec.file_path, dataset_root=dataset_root)
            except Exception as exc:
                logger.error("Path resolution failed for image_id %s: %s", rec.image_id, exc)
                error_records.append(
                    EvaluationErrorRecord(
                        image_id=rec.image_id,
                        file_path=rec.file_path,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            try:
                frame = preprocess_field_image(img_path, enforce_level_a=enforce_level_a)
                pred_name, pred_id, top_3, raw_scores = self.predict_single(frame)

                is_correct = pred_name == rec.ground_truth_class
                gt_id = NAME_TO_CLASS_ID[rec.ground_truth_class]

                eval_rec = SamplePredictionRecord(
                    image_id=rec.image_id,
                    file_path=str(img_path),
                    split=rec.split,
                    crop=rec.crop,
                    ground_truth_class=rec.ground_truth_class,
                    ground_truth_id=gt_id,
                    predicted_class=pred_name,
                    predicted_id=pred_id,
                    is_correct=is_correct,
                    top_3_predictions=[cand.to_dict() for cand in top_3],
                    raw_scores=raw_scores,
                    lighting_condition=rec.lighting_condition,
                    background_type=rec.background_type,
                    disease_severity=rec.disease_severity,
                    capture_device=rec.capture_device,
                    annotator_status=rec.annotator_status,
                    verification_status=rec.verification_status,
                    duplicate_group_id=rec.duplicate_group_id,
                    notes=rec.notes,
                    model_sha256=self.model_sha256,
                    preprocessing_version=PREPROCESSING_VERSION,
                )
                evaluation_records.append(eval_rec)

            except Exception as exc:
                logger.error("Inference failed for image_id %s: %s", rec.image_id, exc)
                error_records.append(
                    EvaluationErrorRecord(
                        image_id=rec.image_id,
                        file_path=str(img_path),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )

        total_manifest_records = len(records)
        disease_task_records = len(disease_records)
        successful_inferences = len(evaluation_records)
        failed_inferences = len(error_records)
        inference_attempts = disease_task_records
        failure_rate = (failed_inferences / inference_attempts) if inference_attempts > 0 else 0.0
        successful_correct = sum(1 for r in evaluation_records if r.is_correct)
        failure_adjusted_accuracy = (successful_correct / inference_attempts) if inference_attempts > 0 else 0.0

        accounting = InferenceAccounting(
            total_records=total_manifest_records,
            valid_records=disease_task_records,
            inference_attempts=inference_attempts,
            successful_inferences=successful_inferences,
            failed_inferences=failed_inferences,
            failure_rate=round(failure_rate, 4),
            failure_adjusted_accuracy=round(failure_adjusted_accuracy, 4),
        )

        formal_passed = (len(error_records) == 0 and len(validation_errors) == 0)

        # Convert to dictionaries for metrics calculation
        eval_dicts = [r.to_dict() for r in evaluation_records]

        # Calculate primary verified metrics (Bug 4)
        summary = calculate_field_metrics(
            eval_dicts,
            exclude_unverified=exclude_unverified_from_primary,
            inference_accounting=accounting.to_dict(),
        )

        # Also calculate exploratory metrics on full evaluated set (Bug 4)
        exploratory_summary = calculate_field_metrics(
            eval_dicts,
            exclude_unverified=False,
            inference_accounting=accounting.to_dict(),
        )

        run_output = {
            "metadata": {
                "evaluation_timestamp": timestamp,
                "manifest_path": str(manifest_path),
                "dataset_root": str(dataset_root),
                "split_filter": split_filter,
                "model_path": str(self.model_path),
                "model_sha256": self.model_sha256,
                "preprocessing_version": PREPROCESSING_VERSION,
                "total_manifest_records": total_manifest_records,
                "disease_task_records": disease_task_records,
                "gate_only_records_bypassed": len(gate_only_records),
                "successfully_evaluated": successful_inferences,
                "failed_inferences": failed_inferences,
                "inference_error_rate": round(failure_rate, 4),
                "failure_adjusted_accuracy": round(failure_adjusted_accuracy, 4),
                "formal_evaluation_passed": formal_passed,
                "evaluation_status": "PASSED" if formal_passed else "FAILED_INTEGRITY_VIOLATION",
                "exclude_unverified_from_primary": exclude_unverified_from_primary,
                "inference_accounting": accounting.to_dict(),
            },
            "metrics": summary.to_dict(),
            "exploratory_metrics": exploratory_summary.to_dict(),
            "errors": [e.to_dict() for e in error_records],
            "predictions": eval_dicts,
        }

        return run_output

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[Path, str],
        prefix: str = "field_eval",
    ) -> Tuple[Path, Path]:
        """
        Saves evaluation run output to JSON and a summary Markdown file.
        Guarantees that prior evaluation runs are never overwritten.
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_file = out_path / f"{prefix}_{timestamp_str}.json"
        md_file = out_path / f"{prefix}_{timestamp_str}_summary.md"

        # Save JSON
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation JSON saved to: %s", json_file)

        # Save Markdown summary
        md_content = self.generate_markdown_summary(results)
        with md_file.open("w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Evaluation summary saved to: %s", md_file)

        return json_file, md_file

    @staticmethod
    def generate_markdown_summary(results: Dict[str, Any]) -> str:
        """Renders an executive Markdown summary of the evaluation results."""
        meta = results["metadata"]
        metrics = results["metrics"]

        md = []
        md.append("# Field Validation Evaluation Summary")
        md.append(f"**Evaluation Timestamp:** {meta['evaluation_timestamp']}")
        md.append(f"**Model Checkpoint SHA-256:** `{meta['model_sha256']}`")
        md.append(f"**Preprocessing Version:** `{meta['preprocessing_version']}`")
        md.append(f"**Manifest Evaluated:** `{meta['manifest_path']}`")
        md.append(f"**Total Records:** {meta['total_manifest_records']}")
        md.append(f"**Successfully Evaluated:** {meta['successfully_evaluated']}")
        md.append(f"**Inference Errors:** {meta['failed_inferences']} ({meta['inference_error_rate'] * 100:.2f}%)")
        md.append("")

        md.append("## Overall Metrics (95% Confidence Intervals)")
        md.append(f"- **Top-1 Accuracy:** {metrics['accuracy'] * 100:.2f}% (95% CI: [{metrics['accuracy_ci_95'][0] * 100:.2f}%, {metrics['accuracy_ci_95'][1] * 100:.2f}%])")
        md.append(f"- **Top-3 Accuracy:** {metrics['top_3_accuracy'] * 100:.2f}% (95% CI: [{metrics['top_3_ci_95'][0] * 100:.2f}%, {metrics['top_3_ci_95'][1] * 100:.2f}%])")
        if metrics.get("macro_f1_ci_95") is not None:
            f1_ci_str = f"[{metrics['macro_f1_ci_95'][0]:.4f}, {metrics['macro_f1_ci_95'][1]:.4f}]"
        else:
            reason = metrics.get("macro_f1_ci_reason", "insufficient clusters")
            f1_ci_str = f"N/A ({reason})"
        md.append(f"- **Macro-F1 Score (Primary/Observed):** {metrics['macro_f1']:.4f} (95% CI: {f1_ci_str})")
        if "fixed_macro_f1" in metrics:
            md.append(f"- **Fixed Macro-F1 (All 15 classes):** {metrics['fixed_macro_f1']:.4f}")
        md.append(f"- **Macro Precision:** {metrics['macro_precision']:.4f}")
        md.append(f"- **Macro Recall:** {metrics['macro_recall']:.4f}")
        md.append("")

        md.append("## Per-Crop Breakdown")
        md.append("| Crop | Support | Accuracy (%) | Macro-F1 | Precision | Recall |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for crop, cmetrics in metrics.get("per_crop", {}).items():
            md.append(
                f"| {crop} | {cmetrics['support']} | {cmetrics['accuracy'] * 100:.2f}% | "
                f"{cmetrics['macro_f1']:.4f} | {cmetrics['precision']:.4f} | {cmetrics['recall']:.4f} |"
            )
        md.append("")

        md.append("## Per-Class Performance")
        md.append("| Class | Support | Precision | Recall | F1-Score | Status |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
        for cname, cmetrics in metrics.get("per_class", {}).items():
            status = "Low Support" if cmetrics["low_support_flag"] else "OK"
            md.append(
                f"| `{cname}` | {cmetrics['support']} | {cmetrics['precision']:.4f} | "
                f"{cmetrics['recall']:.4f} | {cmetrics['f1']:.4f} | {status} |"
            )
        md.append("")

        if "subgroup_breakdowns" in metrics:
            md.append("## Subgroup Performance Slices")
            for sub_name, sub_data in metrics["subgroup_breakdowns"].items():
                md.append(f"### {sub_name.replace('_', ' ').title()}")
                md.append("| Group | Support | Accuracy (%) | Note |")
                md.append("| :--- | :---: | :---: | :--- |")
                for grp, gmetrics in sub_data.items():
                    warn = "[LOW SAMPLE < 10]" if gmetrics.get("low_sample_warning") else ""
                    md.append(f"| {grp} | {gmetrics['support']} | {gmetrics['accuracy'] * 100:.2f}% | {warn} |")
                md.append("")

        return "\n".join(md)
