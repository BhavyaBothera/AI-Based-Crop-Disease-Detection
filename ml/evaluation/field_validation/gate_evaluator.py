"""
Leaf Gate Field Evaluation Engine
=================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Evaluates the leaf-acceptance screening gate (Level A + Level B `leaf_gate.keras`)
against real-world conditions without modifying gate thresholds or backend code.

Quantifies:
1. False Rejection Rate (FRR): Legitimate supported crop leaves rejected or abstained on.
2. False Acceptance Rate (FAR): Non-leaf objects or other plant species erroneously accepted.
3. Challenge Condition Robustness:
   - Background-only images
   - Foliage background clutter
   - Partial / cropped leaves
   - Multiple leaves in frame
   - Out-of-focus or motion-blurred captures
4. End-to-End Pipeline Coupling:
   - Measures disease classifier behavior on images passed through the gate.
   - Audits whether spurious high-confidence diagnoses occur on OOD inputs if the gate fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.app.config import LEAF_GATE_MODEL_PATH
from backend.app.services.input_gate import (
    ABSTAINED,
    ACCEPTED,
    REJECTED,
    assess_image,
)
from backend.app.services.leaf_gate import LeafGateService
from .evaluator import compute_file_sha256, preprocess_field_image
from .manifest_schema import (
    GATE_CATEGORY_NON_LEAF,
    GATE_CATEGORY_OTHER_PLANT,
    GATE_CATEGORY_SUPPORTED_LEAF,
    UNVERIFIED_TIERS,
    VALID_GATE_CATEGORIES,
    VERIFIED_TIERS,
)
from .path_resolver import PathSecurityError, resolve_safe_image_path

logger = logging.getLogger(__name__)

REQUIRED_SIDECAR_FIELDS = {"classes", "feature_backbone", "input"}
EXPECTED_GATE_CLASSES = [GATE_CATEGORY_NON_LEAF, GATE_CATEGORY_OTHER_PLANT, GATE_CATEGORY_SUPPORTED_LEAF]


class GateIntegrityError(RuntimeError):
    """Raised when gate model, sidecar config, or output dimensionality fails integrity checks."""
    pass


@dataclass
class GateSampleRecord:
    sample_id: str
    file_path: str
    ground_truth_category: str  # 'supported_leaf', 'other_plant', 'non_leaf'
    challenge_type: str = "standard"  # 'foliage_clutter', 'partial_leaf', 'multi_leaf', 'blur', 'background_only', 'standard'
    crop: Optional[str] = None
    verification_status: str = "expert_verified"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GatePredictionResult:
    sample_id: str
    file_path: str
    ground_truth_category: str
    challenge_type: str
    decision: str  # 'accepted', 'abstained', 'rejected'
    code: str
    category_emitted: str
    gate_probabilities: Dict[str, float]
    gate_confidence: float
    gate_label: str
    is_correctly_handled: bool
    is_false_rejection: bool
    is_false_acceptance: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "file_path": self.file_path,
            "ground_truth_category": self.ground_truth_category,
            "challenge_type": self.challenge_type,
            "decision": self.decision,
            "code": self.code,
            "category_emitted": self.category_emitted,
            "gate_probabilities": {k: round(v, 4) for k, v in self.gate_probabilities.items()},
            "gate_confidence": round(self.gate_confidence, 4),
            "gate_label": self.gate_label,
            "is_correctly_handled": self.is_correctly_handled,
            "is_false_rejection": self.is_false_rejection,
            "is_false_acceptance": self.is_false_acceptance,
        }


class LeafGateEvaluator:
    """
    Evaluator for input screening and leaf acceptance gate under field conditions.
    """

    def __init__(
        self,
        gate_model_path: Optional[Union[Path, str]] = None,
        leaf_gate_service: Optional[LeafGateService] = None,
    ):
        self.fallback_reason: Optional[str] = None
        if leaf_gate_service is not None:
            self.service = leaf_gate_service
            self.model_path = getattr(leaf_gate_service, "_model_path", LEAF_GATE_MODEL_PATH)
            if not getattr(leaf_gate_service, "is_loaded", False):
                self.fallback_reason = getattr(leaf_gate_service, "load_error", "Mock or uninitialized service")
        else:
            model_p = Path(gate_model_path or LEAF_GATE_MODEL_PATH).resolve()
            self.model_path = model_p
            self.service = LeafGateService(model_path=model_p)
            if model_p.is_file():
                try:
                    self.service.load_model(model_p)
                except Exception as exc:
                    self.fallback_reason = f"Failed to load gate model at {model_p}: {exc}"
                    logger.warning("%s Using deterministic fallback.", self.fallback_reason)
            else:
                self.fallback_reason = f"Gate model file not found at {model_p}."
                logger.warning("%s Using deterministic fallback.", self.fallback_reason)

        self.is_trained_model = bool(getattr(self.service, "is_loaded", False))
        self.gate_backend = "trained_model" if self.is_trained_model else "deterministic_fallback"
        self.evaluation_type = "trained_model_evaluation" if self.is_trained_model else "fallback_evaluation"
        self.model_sha256 = (
            compute_file_sha256(self.model_path) if (self.is_trained_model and self.model_path.is_file()) else "N/A"
        )
        self.sidecar_validation: Dict[str, Any] = self._validate_sidecar_and_model()

    def _validate_sidecar_and_model(self) -> Dict[str, Any]:
        """
        Validates sidecar class mappings, required schema fields, and model output dimensions.
        Fails closed by raising GateIntegrityError if sidecar exists but is invalid or corrupted.
        """
        service_classes = list(getattr(self.service, "classes", EXPECTED_GATE_CLASSES))
        validation_info: Dict[str, Any] = {
            "sidecar_present": False,
            "classes": service_classes,
            "output_dim": len(service_classes),
            "is_valid": True,
            "errors": [],
        }
        if not self.is_trained_model:
            validation_info["notice"] = f"Deterministic fallback active: {self.fallback_reason or 'no trained model loaded'}."
            return validation_info

        sidecar_path = self.model_path.parent / "gate_model_config.json"
        if sidecar_path.is_file():
            validation_info["sidecar_present"] = True
            try:
                cfg = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception as exc:
                err = f"Failed to parse sidecar config at {sidecar_path}: {exc}"
                validation_info["errors"].append(err)
                validation_info["is_valid"] = False
                raise GateIntegrityError(err) from exc

            if not isinstance(cfg, dict):
                err = f"Sidecar config at {sidecar_path} must be a JSON object, got {type(cfg).__name__}."
                validation_info["errors"].append(err)
                validation_info["is_valid"] = False
                raise GateIntegrityError(err)

            missing_fields = REQUIRED_SIDECAR_FIELDS - set(cfg.keys())
            if missing_fields:
                err = f"Sidecar config at {sidecar_path} missing required fields: {sorted(missing_fields)}."
                validation_info["errors"].append(err)
                validation_info["is_valid"] = False
                raise GateIntegrityError(err)

            sidecar_classes = cfg.get("classes", [])
            if sidecar_classes != EXPECTED_GATE_CLASSES:
                err = (
                    f"Sidecar classes {sidecar_classes} do not match expected {EXPECTED_GATE_CLASSES} "
                    f"(exact ordering required)."
                )
                validation_info["errors"].append(err)
                validation_info["is_valid"] = False
                raise GateIntegrityError(err)

            validation_info["sidecar_classes"] = sidecar_classes
            validation_info["feature_backbone"] = cfg.get("feature_backbone")
            validation_info["input"] = cfg.get("input")
        else:
            if self.model_path.is_file():
                err = f"Trained gate model file exists at {self.model_path} but sidecar config is missing at {sidecar_path}."
                validation_info["errors"].append(err)
                validation_info["is_valid"] = False
                raise GateIntegrityError(err)
            validation_info["notice"] = "No sidecar config found; using default classes."

        if getattr(self.service, "_model", None) is not None:
            try:
                output_shape = getattr(self.service._model, "output_shape", None)
                if output_shape is not None:
                    validation_info["model_output_shape"] = list(output_shape)
                    if output_shape[-1] != len(EXPECTED_GATE_CLASSES):
                        err = f"Model output shape {output_shape} last dim != num expected classes {len(EXPECTED_GATE_CLASSES)}."
                        validation_info["errors"].append(err)
                        validation_info["is_valid"] = False
                        raise GateIntegrityError(err)
            except GateIntegrityError:
                raise
            except Exception as exc:
                logger.debug("Could not inspect model.output_shape: %s", exc)

        return validation_info

    def evaluate_sample(
        self,
        sample: GateSampleRecord,
        dataset_root: Optional[Path] = None,
    ) -> GatePredictionResult:
        """
        Runs input gate assessment on a single image sample.
        """
        gt = sample.ground_truth_category
        if gt not in VALID_GATE_CATEGORIES:
            raise ValueError(
                f"Sample '{sample.sample_id}' has invalid ground_truth_category '{gt}'. "
                f"Must be one of {sorted(VALID_GATE_CATEGORIES)}"
            )

        img_p = resolve_safe_image_path(
            sample.file_path,
            dataset_root=dataset_root,
            allow_absolute=(dataset_root is None),
        )

        # Decodes image to raw 224x224 RGB array in [0.0, 255.0]
        frame_4d = preprocess_field_image(img_p, enforce_level_a=False)
        frame_2d = frame_4d[0]  # shape (224, 224, 3)

        # Gate probabilities
        gate_probs = self.service.predict_proba(frame_2d)

        # Full input gate assessment (Level A + Level B)
        assessment = assess_image(frame_2d, gate_probabilities=gate_probs)

        decision = assessment.decision  # 'accepted', 'abstained', 'rejected'
        code = assessment.code
        category_emitted = assessment.category

        probs = assessment.signals.get("gate_probabilities", gate_probs or {})
        conf = assessment.signals.get("gate_confidence", 0.0)
        label = assessment.signals.get("gate_label", "unknown")

        # Define correctness and error flags
        is_false_rejection = False
        is_false_acceptance = False
        is_correctly_handled = False

        if gt == "supported_leaf":
            # Supported leaves SHOULD be accepted
            if decision == ACCEPTED:
                is_correctly_handled = True
            else:
                is_false_rejection = True
        elif gt in ("other_plant", "non_leaf"):
            # Unsupported inputs SHOULD NOT be accepted
            if decision in (REJECTED, ABSTAINED):
                is_correctly_handled = True
            else:
                is_false_acceptance = True
        else:
            # Ambiguous/challenge cases
            is_correctly_handled = decision in (ACCEPTED, ABSTAINED)

        return GatePredictionResult(
            sample_id=sample.sample_id,
            file_path=str(img_p),
            ground_truth_category=gt,
            challenge_type=sample.challenge_type,
            decision=decision,
            code=code,
            category_emitted=category_emitted,
            gate_probabilities=probs,
            gate_confidence=conf,
            gate_label=label,
            is_correctly_handled=is_correctly_handled,
            is_false_rejection=is_false_rejection,
            is_false_acceptance=is_false_acceptance,
        )

    def evaluate_dataset(
        self,
        samples: List[GateSampleRecord],
        dataset_root: Optional[Union[Path, str]] = None,
        exclude_unverified: bool = True,
        allow_exploratory_subset: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a set of gate sample records and returns aggregated metrics.
        Separates verified samples from unverified/suspected annotations (Phase 7).
        Calculates valid rates with support counts; denominators with 0 support return None (not 0.0).
        Tracks full failed inference accounting and failure-adjusted rates (Phase 2).
        Fails closed on security/path-integrity violations and reports ordinary per-image errors.
        """
        root_path = Path(dataset_root).resolve() if dataset_root else None

        # Partition samples by verification status
        verified_samples = [s for s in samples if getattr(s, "verification_status", "expert_verified") in VERIFIED_TIERS]
        unverified_samples = [s for s in samples if getattr(s, "verification_status", "expert_verified") not in VERIFIED_TIERS]

        unverified_breakdown: Dict[str, int] = {}
        for s in unverified_samples:
            st = getattr(s, "verification_status", "unknown")
            unverified_breakdown[st] = unverified_breakdown.get(st, 0) + 1

        def _evaluate_batch(batch: List[GateSampleRecord]) -> Tuple[List[GatePredictionResult], List[Dict[str, Any]], Dict[str, Any]]:
            batch_results: List[GatePredictionResult] = []
            batch_errors: List[Dict[str, Any]] = []

            for sample in batch:
                try:
                    res = self.evaluate_sample(sample, dataset_root=root_path)
                    batch_results.append(res)
                except (PathSecurityError, GateIntegrityError) as sec_exc:
                    # Security and structural integrity violations MUST abort evaluation immediately (Phase 2 P0)
                    logger.critical("Security / integrity violation on gate sample '%s': %s", sample.sample_id, sec_exc)
                    raise
                except Exception as exc:
                    # Ordinary per-image processing failure (e.g. corrupt bytes, unreadable format, missing file)
                    logger.warning("Per-image processing failure on gate sample '%s': %s", sample.sample_id, exc)
                    batch_errors.append({
                        "sample_id": sample.sample_id,
                        "file_path": sample.file_path,
                        "ground_truth_category": sample.ground_truth_category,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })

            # Submissions per category (for failure-adjusted denominators)
            n_supp_submitted = sum(1 for s in batch if s.ground_truth_category == "supported_leaf")
            n_other_submitted = sum(1 for s in batch if s.ground_truth_category == "other_plant")
            n_non_leaf_submitted = sum(1 for s in batch if s.ground_truth_category == "non_leaf")

            # Errors per category
            n_supp_failed = sum(1 for e in batch_errors if e.get("ground_truth_category") == "supported_leaf")
            n_other_failed = sum(1 for e in batch_errors if e.get("ground_truth_category") == "other_plant")
            n_non_leaf_failed = sum(1 for e in batch_errors if e.get("ground_truth_category") == "non_leaf")

            supported_samples = [r for r in batch_results if r.ground_truth_category == "supported_leaf"]
            other_plant_samples = [r for r in batch_results if r.ground_truth_category == "other_plant"]
            non_leaf_samples = [r for r in batch_results if r.ground_truth_category == "non_leaf"]

            # Supported leaf metrics
            n_supp = len(supported_samples)
            supp_accepted = sum(1 for r in supported_samples if r.decision == ACCEPTED)
            supp_abstained = sum(1 for r in supported_samples if r.decision == ABSTAINED)
            supp_rejected = sum(1 for r in supported_samples if r.decision == REJECTED)
            acceptance_rate = round(supp_accepted / n_supp, 4) if n_supp > 0 else None
            frr = round((supp_abstained + supp_rejected) / n_supp, 4) if n_supp > 0 else None
            strict_frr = round(supp_rejected / n_supp, 4) if n_supp > 0 else None
            failure_adjusted_acceptance_rate = round(supp_accepted / n_supp_submitted, 4) if n_supp_submitted > 0 else None
            failure_adjusted_frr = round((supp_abstained + supp_rejected + n_supp_failed) / n_supp_submitted, 4) if n_supp_submitted > 0 else None

            # Other plant metrics
            n_other = len(other_plant_samples)
            other_accepted = sum(1 for r in other_plant_samples if r.decision == ACCEPTED)
            other_abstained = sum(1 for r in other_plant_samples if r.decision == ABSTAINED)
            other_rejected = sum(1 for r in other_plant_samples if r.decision == REJECTED)
            other_rejection_rate = round(other_rejected / n_other, 4) if n_other > 0 else None
            other_far = round(other_accepted / n_other, 4) if n_other > 0 else None
            failure_adjusted_other_rejection_rate = round(other_rejected / n_other_submitted, 4) if n_other_submitted > 0 else None
            failure_adjusted_other_far = round(other_accepted / n_other_submitted, 4) if n_other_submitted > 0 else None

            # Non-leaf metrics
            n_non_leaf = len(non_leaf_samples)
            non_leaf_accepted = sum(1 for r in non_leaf_samples if r.decision == ACCEPTED)
            non_leaf_abstained = sum(1 for r in non_leaf_samples if r.decision == ABSTAINED)
            non_leaf_rejected = sum(1 for r in non_leaf_samples if r.decision == REJECTED)
            non_leaf_rejection_rate = round(non_leaf_rejected / n_non_leaf, 4) if n_non_leaf > 0 else None
            non_leaf_far = round(non_leaf_accepted / n_non_leaf, 4) if n_non_leaf > 0 else None
            failure_adjusted_non_leaf_rejection_rate = round(non_leaf_rejected / n_non_leaf_submitted, 4) if n_non_leaf_submitted > 0 else None
            failure_adjusted_non_leaf_far = round(non_leaf_accepted / n_non_leaf_submitted, 4) if n_non_leaf_submitted > 0 else None

            # Combined OOD (other_plant + non_leaf)
            n_ood_submitted = n_other_submitted + n_non_leaf_submitted
            n_ood = n_other + n_non_leaf
            total_ood_accepted = other_accepted + non_leaf_accepted
            combined_far = round(total_ood_accepted / n_ood, 4) if n_ood > 0 else None
            failure_adjusted_combined_far = round(total_ood_accepted / n_ood_submitted, 4) if n_ood_submitted > 0 else None

            # Breakdown by challenge condition
            challenges: Dict[str, Dict[str, Any]] = {}
            for r in batch_results:
                c = r.challenge_type
                if c not in challenges:
                    challenges[c] = {"total": 0, "accepted": 0, "abstained": 0, "rejected": 0}
                challenges[c]["total"] += 1
                challenges[c][r.decision] += 1

            for c, counts in challenges.items():
                tot = counts["total"]
                counts["acceptance_rate"] = round(counts["accepted"] / tot, 4) if tot else None

            metrics_dict = {
                "supported_leaves": {
                    "submitted": n_supp_submitted,
                    "support": n_supp,
                    "failed": n_supp_failed,
                    "accepted": supp_accepted,
                    "abstained": supp_abstained,
                    "rejected": supp_rejected,
                    "acceptance_rate": acceptance_rate,
                    "false_rejection_rate": frr,
                    "strict_false_rejection_rate": strict_frr,
                    "failure_adjusted_acceptance_rate": failure_adjusted_acceptance_rate,
                    "failure_adjusted_false_rejection_rate": failure_adjusted_frr,
                },
                "other_plant_species": {
                    "submitted": n_other_submitted,
                    "support": n_other,
                    "failed": n_other_failed,
                    "accepted": other_accepted,
                    "abstained": other_abstained,
                    "rejected": other_rejected,
                    "rejection_rate": other_rejection_rate,
                    "false_acceptance_rate": other_far,
                    "failure_adjusted_rejection_rate": failure_adjusted_other_rejection_rate,
                    "failure_adjusted_false_acceptance_rate": failure_adjusted_other_far,
                },
                "non_leaf_objects": {
                    "submitted": n_non_leaf_submitted,
                    "support": n_non_leaf,
                    "failed": n_non_leaf_failed,
                    "accepted": non_leaf_accepted,
                    "abstained": non_leaf_abstained,
                    "rejected": non_leaf_rejected,
                    "rejection_rate": non_leaf_rejection_rate,
                    "false_acceptance_rate": non_leaf_far,
                    "failure_adjusted_rejection_rate": failure_adjusted_non_leaf_rejection_rate,
                    "failure_adjusted_false_acceptance_rate": failure_adjusted_non_leaf_far,
                },
                "overall_ood_summary": {
                    "total_unsupported_samples": n_ood,
                    "total_unsupported_submitted": n_ood_submitted,
                    "total_unsupported_evaluated": n_ood,
                    "total_falsely_accepted": total_ood_accepted,
                    "combined_false_acceptance_rate": combined_far,
                    "failure_adjusted_combined_false_acceptance_rate": failure_adjusted_combined_far,
                },
                "challenge_conditions": challenges,
            }
            return batch_results, batch_errors, metrics_dict

        # Evaluate primary and exploratory populations
        primary_batch = verified_samples if exclude_unverified else samples
        primary_results, primary_errors, primary_metrics = _evaluate_batch(primary_batch)

        exploratory_metrics = None
        if exclude_unverified and unverified_samples:
            _, _, exploratory_metrics = _evaluate_batch(unverified_samples)

        all_results = primary_results
        all_errors = primary_errors

        formal_gate_passed = (len(all_errors) == 0 and not allow_exploratory_subset)
        if allow_exploratory_subset:
            gate_status = "EXPLORATORY_SUBSET"
            is_formal = False
        elif formal_gate_passed:
            gate_status = "FORMAL_PASSED"
            is_formal = True
        else:
            gate_status = "FAILED_INTEGRITY_VIOLATION"
            is_formal = False

        metadata: Dict[str, Any] = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_type": self.evaluation_type,
            "gate_model_path": str(self.model_path),
            "gate_model_sha256": self.model_sha256,
            "is_trained_model": self.is_trained_model,
            "gate_backend": self.gate_backend,
            "fallback_reason": self.fallback_reason,
            "deterministic_fallback_active": not self.is_trained_model,
            "sidecar_validation": self.sidecar_validation,
            "total_samples": len(samples),
            "verified_samples_count": len(verified_samples),
            "unverified_samples_count": len(unverified_samples),
            "unverified_breakdown": unverified_breakdown,
            "exclude_unverified_from_primary": exclude_unverified,
            "submitted_samples": len(samples),
            "attempted_samples": len(primary_batch),
            "evaluated_samples": len(all_results),
            "failed_samples": len(all_errors),
            "formal_gate_evaluation_passed": formal_gate_passed,
            "is_formal_evaluation": is_formal,
            "gate_evaluation_status": gate_status,
            "allow_exploratory_subset": allow_exploratory_subset,
            "gate_inference_accounting": {
                "total_submitted": len(samples),
                "attempted": len(primary_batch),
                "successful": len(all_results),
                "failed": len(all_errors),
                "excluded": len(unverified_samples) if exclude_unverified else 0,
                "failure_rate": round(len(all_errors) / len(primary_batch), 4) if len(primary_batch) > 0 else 0.0,
            },
        }
        if not self.is_trained_model:
            metadata["warning"] = "EVALUATION_PERFORMED_ON_DETERMINISTIC_FALLBACK_NOT_TRAINED_GATE_MODEL"

        summary = {
            "metadata": metadata,
            "gate_metrics": primary_metrics,
            "errors": all_errors,
            "sample_results": [r.to_dict() for r in all_results],
        }
        if exploratory_metrics:
            summary["exploratory_metrics_unverified"] = exploratory_metrics

        return summary

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[Path, str],
        prefix: str = "gate_eval",
    ) -> Tuple[Path, Path]:
        """Saves gate evaluation results to JSON and Markdown summary."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_file = out_path / f"{prefix}_{timestamp_str}.json"
        md_file = out_path / f"{prefix}_{timestamp_str}_summary.md"

        with json_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        md_content = self.generate_markdown_summary(results)
        with md_file.open("w", encoding="utf-8") as f:
            f.write(md_content)

        return json_file, md_file

    @staticmethod
    def generate_markdown_summary(results: Dict[str, Any]) -> str:
        """Renders Markdown summary of leaf gate evaluation."""
        meta = results["metadata"]
        gm = results["gate_metrics"]
        supp = gm["supported_leaves"]
        other = gm["other_plant_species"]
        non_leaf = gm["non_leaf_objects"]
        ood = gm["overall_ood_summary"]

        is_trained = meta.get("is_trained_model", False)
        gate_backend = meta.get("gate_backend", "trained_model" if is_trained else "deterministic_fallback")

        def _fmt(rate: Optional[float], label: str) -> str:
            if rate is None:
                return f"**{label}: N/A (support=0)**"
            return f"**{label}: {rate * 100:.2f}%**"

        is_exploratory = (
            meta.get("gate_evaluation_status") == "EXPLORATORY_SUBSET"
            or not meta.get("is_formal_evaluation", True)
            or meta.get("allow_exploratory_subset", False)
        )
        is_formal_failed = meta.get("gate_evaluation_status") == "FAILED_INTEGRITY_VIOLATION"

        md = []
        if is_exploratory:
            md.append("# EXPLORATORY — NOT FOR FORMAL PERFORMANCE CLAIMS")
            md.append("## Leaf Acceptance Gate Field Evaluation Summary (Exploratory Subset)")
            md.append("")
            md.append("> [!WARNING]")
            md.append("> **EXPLORATORY GATE EVALUATION ACTIVE:** This evaluation was executed with `--allow-exploratory-subset`.")
            md.append("> It reflects exploratory or partial gate assessments and MUST NOT be cited for formal compliance claims.")
            md.append("")
        elif is_formal_failed:
            md.append("# FAILED FORMAL GATE EVALUATION")
            md.append("## Leaf Acceptance Gate Field Evaluation Summary (Failures Detected)")
            md.append("")
            md.append("> [!CAUTION]")
            md.append("> **FORMAL GATE EVALUATION FAILED:** Image processing or integrity failures were detected.")
            md.append("> Formal gate metrics are invalidated by processing dropouts.")
            md.append("")
        else:
            md.append("# Leaf Acceptance Gate Field Evaluation Summary (Formal Evaluation)")

        md.append(f"**Gate Evaluation Status:** `{meta.get('gate_evaluation_status', 'UNKNOWN')}`")
        md.append(f"**Evaluation Timestamp:** {meta['evaluation_timestamp']}")
        md.append(f"**Evaluation Type:** `{meta.get('evaluation_type', 'trained_model_evaluation')}`")
        md.append(f"**Gate Backend:** `{gate_backend}`")
        md.append(f"**Trained Model Evaluated:** `{'YES' if is_trained else 'NO (Deterministic Heuristic Fallback)'}`")
        if meta.get("fallback_reason"):
            md.append(f"**Fallback Reason:** `{meta['fallback_reason']}`")
        md.append(f"**Gate Checkpoint SHA-256:** `{meta['gate_model_sha256']}`")
        md.append(
            f"**Total Samples:** {meta['total_samples']} (Verified: {meta.get('verified_samples_count', 'N/A')}, "
            f"Unverified: {meta.get('unverified_samples_count', 'N/A')}, "
            f"Evaluated: {meta['evaluated_samples']}, Failed: {meta['failed_samples']})"
        )
        md.append("")

        if not is_trained:
            md.append("> [!WARNING]")
            md.append("> **DETERMINISTIC FALLBACK EVALUATION NOTICE**")
            md.append("> This evaluation was executed using the deterministic rule-based plant screening fallback,")
            md.append("> NOT the trained `leaf_gate.keras` neural model. These metrics reflect heuristic baseline")
            md.append("> capabilities only and must NOT be cited as trained neural gate performance.")
            md.append("")

        md.append("## Core Gate Reliability Metrics")
        md.append("| Target Domain | Submitted | Evaluated | Failed | Accepted | Abstained | Rejected | Key Metric (Conditional) | Failure-Adjusted Metric |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        md.append(
            f"| **Supported Crop Leaves** | {supp.get('submitted', supp['support'])} | {supp['support']} | {supp.get('failed', 0)} | "
            f"{supp['accepted']} | {supp['abstained']} | {supp['rejected']} | "
            f"{_fmt(supp['false_rejection_rate'], 'FRR')} | {_fmt(supp.get('failure_adjusted_false_rejection_rate'), 'Adj-FRR')} |"
        )
        md.append(
            f"| **Other Plant Species** | {other.get('submitted', other['support'])} | {other['support']} | {other.get('failed', 0)} | "
            f"{other['accepted']} | {other['abstained']} | {other['rejected']} | "
            f"{_fmt(other['false_acceptance_rate'], 'FAR')} | {_fmt(other.get('failure_adjusted_false_acceptance_rate'), 'Adj-FAR')} |"
        )
        md.append(
            f"| **Non-Leaf Objects / BG** | {non_leaf.get('submitted', non_leaf['support'])} | {non_leaf['support']} | {non_leaf.get('failed', 0)} | "
            f"{non_leaf['accepted']} | {non_leaf['abstained']} | {non_leaf['rejected']} | "
            f"{_fmt(non_leaf['false_acceptance_rate'], 'FAR')} | {_fmt(non_leaf.get('failure_adjusted_false_acceptance_rate'), 'Adj-FAR')} |"
        )
        md.append(
            f"| **Combined OOD Pool** | {ood.get('total_unsupported_submitted', ood['total_unsupported_samples'])} | "
            f"{ood['total_unsupported_samples']} | - | {ood['total_falsely_accepted']} | - | - | "
            f"{_fmt(ood['combined_false_acceptance_rate'], 'Combined FAR')} | {_fmt(ood.get('failure_adjusted_combined_false_acceptance_rate'), 'Adj-FAR')} |"
        )
        md.append("")

        md.append("## Challenge Conditions Robustness")
        md.append("| Challenge Type | Support | Accepted | Abstained | Rejected | Accept Rate |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for cname, cdata in gm.get("challenge_conditions", {}).items():
            acc_rate = cdata.get("acceptance_rate")
            rate_str = f"{acc_rate * 100:.2f}%" if acc_rate is not None else "N/A"
            md.append(
                f"| `{cname}` | {cdata['total']} | {cdata['accepted']} | {cdata['abstained']} | "
                f"{cdata['rejected']} | {rate_str} |"
            )
        md.append("")

        md.append("> [!IMPORTANT]")
        md.append("> Input gate acceptance indicates only that sufficient plant foliage was verified.")
        md.append("> It does **not** constitute proof that downstream disease classification is correct or reliable.")

        return "\n".join(md)
        md.append("")

        md.append("## Challenge Conditions Robustness")
        md.append("| Challenge Type | Support | Accepted | Abstained | Rejected | Accept Rate |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for cname, cdata in gm.get("challenge_conditions", {}).items():
            md.append(
                f"| `{cname}` | {cdata['total']} | {cdata['accepted']} | {cdata['abstained']} | "
                f"{cdata['rejected']} | {cdata['acceptance_rate'] * 100:.2f}% |"
            )
        md.append("")

        md.append("> [!IMPORTANT]")
        md.append("> Input gate acceptance indicates only that sufficient plant foliage was verified.")
        md.append("> It does **not** constitute proof that downstream disease classification is correct or reliable.")

        return "\n".join(md)
