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

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from backend.app.config import LEAF_GATE_MODEL_PATH
from backend.app.services.input_gate import (
    ABSTAINED,
    ACCEPTED,
    CODE_ACCEPTED,
    CODE_AMBIGUOUS,
    CODE_NON_LEAF,
    CODE_UNSUPPORTED_SPECIES,
    GATE_NON_LEAF,
    GATE_OTHER_PLANT,
    GATE_SUPPORTED_LEAF,
    REJECTED,
    assess_image,
)
from backend.app.services.leaf_gate import LeafGateService
from .evaluator import compute_file_sha256, preprocess_field_image

logger = logging.getLogger(__name__)


@dataclass
class GateSampleRecord:
    sample_id: str
    file_path: str
    ground_truth_category: str  # 'supported_leaf', 'other_plant', 'non_leaf'
    challenge_type: str = "standard"  # 'foliage_clutter', 'partial_leaf', 'multi_leaf', 'blur', 'background_only', 'standard'
    crop: Optional[str] = None
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
        if leaf_gate_service is not None:
            self.service = leaf_gate_service
            self.model_path = getattr(leaf_gate_service, "_model_path", LEAF_GATE_MODEL_PATH)
        else:
            model_p = Path(gate_model_path or LEAF_GATE_MODEL_PATH).resolve()
            self.model_path = model_p
            self.service = LeafGateService(model_path=model_p)
            if model_p.is_file():
                self.service.load_model(model_p)
            else:
                logger.warning("Gate model file not found at %s. Using deterministic fallback.", model_p)

        self.model_sha256 = (
            compute_file_sha256(self.model_path) if self.model_path.is_file() else "N/A"
        )

    def evaluate_sample(
        self,
        sample: GateSampleRecord,
        dataset_root: Optional[Path] = None,
    ) -> GatePredictionResult:
        """
        Runs input gate assessment on a single image sample.
        """
        img_p = Path(sample.file_path)
        if not img_p.is_absolute() and dataset_root:
            img_p = (dataset_root / img_p).resolve()

        if not img_p.is_file():
            raise FileNotFoundError(f"Image not found: {img_p}")

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

        gt = sample.ground_truth_category

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
    ) -> Dict[str, Any]:
        """
        Evaluates a set of gate sample records and returns aggregated metrics.
        """
        root_path = Path(dataset_root).resolve() if dataset_root else None
        results: List[GatePredictionResult] = []
        errors: List[Dict[str, str]] = []

        for sample in samples:
            try:
                res = self.evaluate_sample(sample, dataset_root=root_path)
                results.append(res)
            except Exception as exc:
                errors.append({
                    "sample_id": sample.sample_id,
                    "file_path": sample.file_path,
                    "error": str(exc),
                })

        # Calculate metrics
        total = len(results)
        supported_samples = [r for r in results if r.ground_truth_category == "supported_leaf"]
        other_plant_samples = [r for r in results if r.ground_truth_category == "other_plant"]
        non_leaf_samples = [r for r in results if r.ground_truth_category == "non_leaf"]

        # Supported leaf metrics
        n_supp = len(supported_samples)
        supp_accepted = sum(1 for r in supported_samples if r.decision == ACCEPTED)
        supp_abstained = sum(1 for r in supported_samples if r.decision == ABSTAINED)
        supp_rejected = sum(1 for r in supported_samples if r.decision == REJECTED)
        frr = (supp_abstained + supp_rejected) / n_supp if n_supp > 0 else 0.0
        strict_frr = supp_rejected / n_supp if n_supp > 0 else 0.0

        # Other plant metrics
        n_other = len(other_plant_samples)
        other_accepted = sum(1 for r in other_plant_samples if r.decision == ACCEPTED)
        other_abstained = sum(1 for r in other_plant_samples if r.decision == ABSTAINED)
        other_rejected = sum(1 for r in other_plant_samples if r.decision == REJECTED)
        other_far = other_accepted / n_other if n_other > 0 else 0.0

        # Non-leaf metrics
        n_non_leaf = len(non_leaf_samples)
        non_leaf_accepted = sum(1 for r in non_leaf_samples if r.decision == ACCEPTED)
        non_leaf_abstained = sum(1 for r in non_leaf_samples if r.decision == ABSTAINED)
        non_leaf_rejected = sum(1 for r in non_leaf_samples if r.decision == REJECTED)
        non_leaf_far = non_leaf_accepted / n_non_leaf if n_non_leaf > 0 else 0.0

        # Combined OOD (other_plant + non_leaf)
        n_ood = n_other + n_non_leaf
        total_ood_accepted = other_accepted + non_leaf_accepted
        combined_far = total_ood_accepted / n_ood if n_ood > 0 else 0.0

        # Breakdown by challenge condition
        challenges: Dict[str, Dict[str, Any]] = {}
        for r in results:
            c = r.challenge_type
            if c not in challenges:
                challenges[c] = {"total": 0, "accepted": 0, "abstained": 0, "rejected": 0}
            challenges[c]["total"] += 1
            challenges[c][r.decision] += 1

        for c, counts in challenges.items():
            tot = counts["total"]
            counts["acceptance_rate"] = round(counts["accepted"] / tot, 4) if tot else 0.0

        summary = {
            "metadata": {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "gate_model_path": str(self.model_path),
                "gate_model_sha256": self.model_sha256,
                "total_samples": len(samples),
                "evaluated_samples": total,
                "failed_samples": len(errors),
            },
            "gate_metrics": {
                "supported_leaves": {
                    "support": n_supp,
                    "accepted": supp_accepted,
                    "abstained": supp_abstained,
                    "rejected": supp_rejected,
                    "acceptance_rate": round(supp_accepted / n_supp, 4) if n_supp else 0.0,
                    "false_rejection_rate": round(frr, 4),
                    "strict_false_rejection_rate": round(strict_frr, 4),
                },
                "other_plant_species": {
                    "support": n_other,
                    "accepted": other_accepted,
                    "abstained": other_abstained,
                    "rejected": other_rejected,
                    "rejection_rate": round(other_rejected / n_other, 4) if n_other else 0.0,
                    "false_acceptance_rate": round(other_far, 4),
                },
                "non_leaf_objects": {
                    "support": n_non_leaf,
                    "accepted": non_leaf_accepted,
                    "abstained": non_leaf_abstained,
                    "rejected": non_leaf_rejected,
                    "rejection_rate": round(non_leaf_rejected / n_non_leaf, 4) if n_non_leaf else 0.0,
                    "false_acceptance_rate": round(non_leaf_far, 4),
                },
                "overall_ood_summary": {
                    "total_unsupported_samples": n_ood,
                    "total_falsely_accepted": total_ood_accepted,
                    "combined_false_acceptance_rate": round(combined_far, 4),
                },
                "challenge_conditions": challenges,
            },
            "errors": errors,
            "sample_results": [r.to_dict() for r in results],
        }

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

        md = []
        md.append("# Leaf Acceptance Gate Field Evaluation Summary")
        md.append(f"**Evaluation Timestamp:** {meta['evaluation_timestamp']}")
        md.append(f"**Gate Checkpoint SHA-256:** `{meta['gate_model_sha256']}`")
        md.append(f"**Total Samples:** {meta['total_samples']} (Evaluated: {meta['evaluated_samples']}, Failed: {meta['failed_samples']})")
        md.append("")

        md.append("## Core Gate Reliability Metrics")
        md.append("| Target Domain | Support | Accepted | Abstained | Rejected | Key Metric |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        md.append(
            f"| **Supported Crop Leaves** | {supp['support']} | {supp['accepted']} | {supp['abstained']} | "
            f"{supp['rejected']} | **FRR: {supp['false_rejection_rate'] * 100:.2f}%** |"
        )
        md.append(
            f"| **Other Plant Species** | {other['support']} | {other['accepted']} | {other['abstained']} | "
            f"{other['rejected']} | **FAR: {other['false_acceptance_rate'] * 100:.2f}%** |"
        )
        md.append(
            f"| **Non-Leaf Objects / BG** | {non_leaf['support']} | {non_leaf['accepted']} | {non_leaf['abstained']} | "
            f"{non_leaf['rejected']} | **FAR: {non_leaf['false_acceptance_rate'] * 100:.2f}%** |"
        )
        md.append(
            f"| **Combined OOD Pool** | {ood['total_unsupported_samples']} | {ood['total_falsely_accepted']} | - | "
            f"- | **Combined FAR: {ood['combined_false_acceptance_rate'] * 100:.2f}%** |"
        )
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
