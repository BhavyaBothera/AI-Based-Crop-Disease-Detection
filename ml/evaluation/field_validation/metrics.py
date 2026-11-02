"""
Statistical Metrics & Failure Analysis Engine
=============================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Calculates comprehensive evaluation metrics, confidence intervals,
subgroup performance slices, and confusion matrices for field validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .manifest_schema import (
    CLASS_ID_TO_NAME,
    NAME_TO_CLASS_ID,
    RAW_CLASS_TO_CROP,
    TARGET_CLASS_MAPPING,
)

CLASS_NAMES: List[str] = sorted(RAW_CLASS_TO_CROP.keys())
CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)  # 15


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Robust for small sample sizes and boundary probabilities (0 or 1).
    """
    if total <= 0:
        return 0.0, 0.0
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 1.64485
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4.0 * (total**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes 95% percentile bootstrap confidence intervals for Macro-F1.
    """
    if len(y_true) < 10:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_samples: List[float] = []

    for _ in range(num_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = f1_score(
            y_true[idx],
            y_pred[idx],
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        f1_samples.append(float(score))

    lower = float(np.percentile(f1_samples, 2.5))
    upper = float(np.percentile(f1_samples, 97.5))
    return lower, upper


@dataclass
class FieldEvaluationSummary:
    """Comprehensive evaluation summary payload."""
    total_samples: int
    correct_samples: int
    accuracy: float
    accuracy_ci_95: Tuple[float, float]
    top_3_accuracy: float
    top_3_ci_95: Tuple[float, float]
    macro_f1: float
    macro_f1_ci_95: Tuple[float, float]
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, Any]]
    per_crop: Dict[str, Dict[str, Any]]
    confusion_matrix: List[List[int]]
    subgroup_breakdowns: Dict[str, Dict[str, Dict[str, Any]]]
    confidence_stats_correct: Dict[str, float]
    confidence_stats_incorrect: Dict[str, float]
    low_performing_classes: List[str]
    group_level_accuracy: Optional[float] = None
    group_count: int = 0
    excluded_unverified_count: int = 0

    @property
    def total_evaluated(self) -> int:
        return self.total_samples

    @property
    def correct_top1(self) -> int:
        return self.correct_samples

    @property
    def overall_accuracy(self) -> float:
        return self.accuracy

    @property
    def overall_accuracy_ci_95(self) -> Tuple[float, float]:
        return self.accuracy_ci_95

    @property
    def top3_accuracy(self) -> float:
        return self.top_3_accuracy

    @property
    def top3_accuracy_ci_95(self) -> Tuple[float, float]:
        return self.top_3_ci_95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias
EvaluationResults = FieldEvaluationSummary


def calculate_field_metrics(
    eval_records: List[Dict[str, Any]],
    exclude_unverified: bool = False,
    bootstrap_samples: int = 500,
) -> FieldEvaluationSummary:
    """
    Computes all standard, sliced, and failure metrics from prediction records.
    """
    valid_records: List[Dict[str, Any]] = []
    excluded_count = 0

    for r in eval_records:
        v_status = r.get("verification_status") or r.get("metadata", {}).get("verification_status", "unknown")
        gt_class = r.get("ground_truth_class")

        if exclude_unverified and v_status in ("visually_suspected", "ambiguous_unresolved"):
            excluded_count += 1
            continue

        if gt_class in CLASS_NAME_TO_ID:
            valid_records.append(r)
        else:
            excluded_count += 1

    total = len(valid_records)
    if total == 0:
        return FieldEvaluationSummary(
            total_samples=0,
            correct_samples=0,
            accuracy=0.0,
            accuracy_ci_95=(0.0, 0.0),
            top_3_accuracy=0.0,
            top_3_ci_95=(0.0, 0.0),
            macro_f1=0.0,
            macro_f1_ci_95=(0.0, 0.0),
            macro_precision=0.0,
            macro_recall=0.0,
            weighted_f1=0.0,
            per_class={},
            per_crop={},
            confusion_matrix=[[0] * NUM_CLASSES for _ in range(NUM_CLASSES)],
            subgroup_breakdowns={},
            confidence_stats_correct={},
            confidence_stats_incorrect={},
            low_performing_classes=[],
            excluded_unverified_count=excluded_count,
        )

    y_true = np.array([CLASS_NAME_TO_ID[r["ground_truth_class"]] for r in valid_records], dtype=int)
    y_pred = np.array([CLASS_NAME_TO_ID.get(r["predicted_class"], -1) for r in valid_records], dtype=int)

    correct_mask = y_true == y_pred
    correct_count = int(np.sum(correct_mask))
    acc = correct_count / total
    acc_ci = wilson_score_interval(correct_count, total)

    # Top-3 Accuracy
    top3_hits = 0
    for r in valid_records:
        gt = r["ground_truth_class"]
        top3_list = r.get("top_3_predictions", [])
        top3_names = [
            cand if isinstance(cand, str) else cand.get("class_name", "")
            for cand in top3_list
        ]
        if gt in top3_names:
            top3_hits += 1

    top3_acc = top3_hits / total
    top3_ci = wilson_score_interval(top3_hits, total)

    all_class_labels = list(range(NUM_CLASSES))
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_class_labels, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, labels=all_class_labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=all_class_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=all_class_labels, average="weighted", zero_division=0))

    macro_f1_ci = bootstrap_macro_f1_ci(y_true, y_pred, num_bootstrap=bootstrap_samples)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, labels=all_class_labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=all_class_labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=all_class_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_class_labels)

    per_class_dict: Dict[str, Dict[str, Any]] = {}
    low_performing: List[str] = []

    for idx, name in enumerate(CLASS_NAMES):
        support_cnt = int(np.sum(y_true == idx))
        correct_cnt = int(cm[idx, idx]) if support_cnt > 0 else 0
        rec_ci = wilson_score_interval(correct_cnt, support_cnt) if support_cnt > 0 else (0.0, 0.0)

        per_class_dict[name] = {
            "class_id": idx,
            "crop": RAW_CLASS_TO_CROP.get(name, "Unknown"),
            "support": support_cnt,
            "correct": correct_cnt,
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "recall_ci_95": rec_ci,
            "low_support_flag": support_cnt < 10,
        }

        if support_cnt >= 5 and (f1_per_class[idx] < 0.85 or rec_per_class[idx] < 0.80):
            low_performing.append(name)

    # Per-crop breakdown
    crop_metrics: Dict[str, Dict[str, Any]] = {}
    crops = sorted(list({RAW_CLASS_TO_CROP.get(c, "") for c in CLASS_NAMES if c in RAW_CLASS_TO_CROP}))

    for crop in crops:
        crop_indices = [idx for idx, name in enumerate(CLASS_NAMES) if RAW_CLASS_TO_CROP.get(name) == crop]
        crop_mask = np.isin(y_true, crop_indices)
        crop_support = int(np.sum(crop_mask))
        if crop_support > 0:
            crop_correct = int(np.sum(y_true[crop_mask] == y_pred[crop_mask]))
            crop_acc = crop_correct / crop_support
            crop_f1 = float(f1_score(y_true[crop_mask], y_pred[crop_mask], labels=crop_indices, average="macro", zero_division=0))
            crop_prec = float(precision_score(y_true[crop_mask], y_pred[crop_mask], labels=crop_indices, average="macro", zero_division=0))
            crop_rec = float(recall_score(y_true[crop_mask], y_pred[crop_mask], labels=crop_indices, average="macro", zero_division=0))
            crop_metrics[crop] = {
                "support": crop_support,
                "correct": crop_correct,
                "accuracy": float(crop_acc),
                "accuracy_ci_95": wilson_score_interval(crop_correct, crop_support),
                "macro_f1": crop_f1,
                "precision": crop_prec,
                "recall": crop_rec,
            }

    # Confidence distribution (top prediction scores)
    def extract_score(rec: Dict[str, Any]) -> float:
        if "raw_scores" in rec and rec.get("predicted_class") in rec["raw_scores"]:
            return float(rec["raw_scores"][rec["predicted_class"]])
        if "top_3_predictions" in rec and len(rec["top_3_predictions"]) > 0:
            first = rec["top_3_predictions"][0]
            if isinstance(first, dict) and "raw_score" in first:
                return float(first["raw_score"])
        return float(rec.get("top_1_score", 0.0))

    scores_correct = [extract_score(r) for idx, r in enumerate(valid_records) if correct_mask[idx]]
    scores_incorrect = [extract_score(r) for idx, r in enumerate(valid_records) if not correct_mask[idx]]

    def calc_stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    conf_correct = calc_stats(scores_correct)
    conf_incorrect = calc_stats(scores_incorrect)

    # Subgroups (lighting, background, severity, device)
    subgroup_categories = [
        "lighting_condition",
        "background_type",
        "disease_severity",
        "capture_device",
        "verification_status",
    ]
    subgroups_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for cat in subgroup_categories:
        subgroups_dict[cat] = {}
        val_map: Dict[str, List[int]] = {}
        for idx, r in enumerate(valid_records):
            val = r.get(cat) or r.get("metadata", {}).get(cat) or "unknown"
            val = str(val).strip()
            if not val:
                val = "unknown"
            if val not in val_map:
                val_map[val] = []
            val_map[val].append(idx)

        for val_name, record_indices in sorted(val_map.items()):
            sub_count = len(record_indices)
            sub_true = y_true[record_indices]
            sub_pred = y_pred[record_indices]
            sub_correct = int(np.sum(sub_true == sub_pred))
            sub_acc = sub_correct / sub_count if sub_count > 0 else 0.0

            subgroups_dict[cat][val_name] = {
                "support": sub_count,
                "correct": sub_correct,
                "accuracy": float(sub_acc),
                "accuracy_ci_95": wilson_score_interval(sub_correct, sub_count),
                "low_sample_warning": sub_count < 10,
            }

    # Grouped / Plant-level accuracy
    group_map: Dict[str, List[int]] = {}
    for idx, r in enumerate(valid_records):
        grp = r.get("duplicate_group_id") or r.get("metadata", {}).get("duplicate_group_id") or "NONE"
        grp = str(grp).strip()
        if grp and grp != "NONE":
            if grp not in group_map:
                group_map[grp] = []
            group_map[grp].append(idx)

    group_accuracy: Optional[float] = None
    if group_map:
        group_correct = 0
        for grp, indices in group_map.items():
            grp_true = y_true[indices]
            grp_pred = y_pred[indices]
            if np.mean(grp_true == grp_pred) >= 0.5:
                group_correct += 1
        group_accuracy = group_correct / len(group_map)

    return FieldEvaluationSummary(
        total_samples=total,
        correct_samples=correct_count,
        accuracy=float(acc),
        accuracy_ci_95=acc_ci,
        top_3_accuracy=float(top3_acc),
        top_3_ci_95=top3_ci,
        macro_f1=macro_f1,
        macro_f1_ci_95=macro_f1_ci,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        weighted_f1=weighted_f1,
        per_class=per_class_dict,
        per_crop=crop_metrics,
        confusion_matrix=cm.tolist(),
        subgroup_breakdowns=subgroups_dict,
        confidence_stats_correct=conf_correct,
        confidence_stats_incorrect=conf_incorrect,
        low_performing_classes=low_performing,
        group_level_accuracy=group_accuracy,
        group_count=len(group_map),
        excluded_unverified_count=excluded_count,
    )


# Alias
calculate_metrics = calculate_field_metrics
