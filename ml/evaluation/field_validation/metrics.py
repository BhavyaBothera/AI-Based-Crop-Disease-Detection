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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .manifest_schema import (
    RAW_CLASS_TO_CROP,
    UNVERIFIED_TIERS,
    VERIFIED_TIERS,
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
    clusters: Optional[np.ndarray] = None,
    num_bootstrap: int = 500,
    seed: int = 42,
    labels: Optional[List[int]] = None,
) -> Tuple[Optional[Tuple[float, float]], bool, Optional[str]]:
    """
    Computes 95% percentile bootstrap confidence intervals for Macro-F1.
    Supports cluster-aware bootstrap when clusters (duplicate_group_id) are provided.
    Phase 6: Fails reliably with unavailable status when clusters < 5 or sample size < 10.
    Never duplicates the point estimate as an interval.
    Returns:
        (ci_interval_tuple_or_None, is_available_bool, reason_str_or_None)
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return None, False, "Zero samples provided."

    eval_labels = labels if labels is not None else sorted(list(set(y_true) | set(y_pred)))
    if len(eval_labels) == 0:
        return None, False, "No evaluated class labels available."

    n = len(y_true)
    if n < 10:
        return None, False, f"Insufficient sample size ({n} < 10) for reliable bootstrap CI."

    rng = np.random.default_rng(seed)
    f1_samples: List[float] = []

    if clusters is not None and len(clusters) == n:
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) < 5:
            # Phase 6: When unique clusters < 5, do NOT duplicate point estimate; report unavailable
            return None, False, f"Insufficient unique clusters ({len(unique_clusters)} < 5) for cluster-aware bootstrap CI."

        cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
        n_clusters = len(unique_clusters)

        for _ in range(num_bootstrap):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
            score = f1_score(
                y_true[sampled_indices],
                y_pred[sampled_indices],
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
            f1_samples.append(float(score))
    else:
        for _ in range(num_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            score = f1_score(
                y_true[idx],
                y_pred[idx],
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
            f1_samples.append(float(score))

    lower = float(np.percentile(f1_samples, 2.5))
    upper = float(np.percentile(f1_samples, 97.5))
    return (round(lower, 4), round(upper, 4)), True, None


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
    fixed_macro_f1: float
    observed_macro_f1: float
    macro_f1_ci_95: Optional[Tuple[float, float]]
    macro_f1_ci_available: bool
    macro_f1_ci_reason: Optional[str]
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
    observed_classes_count: int = 0
    unobserved_classes: List[str] = field(default_factory=list)
    verified_sample_count: int = 0
    unverified_breakdown: Dict[str, int] = field(default_factory=dict)
    cluster_metrics: Dict[str, Any] = field(default_factory=dict)
    unclustered_sample_count: int = 0
    unclustered_groups_excluded: int = 0
    inference_accounting: Dict[str, Any] = field(default_factory=dict)

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
    exclude_unverified: bool = True,
    bootstrap_samples: int = 500,
    inference_accounting: Optional[Dict[str, Any]] = None,
) -> FieldEvaluationSummary:
    """
    Computes all standard, sliced, cluster-aware, and failure metrics from prediction records.
    By default excludes visually suspected/unresolved samples from primary verified metrics (Bug 4).
    Reports both fixed_macro_f1 and observed_macro_f1 (Phase 6.1).
    Cluster-aware bootstrap reports unavailable when clusters < 5 (Phase 6.2).
    Cluster consensus applies documented plurality with confidence tie-breaking (Phase 6.3).
    """
    valid_records: List[Dict[str, Any]] = []
    excluded_count = 0
    unverified_breakdown: Dict[str, int] = {}

    for r in eval_records:
        v_status = r.get("verification_status") or r.get("metadata", {}).get("verification_status", "unknown")
        gt_class = r.get("ground_truth_class")

        if v_status in UNVERIFIED_TIERS:
            unverified_breakdown[v_status] = unverified_breakdown.get(v_status, 0) + 1

        if exclude_unverified and v_status in UNVERIFIED_TIERS:
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
            fixed_macro_f1=0.0,
            observed_macro_f1=0.0,
            macro_f1_ci_95=None,
            macro_f1_ci_available=False,
            macro_f1_ci_reason="No valid records to evaluate.",
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
            group_level_accuracy=None,
            group_count=0,
            excluded_unverified_count=excluded_count,
            observed_classes_count=0,
            unobserved_classes=list(CLASS_NAMES),
            verified_sample_count=0,
            unverified_breakdown=unverified_breakdown,
            cluster_metrics={},
            unclustered_sample_count=0,
            unclustered_groups_excluded=0,
            inference_accounting=inference_accounting or {},
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
    observed_labels = sorted(list({int(c) for c in y_true} | {int(c) for c in y_pred if c >= 0}))
    eval_labels = observed_labels if observed_labels else all_class_labels

    fixed_macro_f1 = float(f1_score(y_true, y_pred, labels=all_class_labels, average="macro", zero_division=0))
    observed_macro_f1 = float(f1_score(y_true, y_pred, labels=observed_labels, average="macro", zero_division=0)) if observed_labels else 0.0
    macro_f1 = observed_macro_f1

    macro_prec = float(precision_score(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0))

    # Cluster-aware bootstrap confidence interval (Phase 6.2)
    cluster_ids = []
    unclustered_count = 0
    for r in valid_records:
        grp = r.get("duplicate_group_id") or r.get("plant_id") or r.get("metadata", {}).get("duplicate_group_id") or "NONE"
        grp = str(grp).strip()
        if grp and grp != "NONE":
            cluster_ids.append(grp)
        else:
            unclustered_count += 1
            cluster_ids.append(f"single_{r.get('image_id', id(r))}")
    cluster_arr = np.array(cluster_ids)
    macro_f1_ci, ci_available, ci_reason = bootstrap_macro_f1_ci(
        y_true, y_pred, clusters=cluster_arr, num_bootstrap=bootstrap_samples, labels=eval_labels
    )

    # Observed vs unobserved classes (Phase 6.1)
    observed_classes = [name for idx, name in enumerate(CLASS_NAMES) if np.sum(y_true == idx) > 0]
    unobserved_classes = [name for idx, name in enumerate(CLASS_NAMES) if np.sum(y_true == idx) == 0]

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

    # Confidence distribution with NaN/Inf protection (Bug 7)
    def extract_score(rec: Dict[str, Any]) -> float:
        sc = 0.0
        if "raw_scores" in rec and rec.get("predicted_class") in rec["raw_scores"]:
            sc = float(rec["raw_scores"][rec["predicted_class"]])
        elif "top_3_predictions" in rec and len(rec["top_3_predictions"]) > 0:
            first = rec["top_3_predictions"][0]
            if isinstance(first, dict) and "raw_score" in first:
                sc = float(first["raw_score"])
        else:
            sc = float(rec.get("top_1_score", 0.0))
        if math.isnan(sc) or math.isinf(sc):
            return 0.0
        return sc

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

    # Grouped / Plant-level accuracy & cluster metrics (Phase 6.3)
    group_map: Dict[str, List[int]] = {}
    for idx, r in enumerate(valid_records):
        grp = r.get("duplicate_group_id") or r.get("metadata", {}).get("duplicate_group_id") or "NONE"
        grp = str(grp).strip()
        if grp and grp != "NONE":
            if grp not in group_map:
                group_map[grp] = []
            group_map[grp].append(idx)

    group_accuracy: Optional[float] = None
    cluster_metrics: Dict[str, Any] = {}
    if group_map:
        group_correct = 0
        group_top3_hits = 0
        group_confidences: List[float] = []

        for grp, indices in group_map.items():
            grp_true = y_true[indices]
            cluster_gt = grp_true[0]

            grp_preds = y_pred[indices]
            class_votes: Dict[int, int] = {}
            for p in grp_preds:
                class_votes[p] = class_votes.get(p, 0) + 1

            max_votes = max(class_votes.values())
            tied_classes = [c for c, v in class_votes.items() if v == max_votes]

            # Documented plurality tie-breaking rule (Phase 6.3):
            # 1. Class with highest mean predicted confidence among tied classes
            # 2. Lower class ID if still tied
            if len(tied_classes) == 1:
                consensus_class = tied_classes[0]
            else:
                best_class = tied_classes[0]
                best_conf = -1.0
                for c in sorted(tied_classes):
                    c_confs = [
                        extract_score(valid_records[i])
                        for i in indices
                        if y_pred[i] == c
                    ]
                    mean_c_conf = float(np.mean(c_confs)) if c_confs else 0.0
                    if mean_c_conf > best_conf:
                        best_conf = mean_c_conf
                        best_class = c
                consensus_class = best_class

            # Mean confidence of predictions supporting the consensus class
            supp_confs = [
                extract_score(valid_records[i])
                for i in indices
                if y_pred[i] == consensus_class
            ]
            agg_conf = float(np.mean(supp_confs)) if supp_confs else 0.0
            group_confidences.append(agg_conf)

            if consensus_class == cluster_gt:
                group_correct += 1

            # Top-3 hit across group
            top3_in_cluster = sum(
                1 for i in indices
                if valid_records[i].get("ground_truth_class") in [
                    c if isinstance(c, str) else c.get("class_name", "")
                    for c in valid_records[i].get("top_3_predictions", [])
                ]
            )
            if top3_in_cluster / len(indices) >= 0.5:
                group_top3_hits += 1

        group_accuracy = group_correct / len(group_map)
        cluster_metrics = {
            "cluster_count": len(group_map),
            "cluster_accuracy": round(group_accuracy, 4),
            "cluster_top3_accuracy": round(group_top3_hits / len(group_map), 4),
            "mean_consensus_confidence": round(float(np.mean(group_confidences)), 4) if group_confidences else 0.0,
            "aggregation_method": "plurality_with_confidence_tie_breaking",
            "unclustered_groups_excluded": unclustered_count,
        }

    verified_count = sum(1 for r in valid_records if (r.get("verification_status") or "unknown") in VERIFIED_TIERS)

    return FieldEvaluationSummary(
        total_samples=total,
        correct_samples=correct_count,
        accuracy=float(acc),
        accuracy_ci_95=acc_ci,
        top_3_accuracy=float(top3_acc),
        top_3_ci_95=top3_ci,
        macro_f1=macro_f1,
        fixed_macro_f1=fixed_macro_f1,
        observed_macro_f1=observed_macro_f1,
        macro_f1_ci_95=macro_f1_ci,
        macro_f1_ci_available=ci_available,
        macro_f1_ci_reason=ci_reason,
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
        observed_classes_count=len(observed_classes),
        unobserved_classes=unobserved_classes,
        verified_sample_count=verified_count,
        unverified_breakdown=unverified_breakdown,
        cluster_metrics=cluster_metrics,
        unclustered_sample_count=unclustered_count,
        unclustered_groups_excluded=unclustered_count,
        inference_accounting=inference_accounting or {},
    )


# Alias
calculate_metrics = calculate_field_metrics
