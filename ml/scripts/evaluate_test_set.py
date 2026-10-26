#!/usr/bin/env python3
"""
Final One-Time Test-Set Evaluation Script
========================================
Part of the "AI-Based Crop Disease Detection and Treatment Recommendation System"

Model Evaluated:
  ml/models/experiment_weighted/best_model.keras (Selected via Validation Macro-F1: 0.968587)

Strict Integrity Rules:
  1. Evaluates ONLY the fixed test split (split == 'test') from ml/data/dataset_manifest.csv.
  2. Exactly 3,413 test images evaluated.
  3. Uses deterministic evaluation preprocessing:
     - 224x224 RGB [0, 255] raw pixel input.
     - Embedded layer rescaling [-1, 1] inside model (no double preprocessing).
     - No shuffle, no data augmentation.
  4. Generates:
     - docs/audit_reports/final_test_evaluation_report.json
     - docs/audit_reports/final_test_evaluation_report.md
  5. Cross-references recorded validation metrics.
  6. Preserves the dataset audit near-duplicate caveat.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

# Resolve Project Root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.scripts.train_model import (
    CLASS_ID_TO_NAME,
    EXPECTED_COUNTS,
    NUM_CLASSES,
    TARGET_CLASS_MAPPING,
    _load_and_resize_image,
)


def load_test_records(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Loads strictly the test records from dataset_manifest.csv.
    Asserts exact counts and verifies image existence on disk.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    test_records: List[Dict[str, Any]] = []
    train_count = 0
    val_count = 0
    excluded_count = 0
    missing_files: List[str] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            status = row.get("status", "").strip().lower()
            if status != "included":
                excluded_count += 1
                continue

            split = row.get("split", "").strip().lower()
            if split == "train":
                train_count += 1
                continue
            elif split == "val":
                val_count += 1
                continue
            elif split != "test":
                continue

            class_id = int(row["class_id"])
            class_name = row.get("class_name", "").strip()
            if class_name != CLASS_ID_TO_NAME[class_id]:
                raise ValueError(
                    f"Class mapping mismatch at row {row_idx}: {class_id} vs {class_name}"
                )

            rel_path = row["relative_path"].strip()
            full_path = project_root / "ml" / rel_path
            if not full_path.is_file():
                full_path = project_root / rel_path
            if not full_path.is_file():
                missing_files.append(rel_path)

            test_records.append({
                "image_id": row["image_id"],
                "file_path": str(full_path),
                "class_id": class_id,
                "class_name": class_name,
            })

    print(f"Manifest Test Split Verification:")
    print(f"  * Train records skipped : {train_count:,}")
    print(f"  * Val records skipped   : {val_count:,}")
    print(f"  * Test records loaded   : {len(test_records):,}")
    print(f"  * Excluded rows skipped : {excluded_count:,}")
    print(f"  * Missing test files    : {len(missing_files):,}")

    if missing_files:
        raise FileNotFoundError(f"Found {len(missing_files)} missing test files: {missing_files[:5]}")

    if len(test_records) != EXPECTED_COUNTS["test"]:
        raise ValueError(
            f"Test count mismatch: expected {EXPECTED_COUNTS['test']}, got {len(test_records)}"
        )
    if train_count != EXPECTED_COUNTS["train"]:
        raise ValueError(
            f"Train count mismatch: expected {EXPECTED_COUNTS['train']}, got {train_count}"
        )
    if val_count != EXPECTED_COUNTS["val"]:
        raise ValueError(
            f"Validation count mismatch: expected {EXPECTED_COUNTS['val']}, got {val_count}"
        )

    y_true = np.array([r["class_id"] for r in test_records], dtype=np.int32)
    return test_records, y_true


def create_test_dataset(
    records: List[Dict[str, Any]],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Creates a deterministic test tf.data pipeline with zero augmentation."""
    file_paths = [r["file_path"] for r in records]
    labels = [r["class_id"] for r in records]

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(
        lambda p, y: (_load_and_resize_image(p), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def main():
    manifest_path = PROJECT_ROOT / "ml" / "data" / "dataset_manifest.csv"
    model_path = PROJECT_ROOT / "ml" / "models" / "experiment_weighted" / "best_model.keras"
    metadata_path = PROJECT_ROOT / "ml" / "models" / "experiment_weighted" / "best_model_metadata.json"
    val_comparison_path = PROJECT_ROOT / "docs" / "audit_reports" / "model_validation_comparison.json"

    # Verify input files
    for p in [manifest_path, model_path, metadata_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    val_metrics = {}
    if val_comparison_path.is_file():
        with open(val_comparison_path, "r", encoding="utf-8") as f:
            val_comp = json.load(f)
            val_metrics = val_comp["models"]["class_weighted"]

    # 1. Load test records
    test_records, y_true = load_test_records(manifest_path, PROJECT_ROOT)
    test_ds = create_test_dataset(test_records, batch_size=32)

    # 2. Load model
    print(f"\nLoading selected model: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 3. Predict on test dataset
    print("\nRunning forward pass on fixed test split (3,413 images)...")
    start_time = time.time()
    probs = model.predict(test_ds, verbose=1)
    eval_duration = time.time() - start_time
    print(f"Prediction complete in {eval_duration:.2f} seconds.")

    preds = np.argmax(probs, axis=-1)

    # 4. Compute compiled loss and pure cross-entropy
    eval_res = model.evaluate(test_ds, verbose=0)
    compiled_test_loss = float(eval_res[0]) if isinstance(eval_res, list) else float(eval_res)

    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    pure_cce_test_loss = float(cce(y_true, probs).numpy())

    # 5. Calculate Metrics
    total_samples = len(y_true)
    correct_count = int(np.sum(preds == y_true))
    incorrect_count = total_samples - correct_count

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    # Per-Class Metrics
    per_class_prec = precision_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_rec = recall_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_f1 = f1_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    # Verify reconciliation with confusion matrix
    cm_total = int(np.sum(cm))
    cm_correct = int(np.trace(cm))
    assert cm_total == total_samples == 3413, f"CM total {cm_total} != 3413"
    assert cm_correct == correct_count, f"CM correct {cm_correct} != {correct_count}"

    per_class_results = {}
    for cid in range(NUM_CLASSES):
        supp = int(np.sum(y_true == cid))
        per_class_results[str(cid)] = {
            "class_id": cid,
            "class_name": CLASS_ID_TO_NAME[cid],
            "crop": TARGET_CLASS_MAPPING[CLASS_ID_TO_NAME[cid]]["crop"],
            "condition": TARGET_CLASS_MAPPING[CLASS_ID_TO_NAME[cid]]["condition"],
            "support": supp,
            "precision": float(per_class_prec[cid]),
            "recall": float(per_class_rec[cid]),
            "f1_score": float(per_class_f1[cid]),
            "correct": int(cm[cid, cid]),
            "false_negatives": supp - int(cm[cid, cid]),
            "false_positives": int(np.sum(cm[:, cid])) - int(cm[cid, cid]),
        }

    # Identify lowest recall and F1 classes
    sorted_by_recall = sorted(per_class_results.values(), key=lambda x: x["recall"])
    sorted_by_f1 = sorted(per_class_results.values(), key=lambda x: x["f1_score"])

    # 6. Build Final Report Object
    report_data = {
        "evaluation_name": "Final One-Time Test-Set Evaluation",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model_checkpoint": str(model_path),
        "model_metadata": selected_meta,
        "manifest_path": str(manifest_path),
        "split_evaluated": "test",
        "total_test_samples": total_samples,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "evaluation_duration_seconds": round(eval_duration, 2),
        "metrics": {
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "test_macro_precision": macro_prec,
            "test_macro_recall": macro_rec,
            "test_weighted_f1": weighted_f1,
            "test_compiled_loss": compiled_test_loss,
            "test_pure_cross_entropy_loss": pure_cce_test_loss,
            "test_l2_regularization_penalty": compiled_test_loss - pure_cce_test_loss,
        },
        "validation_comparison": {
            "val_accuracy": val_metrics.get("accuracy"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "val_macro_precision": val_metrics.get("macro_precision"),
            "val_macro_recall": val_metrics.get("macro_recall"),
            "val_weighted_f1": val_metrics.get("weighted_f1"),
            "val_compiled_loss": val_metrics.get("validation_loss"),
            "val_pure_cross_entropy_loss": val_metrics.get("pure_cross_entropy_loss"),
            "delta_test_minus_val": {
                "accuracy": acc - val_metrics.get("accuracy", acc),
                "macro_f1": macro_f1 - val_metrics.get("macro_f1", macro_f1),
                "macro_precision": macro_prec - val_metrics.get("macro_precision", macro_prec),
                "macro_recall": macro_rec - val_metrics.get("macro_recall", macro_rec),
                "weighted_f1": weighted_f1 - val_metrics.get("weighted_f1", weighted_f1),
            },
        },
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "lowest_recall_classes": [
            {"class_name": c["class_name"], "recall": c["recall"], "f1": c["f1_score"], "support": c["support"]}
            for c in sorted_by_recall[:3]
        ],
        "lowest_f1_classes": [
            {"class_name": c["class_name"], "f1": c["f1_score"], "recall": c["recall"], "support": c["support"]}
            for c in sorted_by_f1[:3]
        ],
        "rare_class_performance": per_class_results["4"],  # Potato___healthy
        "caveats": {
            "near_duplicate_caveat": (
                "The Phase 2 dataset audit documented five cross-class near-duplicate pairs across split boundaries "
                "awaiting human review. Complete duplicate-free separation is not claimed."
            ),
            "treatment_recommendation_caveat": (
                "Classification evaluation metrics do not infer or validate clinical/agricultural treatment efficacy."
            ),
        },
    }

    # Save JSON Report
    reports_dir = PROJECT_ROOT / "docs" / "audit_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "final_test_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Saved JSON report: {json_path}")

    # Save Markdown Summary
    md_path = reports_dir / "final_test_evaluation_report.md"
    generate_markdown_report(report_data, md_path)
    print(f"[OK] Saved Markdown summary: {md_path}")


def generate_markdown_report(data: Dict[str, Any], output_path: Path):
    """Generates the final comprehensive Markdown evaluation document."""
    m = data["metrics"]
    v_comp = data["validation_comparison"]
    delta = v_comp["delta_test_minus_val"]
    cm = np.array(data["confusion_matrix"])

    md = []
    md.append("# Final Test-Set Evaluation Report")
    md.append("## AI-Based Crop Disease Detection and Treatment Recommendation System")
    md.append("")
    md.append(f"**Evaluation Date**: `{data['timestamp_utc']}`  ")
    md.append(f"**Selected Model**: [`ml/models/experiment_weighted/best_model.keras`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/models/experiment_weighted/best_model.keras)  ")
    md.append(f"**Selection Criterion**: Highest Validation Macro-F1 (`{data['model_metadata']['val_macro_f1']:.6f}`)  ")
    md.append(f"**Dataset Evaluated**: Fixed Test Split (`split == 'test'`) from [`ml/data/dataset_manifest.csv`](file:///C:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/ml/data/dataset_manifest.csv)  ")
    md.append(f"**Total Test Samples**: **{data['total_test_samples']:,}** images  ")
    md.append(f"**Evaluation Duration**: **{data['evaluation_duration_seconds']:.2f} seconds**  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary & Primary Metrics")
    md.append("")
    md.append("| Primary Metric | Test Set Result | Validation Result (Model B) | Generalization Delta (Test − Val) |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Macro-F1 (Primary)** | **`{m['test_macro_f1']:.6f}`** | `{v_comp['val_macro_f1']:.6f}` | **`{delta['macro_f1']:+.6f}`** |")
    md.append(f"| **Accuracy** | **`{m['test_accuracy']:.6f}`** ({data['correct_predictions']:,} / {data['total_test_samples']:,}) | `{v_comp['val_accuracy']:.6f}` | **`{delta['accuracy']:+.6f}`** |")
    md.append(f"| **Macro-Precision** | **`{m['test_macro_precision']:.6f}`** | `{v_comp['val_macro_precision']:.6f}` | **`{delta['macro_precision']:+.6f}`** |")
    md.append(f"| **Macro-Recall** | **`{m['test_macro_recall']:.6f}`** | `{v_comp['val_macro_recall']:.6f}` | **`{delta['macro_recall']:+.6f}`** |")
    md.append(f"| **Weighted-F1** | **`{m['test_weighted_f1']:.6f}`** | `{v_comp['val_weighted_f1']:.6f}` | **`{delta['weighted_f1']:+.6f}`** |")
    md.append(f"| **Total Compiled Loss** | **`{m['test_compiled_loss']:.6f}`** | `{v_comp['val_compiled_loss']:.6f}` | `{(m['test_compiled_loss'] - v_comp['val_compiled_loss']):+.6f}` |")
    md.append(f"| **Pure Cross-Entropy Loss** | **`{m['test_pure_cross_entropy_loss']:.6f}`** | `{v_comp['val_pure_cross_entropy_loss']:.6f}` | `{(m['test_pure_cross_entropy_loss'] - v_comp['val_pure_cross_entropy_loss']):+.6f}` |")
    md.append("")
    md.append(f"- **Correct Classifications**: **{data['correct_predictions']:,}** / {data['total_test_samples']:,} ({m['test_accuracy'] * 100:.2f}%)")
    md.append(f"- **Misclassifications**: **{data['incorrect_predictions']:,}** / {data['total_test_samples']:,} ({(1.0 - m['test_accuracy']) * 100:.2f}%)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Per-Class Test Performance (All 15 Solanaceae Classes)")
    md.append("")
    md.append("| ID | Class Name | Support | Correct | Errors | Precision | Recall | F1-Score |")
    md.append("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |")

    for cid in range(NUM_CLASSES):
        pc = data["per_class"][str(cid)]
        md.append(
            f"| {cid} | `{pc['class_name']}` | {pc['support']} | {pc['correct']} | {pc['false_negatives']} | "
            f"{pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1_score']:.4f} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Challenging & Rare Class Analysis")
    md.append("")
    r4 = data["rare_class_performance"]
    md.append("### Highly Underrepresented Class Performance:")
    md.append(f"1. **Class 4 (`Potato___healthy`)** — Test Support: **{r4['support']} images** (106 Train / 23 Val)")
    md.append(f"   * Precision: **`{r4['precision']:.4f}`**")
    md.append(f"   * Recall: **`{r4['recall']:.4f}`** ({r4['correct']} / {r4['support']} correct)")
    md.append(f"   * F1-Score: **`{r4['f1_score']:.4f}`**")
    md.append(f"   * Errors: {r4['false_negatives']} false negatives, {r4['false_positives']} false positives.")
    md.append("")
    r13 = data["per_class"]["13"]
    md.append(f"2. **Class 13 (`Tomato___Tomato_mosaic_virus`)** — Test Support: **{r13['support']} images** (261 Train / 56 Val)")
    md.append(f"   * Precision: **`{r13['precision']:.4f}`**")
    md.append(f"   * Recall: **`{r13['recall']:.4f}`** ({r13['correct']} / {r13['support']} correct)")
    md.append(f"   * F1-Score: **`{r13['f1_score']:.4f}`**")
    md.append("")
    md.append("### Lowest-Performing Classes on Test Split:")
    md.append("- **Lowest Recall**:")
    for lr in data["lowest_recall_classes"]:
        md.append(f"  * `{lr['class_name']}`: Recall = **`{lr['recall']:.4f}`**, F1 = **`{lr['f1']:.4f}`** (Support: {lr['support']})")
    md.append("- **Lowest F1-Score**:")
    for lf in data["lowest_f1_classes"]:
        md.append(f"  * `{lf['class_name']}`: F1 = **`{lf['f1']:.4f}`**, Recall = **`{lf['recall']:.4f}`** (Support: {lf['support']})")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Test Set 15 × 15 Confusion Matrix")
    md.append("")
    md.append("Axes definition:")
    md.append("- **Rows (Vertical)**: Ground Truth Label ($T_0$ to $T_{14}$)")
    md.append("- **Columns (Horizontal)**: Model Prediction ($0$ to $14$)")
    md.append("")
    md.append("```text")
    header = "Pred: " + " ".join(f"{c:4d}" for c in range(NUM_CLASSES))
    md.append(header)
    md.append("-" * len(header))
    for r in range(NUM_CLASSES):
        row_str = f"T{r:2d} : " + " ".join(f"{cm[r, c]:4d}" for c in range(NUM_CLASSES))
        md.append(row_str)
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Methodological Integrity & Dataset Audit Caveats")
    md.append("")
    md.append("> [!IMPORTANT]")
    md.append("> **Dataset Audit Caveat Preserved**:  ")
    md.append(f"> {data['caveats']['near_duplicate_caveat']}")
    md.append("")
    md.append("> [!NOTE]")
    md.append("> **Strict Finality**:  ")
    md.append("> This test evaluation was conducted exactly once on the pre-selected model without hyperparameter adjustments, threshold tuning, or test-driven iteration.")
    md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
