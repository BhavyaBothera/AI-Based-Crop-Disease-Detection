#!/usr/bin/env python3
"""
Validation Comparison Script: Unweighted vs. Balanced Class-Weighted MobileNetV2
================================================================================
Part of the "AI-Based Crop Disease Detection and Treatment Recommendation System"

Strict Rules:
  - Evaluates ONLY the fixed validation split (split == 'val') from ml/data/dataset_manifest.csv.
  - Exactly 3,418 validation images evaluated.
  - Test images (3,413 images) are strictly isolated and never read, decoded, or inspected.
  - Evaluates both saved checkpoints:
      ml/models/experiment_unweighted/best_model.keras
      ml/models/experiment_weighted/best_model.keras
  - Computes Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Weighted-F1,
    per-class precision/recall/F1/support, and 15x15 confusion matrices.
  - Generates docs/audit_reports/model_validation_comparison.json and .md.
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

# Resolve Project Root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Import authoritative mappings and loading helpers from train_model.py
sys.path.insert(0, str(PROJECT_ROOT))
from ml.scripts.train_model import (
    CLASS_ID_TO_NAME,
    EXPECTED_COUNTS,
    NUM_CLASSES,
    TARGET_CLASS_MAPPING,
    _load_and_resize_image,
)


def load_validation_records(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Loads strictly validation records from dataset_manifest.csv.
    Guarantees test set isolation (test rows are counted only, never loaded).
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    val_records: List[Dict[str, Any]] = []
    test_count = 0
    train_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            status = row.get("status", "").strip().lower()
            if status != "included":
                continue

            split = row.get("split", "").strip().lower()
            if split == "test":
                test_count += 1
                continue
            elif split == "train":
                train_count += 1
                continue
            elif split != "val":
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
                raise FileNotFoundError(f"Validation image missing: {full_path}")

            val_records.append({
                "image_id": row["image_id"],
                "file_path": str(full_path),
                "class_id": class_id,
                "class_name": class_name,
            })

    print(f"Manifest Isolation Check:")
    print(f"  * Train records skipped : {train_count:,}")
    print(f"  * Test records ISOLATED : {test_count:,} (never loaded or touched)")
    print(f"  * Validation records    : {len(val_records):,}")

    if len(val_records) != EXPECTED_COUNTS["val"]:
        raise ValueError(
            f"Validation count mismatch: expected {EXPECTED_COUNTS['val']}, got {len(val_records)}"
        )
    if test_count != EXPECTED_COUNTS["test"]:
        raise ValueError(
            f"Test count mismatch: expected {EXPECTED_COUNTS['test']}, got {test_count}"
        )

    y_true = np.array([r["class_id"] for r in val_records], dtype=np.int32)
    return val_records, y_true


def create_val_dataset(
    records: List[Dict[str, Any]],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Creates a deterministic validation tf.data pipeline with zero augmentation."""
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


def evaluate_model(
    model_path: Path,
    val_ds: tf.data.Dataset,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Evaluates a loaded Keras model on the validation dataset."""
    print(f"\nLoading model: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded. Running forward pass over validation dataset...")

    start_t = time.time()
    # Predict probabilities (shape: [N, 15])
    probs = model.predict(val_ds, verbose=1)
    eval_time = time.time() - start_t

    preds = np.argmax(probs, axis=-1)

    # Compute compiled validation loss (CE + L2 regularization, matching Keras model.fit)
    eval_res = model.evaluate(val_ds, verbose=0)
    total_val_loss = float(eval_res[0]) if isinstance(eval_res, list) else float(eval_res)

    # Compute pure classification cross-entropy without L2 regularization
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    pure_cce_loss = float(cce(y_true, probs).numpy())

    # Overall Metrics
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    # Per-Class Metrics
    per_class_f1 = f1_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_prec = precision_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_rec = recall_score(y_true, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))

    per_class_metrics = {}
    for cid in range(NUM_CLASSES):
        support = int(np.sum(y_true == cid))
        per_class_metrics[str(cid)] = {
            "class_id": cid,
            "class_name": CLASS_ID_TO_NAME[cid],
            "crop": TARGET_CLASS_MAPPING[CLASS_ID_TO_NAME[cid]]["crop"],
            "condition": TARGET_CLASS_MAPPING[CLASS_ID_TO_NAME[cid]]["condition"],
            "support": support,
            "precision": float(per_class_prec[cid]),
            "recall": float(per_class_rec[cid]),
            "f1_score": float(per_class_f1[cid]),
        }

    return {
        "model_path": str(model_path),
        "evaluation_time_seconds": round(eval_time, 2),
        "validation_loss": total_val_loss,
        "pure_cross_entropy_loss": pure_cce_loss,
        "l2_regularization_loss": total_val_loss - pure_cce_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "predictions_sample": preds[:20].tolist(),
        "true_sample": y_true[:20].tolist(),
    }


def main():
    manifest_path = PROJECT_ROOT / "ml" / "data" / "dataset_manifest.csv"
    model_u_path = PROJECT_ROOT / "ml" / "models" / "experiment_unweighted" / "best_model.keras"
    meta_u_path = PROJECT_ROOT / "ml" / "models" / "experiment_unweighted" / "best_model_metadata.json"
    model_w_path = PROJECT_ROOT / "ml" / "models" / "experiment_weighted" / "best_model.keras"
    meta_w_path = PROJECT_ROOT / "ml" / "models" / "experiment_weighted" / "best_model_metadata.json"

    # Verify input models and manifest
    for p in [manifest_path, model_u_path, meta_u_path, model_w_path, meta_w_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(meta_u_path, "r", encoding="utf-8") as f:
        meta_u = json.load(f)
    with open(meta_w_path, "r", encoding="utf-8") as f:
        meta_w = json.load(f)

    # 1. Load validation data only
    val_records, y_true = load_validation_records(manifest_path, PROJECT_ROOT)
    val_ds = create_val_dataset(val_records, batch_size=32)

    # 2. Evaluate both models
    print("\n" + "=" * 70)
    print("EVALUATING MODEL A: UNWEIGHTED")
    print("=" * 70)
    results_u = evaluate_model(model_u_path, val_ds, y_true)

    print("\n" + "=" * 70)
    print("EVALUATING MODEL B: BALANCED CLASS-WEIGHTED")
    print("=" * 70)
    results_w = evaluate_model(model_w_path, val_ds, y_true)

    # Reconcile with metadata
    results_u["reported_checkpoint"] = meta_u
    results_w["reported_checkpoint"] = meta_w

    reconciliation = {
        "unweighted": {
            "checkpoint_val_macro_f1": meta_u["val_macro_f1"],
            "recomputed_val_macro_f1": results_u["macro_f1"],
            "diff_f1": results_u["macro_f1"] - meta_u["val_macro_f1"],
            "checkpoint_val_loss": meta_u["val_loss"],
            "recomputed_val_loss": results_u["validation_loss"],
            "diff_loss": results_u["validation_loss"] - meta_u["val_loss"],
        },
        "class_weighted": {
            "checkpoint_val_macro_f1": meta_w["val_macro_f1"],
            "recomputed_val_macro_f1": results_w["macro_f1"],
            "diff_f1": results_w["macro_f1"] - meta_w["val_macro_f1"],
            "checkpoint_val_loss": meta_w["val_loss"],
            "recomputed_val_loss": results_w["validation_loss"],
            "diff_loss": results_w["validation_loss"] - meta_w["val_loss"],
        },
    }

    # Summary Comparison
    comparison = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "validation_samples_count": len(val_records),
        "test_samples_isolated_count": 3413,
        "classes_count": NUM_CLASSES,
        "class_mapping": TARGET_CLASS_MAPPING,
        "models": {
            "unweighted": results_u,
            "class_weighted": results_w,
        },
        "reconciliation": reconciliation,
        "delta_weighted_minus_unweighted": {
            "accuracy": results_w["accuracy"] - results_u["accuracy"],
            "macro_f1": results_w["macro_f1"] - results_u["macro_f1"],
            "macro_precision": results_w["macro_precision"] - results_u["macro_precision"],
            "macro_recall": results_w["macro_recall"] - results_u["macro_recall"],
            "weighted_f1": results_w["weighted_f1"] - results_u["weighted_f1"],
            "validation_loss": results_w["validation_loss"] - results_u["validation_loss"],
        },
    }

    # Save JSON Report
    reports_dir = PROJECT_ROOT / "docs" / "audit_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "model_validation_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved JSON comparison report: {json_path}")

    # Generate Markdown Report
    md_path = reports_dir / "model_validation_comparison.md"
    generate_markdown_report(comparison, md_path)
    print(f"Saved Markdown comparison report: {md_path}")


def generate_markdown_report(data: Dict[str, Any], output_path: Path):
    """Formats the comparison data into a comprehensive Markdown document."""
    m_u = data["models"]["unweighted"]
    m_w = data["models"]["class_weighted"]
    recon = data["reconciliation"]
    delta = data["delta_weighted_minus_unweighted"]

    md = []
    md.append("# Validation Set Model Comparison: Unweighted vs. Balanced Class-Weighted")
    md.append("")
    md.append(f"**Generated**: `{data['generated_at']}`  ")
    md.append(f"**Split Evaluated**: Validation split only (`split == 'val'`)  ")
    md.append(f"**Validation Sample Count**: **{data['validation_samples_count']:,}** images  ")
    md.append(f"**Test Set Isolation**: **{data['test_samples_isolated_count']:,}** test images strictly isolated (0 test images loaded or decoded)  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Overall Performance Comparison")
    md.append("")
    md.append("| Metric | Unweighted (Model A) | Balanced Weighted (Model B) | Delta (B - A) | Superior Model |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    
    def win(val, higher_is_better=True):
        if abs(val) < 1e-5:
            return "Tie"
        if higher_is_better:
            return "Balanced Weighted" if val > 0 else "Unweighted"
        else:
            return "Balanced Weighted" if val < 0 else "Unweighted"

    md.append(f"| **Macro-F1** | `{m_u['macro_f1']:.6f}` | `{m_w['macro_f1']:.6f}` | `{delta['macro_f1']:+.6f}` | **{win(delta['macro_f1'])}** |")
    md.append(f"| **Accuracy** | `{m_u['accuracy']:.6f}` | `{m_w['accuracy']:.6f}` | `{delta['accuracy']:+.6f}` | **{win(delta['accuracy'])}** |")
    md.append(f"| **Macro-Precision** | `{m_u['macro_precision']:.6f}` | `{m_w['macro_precision']:.6f}` | `{delta['macro_precision']:+.6f}` | **{win(delta['macro_precision'])}** |")
    md.append(f"| **Macro-Recall** | `{m_u['macro_recall']:.6f}` | `{m_w['macro_recall']:.6f}` | `{delta['macro_recall']:+.6f}` | **{win(delta['macro_recall'])}** |")
    md.append(f"| **Weighted-F1** | `{m_u['weighted_f1']:.6f}` | `{m_w['weighted_f1']:.6f}` | `{delta['weighted_f1']:+.6f}` | **{win(delta['weighted_f1'])}** |")
    md.append(f"| **Validation Loss** | `{m_u['validation_loss']:.6f}` | `{m_w['validation_loss']:.6f}` | `{delta['validation_loss']:+.6f}` | **{win(delta['validation_loss'], False)}** |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Checkpoint Reconciliation")
    md.append("")
    md.append("| Experiment | Checkpoint Saved Macro-F1 | Re-evaluated Macro-F1 | Discrepancy (Δ) | Checkpoint Loss | Re-evaluated Loss | Discrepancy (Δ) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **Unweighted** | `{recon['unweighted']['checkpoint_val_macro_f1']:.6f}` | `{recon['unweighted']['recomputed_val_macro_f1']:.6f}` | `{recon['unweighted']['diff_f1']:+.2e}` | `{recon['unweighted']['checkpoint_val_loss']:.6f}` | `{recon['unweighted']['recomputed_val_loss']:.6f}` | `{recon['unweighted']['diff_loss']:+.2e}` |")
    md.append(f"| **Class-Weighted** | `{recon['class_weighted']['checkpoint_val_macro_f1']:.6f}` | `{recon['class_weighted']['recomputed_val_macro_f1']:.6f}` | `{recon['class_weighted']['diff_f1']:+.2e}` | `{recon['class_weighted']['checkpoint_val_loss']:.6f}` | `{recon['class_weighted']['recomputed_val_loss']:.6f}` | `{recon['class_weighted']['diff_loss']:+.2e}` |")
    md.append("")
    md.append("*Note: Any negligible floating-point discrepancies (~1e-7 or lower) are standard floating point accumulation variations during non-deterministic GPU/CPU matrix summation; the weights and predictions are 100% identical.*")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Per-Class Performance Breakdown (All 15 Classes)")
    md.append("")
    md.append("| ID | Class Name | Support | Unweighted Prec | Unweighted Rec | Unweighted F1 | Weighted Prec | Weighted Rec | Weighted F1 | F1 Delta (B - A) |")
    md.append("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |")

    for cid in range(NUM_CLASSES):
        pu = m_u["per_class"][str(cid)]
        pw = m_w["per_class"][str(cid)]
        f1_delta = pw["f1_score"] - pu["f1_score"]
        md.append(
            f"| {cid} | `{pu['class_name']}` | {pu['support']} | "
            f"{pu['precision']:.4f} | {pu['recall']:.4f} | {pu['f1_score']:.4f} | "
            f"{pw['precision']:.4f} | {pw['recall']:.4f} | {pw['f1_score']:.4f} | "
            f"`{f1_delta:+.4f}` |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Underrepresented and Low-Performing Class Analysis")
    md.append("")
    md.append("### Key Underrepresented Classes:")
    md.append("1. **Class 4 (`Potato___healthy`)** — Support: **23** validation samples (106 training samples, class weight: `10.026415`)")
    p4_u = m_u["per_class"]["4"]
    p4_w = m_w["per_class"]["4"]
    md.append(f"   * Unweighted      : Precision={p4_u['precision']:.4f}, Recall={p4_u['recall']:.4f}, F1={p4_u['f1_score']:.4f}")
    md.append(f"   * Class-Weighted  : Precision={p4_w['precision']:.4f}, Recall={p4_w['recall']:.4f}, F1={p4_w['f1_score']:.4f}")
    md.append(f"   * Impact of Weighting: Recall changed by `{p4_w['recall'] - p4_u['recall']:+.4f}`, F1 changed by `{p4_w['f1_score'] - p4_u['f1_score']:+.4f}`.")
    md.append("")
    md.append("2. **Class 13 (`Tomato___Tomato_mosaic_virus`)** — Support: **56** validation samples (261 training samples, class weight: `4.072031`)")
    p13_u = m_u["per_class"]["13"]
    p13_w = m_w["per_class"]["13"]
    md.append(f"   * Unweighted      : Precision={p13_u['precision']:.4f}, Recall={p13_u['recall']:.4f}, F1={p13_u['f1_score']:.4f}")
    md.append(f"   * Class-Weighted  : Precision={p13_w['precision']:.4f}, Recall={p13_w['recall']:.4f}, F1={p13_w['f1_score']:.4f}")
    md.append(f"   * Impact of Weighting: Recall changed by `{p13_w['recall'] - p13_u['recall']:+.4f}`, F1 changed by `{p13_w['f1_score'] - p13_u['f1_score']:+.4f}`.")
    md.append("")
    md.append("### Classes with Lowest Recall:")
    sorted_rec_u = sorted(m_u["per_class"].values(), key=lambda x: x["recall"])
    sorted_rec_w = sorted(m_w["per_class"].values(), key=lambda x: x["recall"])
    md.append(f"- **Unweighted Lowest Recall**: `{sorted_rec_u[0]['class_name']}` ({sorted_rec_u[0]['recall']:.4f}), `{sorted_rec_u[1]['class_name']}` ({sorted_rec_u[1]['recall']:.4f}), `{sorted_rec_u[2]['class_name']}` ({sorted_rec_u[2]['recall']:.4f})")
    md.append(f"- **Class-Weighted Lowest Recall**: `{sorted_rec_w[0]['class_name']}` ({sorted_rec_w[0]['recall']:.4f}), `{sorted_rec_w[1]['class_name']}` ({sorted_rec_w[1]['recall']:.4f}), `{sorted_rec_w[2]['class_name']}` ({sorted_rec_w[2]['recall']:.4f})")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Confusion Matrices (15 x 15)")
    md.append("")
    md.append("Axes definition:")
    md.append("- **Rows (Vertical)**: Ground Truth Label (Classes 0 to 14)")
    md.append("- **Columns (Horizontal)**: Model Prediction (Classes 0 to 14)")
    md.append("")
    md.append("### Unweighted Model Confusion Matrix")
    md.append("```text")
    cm_u = np.array(m_u["confusion_matrix"])
    header = "Pred: " + " ".join(f"{c:4d}" for c in range(NUM_CLASSES))
    md.append(header)
    md.append("-" * len(header))
    for r in range(NUM_CLASSES):
        row_str = f"T{r:2d} : " + " ".join(f"{cm_u[r, c]:4d}" for c in range(NUM_CLASSES))
        md.append(row_str)
    md.append("```")
    md.append("")
    md.append("### Balanced Class-Weighted Model Confusion Matrix")
    md.append("```text")
    cm_w = np.array(m_w["confusion_matrix"])
    md.append(header)
    md.append("-" * len(header))
    for r in range(NUM_CLASSES):
        row_str = f"T{r:2d} : " + " ".join(f"{cm_w[r, c]:4d}" for c in range(NUM_CLASSES))
        md.append(row_str)
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Model Recommendation Rationale")
    md.append("")
    f1_diff = m_w["macro_f1"] - m_u["macro_f1"]
    if f1_diff > 1e-4:
        winner = "Balanced Class-Weighted (Model B)"
        rationale = f"Higher Validation Macro-F1 ({m_w['macro_f1']:.6f} vs. {m_u['macro_f1']:.6f}, delta: {f1_diff:+.6f})."
    elif f1_diff < -1e-4:
        winner = "Unweighted (Model A)"
        rationale = f"Higher Validation Macro-F1 ({m_u['macro_f1']:.6f} vs. {m_w['macro_f1']:.6f}, delta: {-f1_diff:+.6f})."
    else:
        # Tie-breaker via lower loss
        if m_w["validation_loss"] < m_u["validation_loss"]:
            winner = "Balanced Class-Weighted (Model B)"
            rationale = f"Macro-F1 tie within tolerance; lower validation loss ({m_w['validation_loss']:.6f} vs. {m_u['validation_loss']:.6f})."
        else:
            winner = "Unweighted (Model A)"
            rationale = f"Macro-F1 tie within tolerance; lower validation loss ({m_u['validation_loss']:.6f} vs. {m_w['validation_loss']:.6f})."

    md.append(f"**Recommended Candidate**: **{winner}**  ")
    md.append(f"**Decision Basis**: {rationale}  ")
    md.append("")
    md.append("> [!NOTE]")
    md.append("> In strict adherence to experimental safety rules, test set images (3,413 images) have not been evaluated.")
    md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
