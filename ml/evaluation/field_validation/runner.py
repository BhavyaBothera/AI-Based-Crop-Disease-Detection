"""
Field Evaluation CLI Runner Orchestration
==========================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Orchestrates:
1. Manifest schema and leakage audit (`data_auditor.py`)
2. Standalone disease classification evaluation (`evaluator.py`)
3. Leaf acceptance gate evaluation (`gate_evaluator.py`)
4. Baseline comparison against controlled test split
5. Non-destructive report generation (JSON + Markdown)

Usage:
  python -m ml.evaluation.field_validation.runner --manifest path/to/field_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .data_auditor import DataQualityAuditor
from .evaluator import FieldEvaluator
from .gate_evaluator import GateSampleRecord, LeafGateEvaluator
from .manifest_schema import (
    load_and_validate_manifest,
    map_ground_truth_to_gate_category,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("field_validation.runner")

DEFAULT_MODEL_PATH = Path("ml/models/experiment_weighted/best_model.keras")
DEFAULT_GATE_PATH = Path("ml/models/input_gate/leaf_gate.keras")
DEFAULT_OUTPUT_DIR = Path("docs/audit_reports/field_validation_runs")
DEFAULT_CONTROLLED_REPORT = Path("docs/audit_reports/final_test_evaluation_report.md")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PHYTOSIGNAL 2.0 Field Validation & Generalization Audit Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the field evaluation manifest CSV.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Directory containing field image files (defaults to manifest parent).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained disease classification Keras model.",
    )
    parser.add_argument(
        "--gate-path",
        type=Path,
        default=DEFAULT_GATE_PATH,
        help="Path to the trained leaf gate Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation run artifacts will be written.",
    )
    parser.add_argument(
        "--split",
        choices=["field_dev", "field_test"],
        default=None,
        help="Evaluate only a specific manifest partition.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run data quality, integrity, and leakage audit without ML inference.",
    )
    parser.add_argument(
        "--eval-gate",
        action="store_true",
        help="Also evaluate leaf-acceptance gate performance on the manifest samples.",
    )
    parser.add_argument(
        "--exclude-unverified",
        dest="exclude_unverified",
        action="store_true",
        default=True,
        help="Exclude unverified/suspected annotations from primary supervised metrics (default: True).",
    )
    parser.add_argument(
        "--include-unverified",
        dest="exclude_unverified",
        action="store_false",
        help="Include unverified/suspected annotations in primary supervised metrics (default: False; unverified samples are excluded by default).",
    )
    parser.add_argument(
        "--baseline-comparison",
        type=Path,
        default=None,
        help="Path to controlled baseline evaluation report or metrics JSON to compare with.",
    )
    return parser.parse_args(argv)


def run_field_validation(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    dataset_root = (args.dataset_root or manifest_path.parent).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Step 1: Data Quality & Leakage Audit
    logger.info("Executing Data Quality & Leakage Audit...")
    auditor = DataQualityAuditor(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    audit_report = auditor.audit_all()
    audit_summary = auditor.generate_markdown_report(audit_report)

    # Save audit report
    audit_json_path = output_dir / f"audit_report_{timestamp_str}.json"
    audit_md_path = output_dir / f"audit_report_{timestamp_str}.md"
    with audit_json_path.open("w", encoding="utf-8") as f:
        json.dump(audit_report.to_dict(), f, indent=2)
    with audit_md_path.open("w", encoding="utf-8") as f:
        f.write(audit_summary)

    logger.info("Data audit saved to %s and %s", audit_json_path.name, audit_md_path.name)

    if not audit_report.is_clean:
        logger.warning(
            "Data audit identified issues: %d corruptions, %d format errors, %d cross-split leaks, %d controlled leaks.",
            len(audit_report.corrupt_images),
            len(audit_report.unsupported_format_images),
            len(audit_report.group_leaks),
            len(audit_report.controlled_dataset_leaks),
        )

    if args.audit_only:
        print("\n" + audit_summary)
        return 0

    # Step 2: Disease Model Evaluation
    model_path = args.model_path.resolve()
    if not model_path.is_file():
        logger.error("Model checkpoint not found: %s", model_path)
        return 1

    logger.info("Initializing FieldEvaluator with model: %s", model_path)
    evaluator = FieldEvaluator(model_path=model_path)
    eval_results = evaluator.evaluate_manifest(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split_filter=args.split,
        exclude_unverified_from_primary=args.exclude_unverified,
    )

    eval_json, eval_md = evaluator.save_results(
        eval_results,
        output_dir=output_dir,
        prefix="field_disease_eval",
    )

    # Step 3: Leaf Gate Evaluation (if requested)
    gate_results = None
    if args.eval_gate:
        gate_path = args.gate_path.resolve()
        logger.info("Initializing LeafGateEvaluator with gate model: %s", gate_path)
        gate_eval = LeafGateEvaluator(gate_model_path=gate_path)

        records, _ = load_and_validate_manifest(manifest_path)
        if args.split:
            records = [r for r in records if r.split == args.split]

        gate_records = [r for r in records if getattr(r, "evaluation_task", "both") in ("gate", "both")]
        gate_samples: List[GateSampleRecord] = []
        unmapped_gate_records: List[Dict[str, Any]] = []

        for r in gate_records:
            try:
                gate_cat = map_ground_truth_to_gate_category(r.ground_truth_class, r.crop)
                gate_samples.append(
                    GateSampleRecord(
                        sample_id=r.image_id,
                        file_path=r.file_path,
                        ground_truth_category=gate_cat,
                        crop=r.crop,
                        notes=r.notes,
                    )
                )
            except ValueError as exc:
                unmapped_gate_records.append({
                    "sample_id": r.image_id,
                    "ground_truth_class": r.ground_truth_class,
                    "crop": r.crop,
                    "error": str(exc),
                })

        if unmapped_gate_records:
            logger.warning(
                "Skipping %d gate records with unmapped or ambiguous categories: %s",
                len(unmapped_gate_records),
                [u["sample_id"] for u in unmapped_gate_records[:5]],
            )

        gate_results = gate_eval.evaluate_dataset(gate_samples, dataset_root=dataset_root)
        if unmapped_gate_records:
            gate_results["metadata"]["unmapped_gate_records"] = unmapped_gate_records

        gate_json, gate_md = gate_eval.save_results(
            gate_results,
            output_dir=output_dir,
            prefix="field_gate_eval",
        )
        logger.info("Gate evaluation saved to %s", gate_json.name)

    # Step 4: Comparative Report Generation (if baseline specified)
    if args.baseline_comparison:
        baseline_path = args.baseline_comparison.resolve()
        logger.info("Generating comparison against controlled baseline: %s", baseline_path)
        comp_md_path = output_dir / f"controlled_vs_field_{timestamp_str}.md"
        comp_content = generate_baseline_comparison(
            controlled_report_path=baseline_path,
            field_results=eval_results,
        )
        with comp_md_path.open("w", encoding="utf-8") as f:
            f.write(comp_content)
        logger.info("Comparative analysis saved to %s", comp_md_path.name)

    print("\n" + evaluator.generate_markdown_summary(eval_results))
    return 0


def parse_controlled_baseline(controlled_report_path: Union[Path, str]) -> Dict[str, Any]:
    """
    Parses a controlled baseline evaluation report (.json or .md).

    Extracts:
    - accuracy: float in [0.0, 1.0]
    - macro_f1: float in [0.0, 1.0]
    - top_3_accuracy: Optional[float]
    - source_file: str
    - model_checkpoint: str
    - timestamp: str

    Raises:
    - FileNotFoundError if the baseline file does not exist.
    - ValueError if the baseline file is malformed, missing required metrics, or unparseable.
    """
    p = Path(controlled_report_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Controlled baseline report not found: {p}")

    content = p.read_text(encoding="utf-8")
    acc: Optional[float] = None
    f1: Optional[float] = None
    top_3: Optional[float] = None
    timestamp: str = "Unknown"
    provenance: str = "Unknown"

    if p.suffix.lower() == ".json":
        try:
            data = json.loads(content)
        except Exception as exc:
            raise ValueError(f"Failed to parse baseline JSON file at {p}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Baseline JSON at {p} must be a dictionary.")

        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else data

        for key in ("test_accuracy", "accuracy", "val_accuracy"):
            if key in metrics and metrics[key] is not None:
                try:
                    acc = float(metrics[key])
                    break
                except (ValueError, TypeError):
                    pass

        for key in ("test_macro_f1", "macro_f1", "val_macro_f1"):
            if key in metrics and metrics[key] is not None:
                try:
                    f1 = float(metrics[key])
                    break
                except (ValueError, TypeError):
                    pass

        for key in ("test_top_3_accuracy", "top_3_accuracy", "val_top_3_accuracy"):
            if key in metrics and metrics[key] is not None:
                try:
                    top_3 = float(metrics[key])
                    break
                except (ValueError, TypeError):
                    pass

        timestamp = str(data.get("timestamp_utc") or data.get("evaluation_date") or data.get("timestamp") or "Unknown")
        provenance = str(data.get("model_checkpoint") or data.get("model_path") or data.get("model_name") or "Unknown")

    else:
        import re

        f1_m = re.search(r"\|\s*\*\*Macro-F1(?:\s*\(Primary\))?\*\*\s*\|\s*(?:\*\*)?`?([0-9.]+)`?", content, re.IGNORECASE)
        if f1_m:
            try:
                f1 = float(f1_m.group(1))
            except ValueError:
                pass

        acc_m = re.search(r"\|\s*\*\*Accuracy\*\*\s*\|\s*(?:\*\*)?`?([0-9.]+)`?", content, re.IGNORECASE)
        if acc_m:
            try:
                acc = float(acc_m.group(1))
            except ValueError:
                pass

        top3_m = re.search(r"\|\s*\*\*Top-3\s+Accuracy\*\*\s*\|\s*(?:\*\*)?`?([0-9.]+)`?", content, re.IGNORECASE)
        if top3_m:
            try:
                top_3 = float(top3_m.group(1))
            except ValueError:
                pass

        date_m = re.search(r"\*\*(?:Evaluation\s+Date|Timestamp)\*\*:\s*`?([^`\r\n]+)`?", content, re.IGNORECASE)
        if date_m:
            timestamp = date_m.group(1).strip()

        model_m = re.search(r"\*\*(?:Selected\s+Model|Model\s+Checkpoint)\*\*:\s*(?:\[`?([^`\]\r\n]+)`?\]|`?([^`\r\n]+)`?)", content, re.IGNORECASE)
        if model_m:
            provenance = (model_m.group(1) or model_m.group(2) or "Unknown").strip()

    if acc is None or f1 is None:
        raise ValueError(
            f"Baseline file at {p} is malformed or missing required metrics ('accuracy', 'macro_f1'). "
            f"Found accuracy={acc}, macro_f1={f1}."
        )

    if not (0.0 <= acc <= 1.0) or not (0.0 <= f1 <= 1.0):
        raise ValueError(
            f"Parsed baseline metrics outside valid range [0.0, 1.0]: accuracy={acc}, macro_f1={f1} in {p}."
        )

    if top_3 is not None and not (0.0 <= top_3 <= 1.0):
        top_3 = None

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "top_3_accuracy": top_3,
        "source_file": str(p),
        "timestamp": timestamp,
        "model_checkpoint": provenance,
    }


def generate_baseline_comparison(
    controlled_report_path: Union[Path, str],
    field_results: Dict[str, Any],
) -> str:
    """Generates comparison between parsed controlled baseline and field results."""
    baseline = parse_controlled_baseline(controlled_report_path)

    field_m = field_results["metrics"]
    field_acc = field_m["accuracy"]
    field_f1 = field_m["macro_f1"]
    field_top3 = field_m.get("top_3_accuracy")

    ctrl_acc = baseline["accuracy"]
    ctrl_f1 = baseline["macro_f1"]
    ctrl_top3 = baseline["top_3_accuracy"]

    delta_acc = field_acc - ctrl_acc
    delta_f1 = field_f1 - ctrl_f1

    md = []
    md.append("# Generalization Analysis: Controlled Baseline vs. Real Field")
    md.append(f"**Field Evaluation Timestamp:** {field_results['metadata']['evaluation_timestamp']}")
    md.append(f"**Controlled Baseline Source:** `{baseline['source_file']}`")
    md.append(f"**Controlled Model Checkpoint:** `{baseline['model_checkpoint']}`")
    md.append(f"**Controlled Evaluation Date:** `{baseline['timestamp']}`")
    md.append("")
    md.append("## High-Level Benchmark Comparison")
    md.append("| Metric | Controlled Baseline | Field Validation | Delta (Shift) |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Top-1 Accuracy** | {ctrl_acc * 100:.2f}% | {field_acc * 100:.2f}% | {delta_acc * 100:+.2f}% |")
    md.append(f"| **Macro-F1 Score** | {ctrl_f1:.4f} | {field_f1:.4f} | {delta_f1:+.4f} |")

    if ctrl_top3 is not None and field_top3 is not None:
        delta_top3 = field_top3 - ctrl_top3
        md.append(f"| **Top-3 Accuracy** | {ctrl_top3 * 100:.2f}% | {field_top3 * 100:.2f}% | {delta_top3 * 100:+.2f}% |")
    elif field_top3 is not None:
        md.append(f"| **Top-3 Accuracy** | N/A | {field_top3 * 100:.2f}% | N/A |")

    md.append("")
    md.append("## Distribution Shift Drivers")
    md.append("The observed generalization delta cannot be attributed solely to background differences. Real-world degradation stems from:")
    md.append("1. **Optical Acquisition Variations:** Sensor noise, autofocus drift, hand shake, rolling shutter.")
    md.append("2. **Photometric Variability:** Uncontrolled solar azimuth, leaf specular reflections, overcast cast.")
    md.append("3. **Pathology Progression:** Field lesions vary continuously in severity, unlike curated museum specimens.")
    md.append("4. **Background Clutter:** Soil, irrigation lines, weed foliage, shadows.")
    md.append("")
    return "\n".join(md)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(run_field_validation(parse_args()))


if __name__ == "__main__":
    main()
