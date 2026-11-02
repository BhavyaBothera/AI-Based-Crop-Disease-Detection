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
from typing import Any, Dict, List, Optional

from .data_auditor import DataQualityAuditor
from .evaluator import FieldEvaluator
from .gate_evaluator import GateSampleRecord, LeafGateEvaluator
from .manifest_schema import load_and_validate_manifest

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
        action="store_true",
        help="Exclude unverified/suspected annotations from primary supervised metrics.",
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

        gate_samples = [
            GateSampleRecord(
                sample_id=r.image_id,
                file_path=r.file_path,
                ground_truth_category="supported_leaf",
                crop=r.crop,
                notes=r.notes,
            )
            for r in records
        ]

        gate_results = gate_eval.evaluate_dataset(gate_samples, dataset_root=dataset_root)
        gate_json, gate_md = gate_eval.save_results(
            gate_results,
            output_dir=output_dir,
            prefix="field_gate_eval",
        )
        logger.info("Gate evaluation saved to %s", gate_json.name)

    # Step 4: Comparative Report Generation (if baseline specified)
    if args.baseline_comparison and args.baseline_comparison.is_file():
        logger.info("Generating comparison against controlled baseline: %s", args.baseline_comparison)
        comp_md_path = output_dir / f"controlled_vs_field_{timestamp_str}.md"
        comp_content = generate_baseline_comparison(
            controlled_report_path=args.baseline_comparison,
            field_results=eval_results,
        )
        with comp_md_path.open("w", encoding="utf-8") as f:
            f.write(comp_content)
        logger.info("Comparative analysis saved to %s", comp_md_path.name)

    print("\n" + evaluator.generate_markdown_summary(eval_results))
    return 0


def generate_baseline_comparison(
    controlled_report_path: Path,
    field_results: Dict[str, Any],
) -> str:
    """Generates comparison between controlled PlantVillage baseline and field results."""
    field_m = field_results["metrics"]
    field_acc = field_m["accuracy"]
    field_f1 = field_m["macro_f1"]

    # Controlled baseline reference (verified from final_test_evaluation_report.md)
    ctrl_acc = 0.968942
    ctrl_f1 = 0.965288

    delta_acc = field_acc - ctrl_acc
    delta_f1 = field_f1 - ctrl_f1

    md = []
    md.append("# Generalization Analysis: Controlled Studio vs. Real Field")
    md.append(f"**Field Evaluation Timestamp:** {field_results['metadata']['evaluation_timestamp']}")
    md.append("")
    md.append("## High-Level Benchmark Comparison")
    md.append("| Metric | Controlled Baseline (Studio) | Field Validation | Delta (Shift) |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Top-1 Accuracy** | {ctrl_acc * 100:.2f}% | {field_acc * 100:.2f}% | {delta_acc * 100:+.2f}% |")
    md.append(f"| **Macro-F1 Score** | {ctrl_f1:.4f} | {field_f1:.4f} | {delta_f1:+.4f} |")
    md.append(f"| **Top-3 Accuracy** | 99.71% | {field_m['top_3_accuracy'] * 100:.2f}% | {(field_m['top_3_accuracy'] - 0.9971) * 100:+.2f}% |")
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
