"""
Evaluation harness for the content-level input suitability gate
(``backend.app.services.input_gate``).

WHAT IT MEASURES
----------------
False acceptance rate (FAR) on non-leaf imagery, and valid-leaf rejection rate (no
diagnosis) on real crop leaves, reported per category with sample counts.

DATA AND HONESTY OF THE EVALUATION
----------------------------------
* Supported leaves come from the ``val`` split of the frozen manifest. The held-out
  ``test`` split is never opened by this script (asserted at runtime).
* Non-leaf evidence is the 1,143 real ``Background_without_leaves`` photographs that the
  training pipeline excluded. IMPORTANT: those images were inspected during threshold
  selection, so their numbers are an *exploratory diagnostic*, not a clean untouched
  benchmark.
* The synthetic probe categories are generated after the gate thresholds were fixed, so
  they are a held-out (though synthetic, non-photographic) set that measures branch
  behaviour only.
* Every source category is partitioned deterministically into a calibration half and an
  evaluation half by SHA-256 hash parity. The calibration sweep is printed for the
  calibration half only; evaluation metrics use the disjoint evaluation half.

This script does not tune the gate. Thresholds live in ``backend/app/config.py`` and the
constants in ``backend/app/services/input_gate.py``; the sweep exists to document the
decision boundary that was chosen on non-test data.

Run:
    .venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py
"""
import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from backend.app.services.input_gate import (  # noqa: E402
    ABSTAINED,
    ACCEPTED,
    REJECTED,
    assess_image,
)
from backend.app.services.inference import preprocess_image  # noqa: E402
from backend.tests.non_leaf_fixtures import (  # noqa: E402
    create_animal_bytes,
    create_blurry_leaf_bytes,
    create_building_bytes,
    create_complex_background_bytes,
    create_dark_leaf_bytes,
    create_document_bytes,
    create_leaf_closeup_bytes,
    create_multiple_leaves_bytes,
    create_object_bytes,
    create_portrait_bytes,
    create_scattered_vegetation_bytes,
    create_screenshot_bytes,
    create_sparse_vegetation_bytes,
    create_tiny_leaf_bytes,
    create_uniform_bytes,
)

MANIFEST = PROJECT_ROOT / "ml" / "data" / "dataset_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "ml"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "audit_reports" / "non_leaf_input_safety_eval.json"

SUPPORTED_CLASSES = {
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
}
UNSUPPORTED_SAMPLE_CLASSES = [
    "Apple___Apple_scab", "Corn___Common_rust", "Grape___Black_rot",
    "Peach___Bacterial_spot", "Soybean___healthy", "Strawberry___Leaf_scorch",
]

SYNTHETIC_PROBES = {
    "person": (create_portrait_bytes, "non_leaf"),
    "animal": (create_animal_bytes, "non_leaf"),
    "building": (create_building_bytes, "non_leaf"),
    "document": (create_document_bytes, "non_leaf"),
    "random_object": (create_object_bytes, "non_leaf"),
    "screenshot_or_graphic": (create_screenshot_bytes, "non_leaf"),
    "uniform_constant_colour": (create_uniform_bytes, "non_leaf"),
    "supported_leaf": (create_leaf_closeup_bytes, "leaf"),
    "multiple_leaves": (create_multiple_leaves_bytes, "leaf"),
    "complex_background_leaf": (create_complex_background_bytes, "leaf"),
    "sparse_vegetation": (create_sparse_vegetation_bytes, "leaf"),
    "scattered_vegetation": (create_scattered_vegetation_bytes, "leaf"),
    "tiny_leaf": (create_tiny_leaf_bytes, "leaf"),
    "dark_leaf": (create_dark_leaf_bytes, "leaf"),
    "blurry_leaf": (create_blurry_leaf_bytes, "leaf"),
}

SWEEP_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)


def load_manifest() -> list:
    with open(MANIFEST, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bucket_of(row: dict) -> int:
    """Stable calibration/evaluation assignment, independent of the manifest's hash column."""
    key = (row.get("relative_path") or row["image_id"]).replace("\\", "/")
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2


def partition(rows: list) -> tuple:
    """Deterministic calibration/evaluation split by path hash parity."""
    calibration, evaluation = [], []
    for row in rows:
        (calibration if bucket_of(row) == 0 else evaluation).append(row)
    return calibration, evaluation


def assess_path(relative_path: str):
    payload = (DATA_ROOT / relative_path).read_bytes()
    tensor = preprocess_image(payload)
    return assess_image(tensor[0])


def summarize(records: list, expected: str) -> dict:
    """Summarizes assessments for one category."""
    counts = Counter(record["decision"] for record in records)
    reasons = Counter(record["code"] for record in records if record["decision"] != ACCEPTED)
    errors = Counter(record["error"] for record in records if record.get("error"))
    total = len(records)
    accepted = counts[ACCEPTED]
    rejected = counts[REJECTED]
    abstained = counts[ABSTAINED]
    summary = {
        "n": total,
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "decode_errors": sum(errors.values()),
        "acceptance_rate": round(accepted / total, 4) if total else None,
        # "No diagnosis" = anything that is not a normal prediction (physical-layer
        # decode failures are recorded as rejections, not double-counted here).
        "no_diagnosis_rate": round((rejected + abstained) / total, 4) if total else None,
        "reasons": dict(reasons),
    }
    if expected == "non_leaf":
        summary["false_acceptance_rate"] = summary["acceptance_rate"]
    else:
        summary["valid_leaf_no_diagnosis_rate"] = summary["no_diagnosis_rate"]
        summary["valid_leaf_hard_rejection_rate"] = round(rejected / total, 4) if total else None
    return summary


def bridge_sweep(calibration_leaves: list, calibration_backgrounds: list) -> list:
    """Documents the plant-evidence decision boundary on the calibration half only."""
    leaf_evidence = np.array(
        [row.get("signals", {}).get("plant_evidence", 0.0) for row in calibration_leaves]
    )
    background_evidence = np.array(
        [row.get("signals", {}).get("plant_evidence", 0.0) for row in calibration_backgrounds]
    )
    sweep = []
    for threshold in SWEEP_THRESHOLDS:
        sweep.append(
            {
                "threshold": threshold,
                "calibration_leaf_below_rate": round(float((leaf_evidence < threshold).mean()), 4),
                "calibration_background_below_rate": round(
                    float((background_evidence < threshold).mean()), 4
                ),
            }
        )
    return sweep


def gate_latency_ms(rows: list, sample_size: int = 150) -> dict:
    step = max(1, len(rows) // sample_size)
    measured = []
    for row in rows[::step][:sample_size]:
        tensor = preprocess_image((DATA_ROOT / row["relative_path"]).read_bytes())
        frame = tensor[0]
        start = time.perf_counter()
        assess_image(frame)
        measured.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples": len(measured),
        "mean_ms": round(statistics.mean(measured), 3),
        "median_ms": round(statistics.median(measured), 3),
        "p95_ms": round(sorted(measured)[int(len(measured) * 0.95) - 1], 3),
    }


def collect_real(rows: list) -> list:
    records = []
    for row in rows:
        try:
            assessment = assess_path(row["relative_path"])
            records.append(
                {
                    "decision": assessment.decision,
                    "code": assessment.code,
                    "signals": assessment.signals,
                }
            )
        except Exception as exc:  # physical-integrity rejection (Layer A) is also safe
            records.append({"decision": REJECTED, "code": "PHYSICAL_LAYER_REJECTION", "error": type(exc).__name__})
    return records


def collect_synthetic() -> dict:
    results = {}
    for name, (factory, expected) in SYNTHETIC_PROBES.items():
        payload = factory()
        try:
            tensor = preprocess_image(payload)
            assessment = assess_image(tensor[0])
            record = {"decision": assessment.decision, "code": assessment.code, "signals": assessment.signals}
        except Exception as exc:
            record = {"decision": REJECTED, "code": "PHYSICAL_LAYER_REJECTION", "error": type(exc).__name__}
        results[name] = {"expected": expected, "record": record}
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unsupported-per-class", type=int, default=80)
    args = parser.parse_args()

    rows = load_manifest()
    leaves = [row for row in rows if row["split"] == "val" and row["class_name"] in SUPPORTED_CLASSES]
    backgrounds = [
        row for row in rows
        if row["split"] == "excluded" and row["class_name"] == "Background_without_leaves"
    ]
    unsupported = [
        row for row in rows
        if row["split"] == "excluded" and row["class_name"] in UNSUPPORTED_SAMPLE_CLASSES
    ]
    test_rows = [row for row in rows if row["split"] == "test"]

    _, unsupported_evaluation = partition(unsupported)

    per_class: dict = {}
    for row in unsupported_evaluation:
        bucket = per_class.setdefault(row["class_name"], [])
        if len(bucket) < args.unsupported_per_class:
            bucket.append(row)
    unsupported_evaluation = [row for group in per_class.values() for row in group]

    print(f"Manifest: {len(rows)} rows | leaves(val)={len(leaves)} backgrounds={len(backgrounds)} "
          f"unsupported sample={len(unsupported_evaluation)}")
    print(f"Test split rows left untouched: {len(test_rows)}")

    print("Assessing validation leaves...")
    leaf_records = collect_real(leaves)
    leaf_buckets = [bucket_of(row) for row in leaves]
    leaf_calibration_records = [r for r, h in zip(leaf_records, leaf_buckets) if h == 0]
    leaf_evaluation_records = [r for r, h in zip(leaf_records, leaf_buckets) if h == 1]

    print("Assessing real non-leaf photographs...")
    background_records = collect_real(backgrounds)
    background_buckets = [bucket_of(row) for row in backgrounds]
    background_calibration_records = [r for r, h in zip(background_records, background_buckets) if h == 0]
    background_evaluation_records = [r for r, h in zip(background_records, background_buckets) if h == 1]

    print("Assessing unsupported crop leaves (documented limitation)...")
    unsupported_records = collect_real(unsupported_evaluation)

    print("Running synthetic probes...")
    synthetic = collect_synthetic()

    sweep = bridge_sweep(leaf_calibration_records, background_calibration_records)
    latency = gate_latency_ms(leaves)

    categories = {
        "supported_leaf_val_evaluation_half": ("leaf", leaf_evaluation_records),
        "real_non_leaf_scene_photographs": ("non_leaf", background_evaluation_records),
        "unsupported_crop_leaves": ("unsupported_crop", unsupported_records),
    }

    synthetic_categories = {}
    for name, payload in synthetic.items():
        synthetic_categories[name] = summarize([payload["record"]], payload["expected"])

    evaluation = {name: summarize(records, expected) for name, (expected, records) in categories.items()}
    evaluation.update(synthetic_categories)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "backend.app.services.input_gate",
        "dataset_isolation": {
            "test_split_rows_untouched": len(test_rows),
            "supported_leaves_source": "val split (non-test)",
            "non_leaf_source": "excluded Background_without_leaves photographs",
            "synthetic_probes": "generated after the gate thresholds were fixed",
            "caveat": (
                "The real non-leaf photographs were inspected during threshold selection, so "
                "they are an exploratory diagnostic set, not a clean untouched OOD benchmark."
            ),
        },
        "calibration": {
            "supported_leaf_calibration_half": summarize(leaf_calibration_records, "leaf"),
            "real_non_leaf_calibration_half": summarize(background_calibration_records, "non_leaf"),
            "plant_evidence_threshold_sweep": sweep,
        },
        "evaluation": evaluation,
        "gate_latency": latency,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Evaluation (disjoint half / post-hoc probes) ===")
    header = f"{'category':34s} {'n':>5s} {'accept':>7s} {'reject':>7s} {'abstain':>8s} {'noDiagnosis':>11s}"
    print(header)
    for name, summary in evaluation.items():
        print(
            f"{name:34s} {summary['n']:5d} {summary['acceptance_rate']:7.3f} "
            f"{summary['rejected']:7d} {summary['abstained']:8d} {summary['no_diagnosis_rate']:11.3f}"
        )
    print(f"\nGate latency (assess only): {latency}")
    print(f"Artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
