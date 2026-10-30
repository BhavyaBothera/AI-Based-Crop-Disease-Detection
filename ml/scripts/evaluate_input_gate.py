"""
Held-out evaluation of the PHYTOSIGNAL leaf-acceptance gate.

WHAT IT MEASURES
----------------
The *deployed* artefact (``ml/models/input_gate/leaf_gate.keras``) end to end, through the
same ``assess_image`` decision function the API uses, on sets that were never used to fit
the classifier:

    supported leaves      disease `val` split, path-hash eval half (never seen in training)
    real non-leaf scenes  1,143 leaf-free field photographs, eval slice
    people / animals / buildings / documents / objects / graphics
                          downloaded real photographs, eval slice (never in training)
    unsupported species   apple, corn, grape, ... eval slice (seen species, unseen images)
    unseen species        Strawberry + Soybean, held out ENTIRELY from training
    synthetic probes      programmatic person/document/screenshot/leaf shapes, written after
                          the model was trained
    quality edge cases    dark, blurry, tiny, multiple-leaf probes

The disease `test` split is never opened by this script.

Reported per category with sample counts: acceptance rate, hard-rejection rate, abstention
rate and "no diagnosis" rate. For non-leaf categories the acceptance rate IS the false
acceptance rate; for leaf categories the no-diagnosis rate IS the valid-leaf rejection rate.

Run:
    .venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from backend.app.config import LEAF_GATE_MIN_CONFIDENCE, LEAF_GATE_MODEL_PATH  # noqa: E402
from backend.app.services.input_gate import (  # noqa: E402
    ABSTAINED,
    ACCEPTED,
    REJECTED,
    assess_image,
)
from backend.app.services.inference import preprocess_image  # noqa: E402
from backend.app.services.leaf_gate import LeafGateService  # noqa: E402
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
OOD_MANIFEST = PROJECT_ROOT / "ml" / "data" / "ood_corpus" / "manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "audit_reports" / "non_leaf_input_safety_eval.json"
FEATURE_DIR = PROJECT_ROOT / "ml" / "data" / "gate_features"

SUPPORTED_CLASSES = {
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
}
HELD_OUT_SPECIES_PREFIXES = ("Strawberry___", "Soybean___")
BACKGROUND_CLASS = "Background_without_leaves"

SYNTHETIC_PROBES = {
    "synthetic_person": (create_portrait_bytes, "non_leaf"),
    "synthetic_animal": (create_animal_bytes, "non_leaf"),
    "synthetic_building": (create_building_bytes, "non_leaf"),
    "synthetic_document": (create_document_bytes, "non_leaf"),
    "synthetic_random_object": (create_object_bytes, "non_leaf"),
    "synthetic_screenshot_or_graphic": (create_screenshot_bytes, "non_leaf"),
    "synthetic_uniform_canvas": (create_uniform_bytes, "non_leaf"),
    "synthetic_supported_leaf": (create_leaf_closeup_bytes, "leaf"),
    "synthetic_multiple_leaves": (create_multiple_leaves_bytes, "leaf"),
    "synthetic_complex_background_leaf": (create_complex_background_bytes, "leaf"),
    "synthetic_sparse_vegetation": (create_sparse_vegetation_bytes, "leaf"),
    "synthetic_scattered_vegetation": (create_scattered_vegetation_bytes, "leaf"),
    "synthetic_tiny_leaf": (create_tiny_leaf_bytes, "leaf"),
    "synthetic_dark_leaf": (create_dark_leaf_bytes, "leaf"),
    "synthetic_blurry_leaf": (create_blurry_leaf_bytes, "leaf"),
}


def bucket(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:2], "big") % 100


def load_manifest() -> list[dict]:
    with open(MANIFEST, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_categories(rows: list[dict], gate: LeafGateService) -> dict[str, dict]:
    """Assembles the evaluation sets, all of which were excluded from training."""
    categories: dict[str, dict] = {}

    def add(name: str, kind: str, paths: list[str]) -> None:
        categories.setdefault(name, {"kind": kind, "paths": []})["paths"].extend(paths)

    leaf_eval, leaf_val, backgrounds = [], [], []
    unsupported_eval, unseen = [], defaultdict(list)

    for row in rows:
        relative = row["relative_path"].replace("\\", "/")
        split, class_name = row["split"], row["class_name"]

        if split == "test":
            continue  # frozen: never opened

        if class_name in SUPPORTED_CLASSES and split == "val":
            (leaf_eval if bucket(relative) >= 50 else leaf_val).append(relative)
            continue

        if split != "excluded":
            continue

        if class_name == BACKGROUND_CLASS:
            if bucket(relative) >= 85:
                backgrounds.append(relative)
            continue

        if class_name.startswith(HELD_OUT_SPECIES_PREFIXES):
            unseen[class_name.split("___")[0]].append(relative)
            continue

        if bucket(relative) >= 85:
            unsupported_eval.append(relative)

    add("supported_leaf_val_eval_half", "leaf", leaf_eval)
    add("supported_leaf_val_tuning_half", "leaf", leaf_val)
    add("real_field_scene_without_leaves", "non_leaf", backgrounds)
    add("unsupported_species_seen", "unsupported", unsupported_eval[:1200])
    for species, paths in unseen.items():
        add(f"unseen_species_{species}", "unsupported", paths[:300])

    if OOD_MANIFEST.exists():
        corpus = json.loads(OOD_MANIFEST.read_text(encoding="utf-8"))
        per_category: dict[str, list[str]] = defaultdict(list)
        for entry in corpus["files"]:
            if bucket(entry["path"]) >= 50:
                per_category[entry["category"]].append(entry["path"])
        for name, paths in sorted(per_category.items()):
            add(f"real_non_leaf_{name}", "non_leaf", paths)
    else:
        raise SystemExit(f"OOD corpus manifest missing at {OOD_MANIFEST}")

    return categories


def assess_paths(gate: LeafGateService, paths: list[str]) -> list[dict]:
    """Runs the deployed gate + decision function over a list of images."""
    records = []
    for relative in paths:
        try:
            tensor = preprocess_image((DATA_ROOT / relative).read_bytes())
        except Exception as exc:  # physical-integrity rejection is still a safe outcome
            records.append({"decision": REJECTED, "code": "PHYSICAL_LAYER_REJECTION",
                            "error": type(exc).__name__})
            continue
        frame = tensor[0]
        probabilities = gate.predict_proba(frame)
        assessment = assess_image(frame, probabilities)
        records.append({"decision": assessment.decision, "code": assessment.code,
                        "signals": assessment.signals})
    return records


def assess_probe(gate: LeafGateService, payload: bytes) -> dict:
    try:
        tensor = preprocess_image(payload)
    except Exception as exc:
        return {"decision": REJECTED, "code": "PHYSICAL_LAYER_REJECTION", "error": type(exc).__name__}
    frame = tensor[0]
    assessment = assess_image(frame, gate.predict_proba(frame))
    return {"decision": assessment.decision, "code": assessment.code, "signals": assessment.signals}


def summarize(records: list[dict], kind: str) -> dict:
    counts = Counter(record["decision"] for record in records)
    reasons = Counter(record["code"] for record in records if record["decision"] != ACCEPTED)
    errors = Counter(record.get("error") for record in records if record.get("error"))
    total = len(records)
    accepted, rejected, abstained = counts[ACCEPTED], counts[REJECTED], counts[ABSTAINED]
    summary = {
        "n": total,
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "decode_errors": sum(errors.values()),
        "acceptance_rate": round(accepted / total, 4) if total else None,
        "no_diagnosis_rate": round((rejected + abstained) / total, 4) if total else None,
        "reasons": dict(reasons),
    }
    if kind == "non_leaf":
        summary["false_acceptance_rate"] = summary["acceptance_rate"]
    else:
        summary["valid_leaf_no_diagnosis_rate"] = summary["no_diagnosis_rate"]
        summary["valid_leaf_hard_rejection_rate"] = round(rejected / total, 4) if total else None
    return summary


def verdict_breakdown(records: list[dict]) -> dict:
    """Counts the raw gate verdicts, ignoring the confidence floor."""
    labels = Counter(record.get("signals", {}).get("gate_label")
                     for record in records if record.get("signals", {}).get("gate_label"))
    return dict(labels)


def latency_ms(gate: LeafGateService, paths: list[str], sample: int = 120) -> dict:
    step = max(1, len(paths) // sample)
    measured = []
    for relative in paths[::step][:sample]:
        frame = preprocess_image((DATA_ROOT / relative).read_bytes())[0]
        start = time.perf_counter()
        gate.predict_proba(frame)
        measured.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples": len(measured),
        "mean_ms": round(statistics.mean(measured), 2),
        "median_ms": round(statistics.median(measured), 2),
        "p95_ms": round(sorted(measured)[int(len(measured) * 0.95) - 1], 2),
    }


def load_val_scores() -> dict | None:
    """Confidence distribution on the tuning half, used to document the chosen floor."""
    path = FEATURE_DIR / "val.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {"source": "gate_features/val.npz", "n": int(len(data["labels"]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not LEAF_GATE_MODEL_PATH.is_file():
        raise SystemExit(f"Trained gate not found at {LEAF_GATE_MODEL_PATH}. Run train_input_gate.py first.")

    gate = LeafGateService()
    gate.load_model(LEAF_GATE_MODEL_PATH)
    print(f"Gate loaded: classes={gate.classes} threshold={LEAF_GATE_MIN_CONFIDENCE}")

    rows = load_manifest()
    categories = build_categories(rows, gate)

    results: dict[str, dict] = {}
    for name, spec in categories.items():
        print(f"Assessing {name} ({len(spec['paths'])} images)...")
        records = assess_paths(gate, spec["paths"])
        summary = summarize(records, spec["kind"])
        summary["gate_verdicts_ignoring_floor"] = verdict_breakdown(records)
        results[name] = summary

    print("Assessing synthetic probes...")
    for name, (factory, kind) in SYNTHETIC_PROBES.items():
        record = assess_probe(gate, factory())
        summary = summarize([record], "non_leaf" if kind == "non_leaf" else "leaf")
        summary["gate_verdicts_ignoring_floor"] = verdict_breakdown([record])
        results[name] = summary

    leaf_paths = categories["supported_leaf_val_eval_half"]["paths"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "module": "backend.app.services.input_gate + backend.app.services.leaf_gate",
            "model": LEAF_GATE_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "classes": gate.classes,
            "confidence_floor": LEAF_GATE_MIN_CONFIDENCE,
        },
        "dataset_isolation": {
            "supported_leaves": "disease val split, path-hash eval half (never trained on)",
            "real_non_leaf_scenes": "1,143 leaf-free field photographs, hash eval slice",
            "real_non_leaf_photos": "Wikimedia Commons corpus, per-category hash eval half",
            "unsupported_species_seen": "excluded species, hash eval slice",
            "unseen_species": "Strawberry and Soybean held out of training entirely",
            "synthetic_probes": "generated by backend/tests/non_leaf_fixtures.py",
            "disease_test_split": "never opened by this script or by training",
        },
        "evaluation": results,
        "gate_latency": latency_ms(gate, leaf_paths),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Held-out evaluation ===")
    print(f"{'category':38s} {'n':>5s} {'accept':>7s} {'reject':>7s} {'abstain':>8s} {'noDx':>7s}")
    for name, summary in results.items():
        rate = summary["acceptance_rate"] if summary["acceptance_rate"] is not None else 0.0
        no_dx = summary["no_diagnosis_rate"] if summary["no_diagnosis_rate"] is not None else 0.0
        print(f"{name:38s} {summary['n']:5d} {rate:7.3f} {summary['rejected']:7d} "
              f"{summary['abstained']:8d} {no_dx:7.3f}")
    print(f"\nLatency: {payload['gate_latency']}")
    print(f"Artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
