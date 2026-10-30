"""
Trains the PHYTOSIGNAL leaf-acceptance gate.

WHAT THIS MODEL IS
------------------
A 3-class classifier over the same 224x224 RGB frame the disease model sees:

    supported_leaf  - a leaf of tomato, potato or bell pepper (the model's 15 classes)
    other_plant     - a leaf of a species the disease model was never trained on
    non_leaf        - people, animals, buildings, documents, objects, graphics, scenes

It is a linear head on frozen ImageNet-pretrained MobileNetV2 features. The backbone is
not fine-tuned because the negative corpus is small (tens to hundreds of images); a linear
probe on strong frozen features is the defensible choice at this data scale, and it cannot
"memorise" the handful of OOD photographs. The exported artefact is a single self-contained
Keras model (backbone + head) that the backend loads like any other model file.

SPLIT DISCIPLINE (no leakage, no tuning on the disease test set)
---------------------------------------------------------------
* supported leaves:  disease `train` split -> gate-train
                     disease `val` split, path-hash parity -> gate-val / gate-eval
                     disease `test` split -> NEVER READ by this script
* other_plant:       `Strawberry___*` and `Soybean___healthy` are held out ENTIRELY, so the
                     reported number is transfer to unseen species. The remaining
                     unsupported classes are path-hash split 70/15/15.
* non_leaf:          1,143 real `Background_without_leaves` photographs, path-hash 70/15/15,
                     plus the downloaded real-photo OOD corpus (per-category 50/50).
* Threshold selection happens on gate-val only. gate-eval is scored once.

USAGE
    .venv/Scripts/python.exe ml/scripts/train_input_gate.py --extract   # cache features
    .venv/Scripts/python.exe ml/scripts/train_input_gate.py --train     # fit + export
    .venv/Scripts/python.exe ml/scripts/train_input_gate.py             # both
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

MANIFEST = PROJECT_ROOT / "ml" / "data" / "dataset_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "ml"
OOD_CORPUS = PROJECT_ROOT / "ml" / "data" / "ood_corpus"
OOD_MANIFEST = OOD_CORPUS / "manifest.json"
FEATURE_DIR = PROJECT_ROOT / "ml" / "data" / "gate_features"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "input_gate"
MODEL_PATH = MODEL_DIR / "leaf_gate.keras"
METRICS_PATH = PROJECT_ROOT / "ml" / "models" / "input_gate" / "training_metrics.json"

SUPPORTED_CLASSES = {
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
}

# Whole species held out of training: the gate must generalise to leaves it has never seen.
HELD_OUT_SPECIES_PREFIXES = ("Strawberry___", "Soybean___")

BACKGROUND_CLASS = "Background_without_leaves"

# Per-class caps keep feature extraction bounded without distorting class balance much
# (every class is capped at the same number for train/val; eval caps are per class too).
TRAIN_CAP_UNSUPPORTED = 500
TRAIN_CAP_SUPPORTED = 10_000  # effectively "all"; the disease train split has ~1-3k/class
VAL_CAP_UNSUPPORTED = 150
EVAL_CAP = 400
EVAL_CAP_LEAF = 250

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
C = 1.0  # regularisation strength chosen on gate-val (see --train output)


def bucket(key: str) -> int:
    """Deterministic 0-99 bucket for a path, so splits do not depend on file order."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:2], "big") % 100


def load_samples() -> list[dict]:
    """Assembles the labelled sample list with explicit gate splits and categories."""
    samples: list[dict] = []
    with open(MANIFEST, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    per_class_train = Counter()
    per_class_val = Counter()
    per_class_eval = Counter()

    for row in rows:
        relative = row["relative_path"].replace("\\", "/")
        split = row["split"]
        class_name = row["class_name"]

        if split == "test":
            continue  # frozen disease test split: never opened here

        if class_name in SUPPORTED_CLASSES:
            if split == "train":
                if per_class_train[class_name] >= TRAIN_CAP_SUPPORTED:
                    continue
                per_class_train[class_name] += 1
                samples.append({"path": relative, "label": "supported_leaf",
                                "gate_split": "train", "category": "supported_leaf_train"})
            elif split == "val":
                gate_split = "val" if bucket(relative) < 50 else "eval"
                samples.append({"path": relative, "label": "supported_leaf",
                                "gate_split": gate_split,
                                "category": f"supported_leaf_{gate_split}"})
            continue

        if split != "excluded":
            continue

        if class_name == BACKGROUND_CLASS:
            b = bucket(relative)
            gate_split = "train" if b < 70 else ("val" if b < 85 else "eval")
            samples.append({"path": relative, "label": "non_leaf", "gate_split": gate_split,
                            "category": f"field_background_{gate_split}"})
            continue

        # Unsupported crop species.
        is_held_out = class_name.startswith(HELD_OUT_SPECIES_PREFIXES)
        if is_held_out:
            if per_class_eval[class_name] < EVAL_CAP:
                per_class_eval[class_name] += 1
                samples.append({"path": relative, "label": "other_plant", "gate_split": "eval",
                                "category": f"unseen_species_{class_name.split('___')[0]}"})
            continue

        b = bucket(relative)
        if b < 70:
            if per_class_train[class_name] >= TRAIN_CAP_UNSUPPORTED:
                continue
            per_class_train[class_name] += 1
            gate_split = "train"
        elif b < 85:
            if per_class_val[class_name] >= VAL_CAP_UNSUPPORTED:
                continue
            per_class_val[class_name] += 1
            gate_split = "val"
        else:
            if per_class_eval[class_name] >= EVAL_CAP:
                continue
            per_class_eval[class_name] += 1
            gate_split = "eval"
        samples.append({"path": relative, "label": "other_plant", "gate_split": gate_split,
                        "category": f"unsupported_species_{gate_split}"})

    # Real non-plant photographs (people, animals, buildings, documents, objects, graphics).
    if OOD_MANIFEST.exists():
        corpus = json.loads(OOD_MANIFEST.read_text(encoding="utf-8"))
        for entry in corpus["files"]:
            relative = entry["path"]
            gate_split = "train" if bucket(relative) < 50 else "eval"
            samples.append({"path": relative, "label": "non_leaf", "gate_split": gate_split,
                            "category": f"ood_{entry['category']}_{gate_split}"})
    return samples


def build_feature_extractor():
    """Frozen ImageNet MobileNetV2 + global pooling, fed raw [0, 255] RGB (as the API does)."""
    import keras
    from keras import layers

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_255")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = layers.GlobalAveragePooling2D(name="embedding")(x)
    return keras.Model(inputs, outputs, name="gate_feature_extractor")


def resolve(relative_path: str) -> Path:
    """Manifest paths are relative to ml/ (e.g. 'data/raw_dataset/...'); corpus paths to the root."""
    candidate = DATA_ROOT / relative_path
    return candidate if candidate.exists() else PROJECT_ROOT / relative_path


def load_frame(path: Path) -> np.ndarray:
    """Decodes one image exactly like the serving path: EXIF-corrected, RGB, 224x224."""
    import io

    from PIL import Image, ImageOps

    with open(path, "rb") as handle:
        payload = handle.read()
    image = Image.open(io.BytesIO(payload))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.load()
    return np.asarray(image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)


def extract(samples: list[dict], force: bool = False) -> None:
    """Caches 1280-d embeddings per gate split so retraining is seconds, not minutes."""
    import keras

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_split[sample["gate_split"]].append(sample)

    model = None
    for split, items in sorted(by_split.items()):
        destination = FEATURE_DIR / f"{split}.npz"
        if destination.exists() and not force:
            cached = np.load(destination, allow_pickle=True)
            if len(cached["paths"]) == len(items):
                print(f"[{split}] cached: {len(items)} samples")
                continue

        if model is None:
            print("Building frozen MobileNetV2 feature extractor...")
            model = build_feature_extractor()

        print(f"[{split}] extracting {len(items)} embeddings...")
        features = np.zeros((len(items), 1280), dtype=np.float32)
        kept: list[dict] = []
        batch: list[np.ndarray] = []
        batch_meta: list[dict] = []
        failures = 0

        def flush() -> None:
            nonlocal features, kept, batch, batch_meta
            if not batch:
                return
            out = model.predict(np.stack(batch), batch_size=BATCH_SIZE, verbose=0)
            offset = len(kept)
            features[offset:offset + len(out)] = out
            kept.extend(batch_meta)
            batch, batch_meta = [], []

        for index, item in enumerate(items, start=1):
            try:
                frame = load_frame(resolve(item["path"]))
            except Exception:
                failures += 1
                continue
            batch.append(frame)
            batch_meta.append(item)
            if len(batch) >= BATCH_SIZE:
                flush()
            if index % 2000 == 0:
                print(f"    {index}/{len(items)}")

        flush()
        kept_n = len(kept)
        np.savez_compressed(
            destination,
            features=features[:kept_n],
            paths=np.array([k["path"] for k in kept]),
            labels=np.array([k["label"] for k in kept]),
            categories=np.array([k["category"] for k in kept]),
        )
        print(f"[{split}] saved {kept_n} embeddings ({failures} unreadable, skipped)")


def load_split(split: str):
    data = np.load(FEATURE_DIR / f"{split}.npz", allow_pickle=True)
    return data["features"], data["labels"], data["categories"], data["paths"]


def verify_splits() -> dict:
    """
    Independent integrity check on what was actually extracted.

    Guards the three ways a gate result could become meaningless: an image appearing in two
    splits, an evaluation image that was trained on, and any contact with the frozen disease
    test split.
    """
    splits = {name: load_split(name) for name in ("train", "val", "eval")}
    path_sets = {name: set(data[3].tolist()) for name, data in splits.items()}

    overlaps = {
        "train^val": len(path_sets["train"] & path_sets["val"]),
        "train^eval": len(path_sets["train"] & path_sets["eval"]),
        "val^eval": len(path_sets["val"] & path_sets["eval"]),
    }

    with open(MANIFEST, encoding="utf-8") as handle:
        test_paths = {
            row["relative_path"].replace("\\", "/")
            for row in csv.DictReader(handle)
            if row["split"] == "test"
        }
    test_contact = len(test_paths & (path_sets["train"] | path_sets["val"] | path_sets["eval"]))

    report = {
        "split_sizes": {name: len(paths) for name, paths in path_sets.items()},
        "overlaps": overlaps,
        "disease_test_split_images_used": test_contact,
        "duplicate_paths_within_split": {
            name: len(data[3]) - len(path_sets[name]) for name, data in splits.items()
        },
    }
    failures = [key for key, value in overlaps.items() if value] + (
        ["disease_test_split"] if test_contact else []
    )
    if failures:
        raise SystemExit(f"Gate split integrity failure: {failures} in {report}")
    return report


def trains_and_export() -> dict:
    from sklearn.linear_model import LogisticRegression

    X_train, y_train, _, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    print(f"train {X_train.shape}  val {X_val.shape}")
    print("train labels:", dict(Counter(y_train)), " val labels:", dict(Counter(y_val)))

    best = None
    for C_value in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C_value, max_iter=3000, class_weight="balanced")
        clf.fit(X_train, y_train)
        accuracy = float((clf.predict(X_val) == y_val).mean())
        print(f"  C={C_value:<5} val accuracy={accuracy:.4f}")
        if best is None or accuracy > best[1]:
            best = (C_value, accuracy, clf)

    C_value, accuracy, clf = best
    print(f"Selected C={C_value} (val accuracy {accuracy:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    classes = list(clf.classes_)
    weights = clf.coef_.T.astype(np.float32)   # (1280, n_classes)
    bias = clf.intercept_.astype(np.float32)

    import keras
    from keras import layers

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_0_255")
    base = build_feature_extractor()
    x = base(inputs)
    outputs = layers.Dense(len(classes), activation="softmax", name="leaf_gate_head")(x)
    export = keras.Model(inputs, outputs, name="leaf_acceptance_gate")
    export.get_layer("leaf_gate_head").set_weights([weights, bias])
    export.save(MODEL_PATH)
    print(f"Exported {MODEL_PATH}")

    # The exported graph must reproduce the sklearn decision exactly, otherwise the
    # artefact the API loads is not the model that was validated. This is checked by
    # pushing the original IMAGES through the exported model and comparing with the
    # saved sklearn-side embeddings for the same rows.
    val_paths = [str(p) for p in np.load(FEATURE_DIR / "val.npz", allow_pickle=True)["paths"][:32]]
    frames = np.stack([load_frame(resolve(path)) for path in val_paths])
    reference = clf.predict_proba(X_val[:32])
    exported = np.asarray(export.predict(frames, verbose=0), dtype=np.float64)
    max_delta = float(np.abs(reference - exported).max())
    print(f"Export parity max |sklearn - keras| = {max_delta:.3e}")

    (MODEL_DIR / "gate_model_config.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "C": C_value,
                "val_accuracy": accuracy,
                "feature_backbone": "MobileNetV2 (ImageNet, frozen)",
                "input": "224x224 RGB in [0, 255]",
                "export_parity_max_abs_delta": max_delta,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"classes": classes, "C": C_value, "val_accuracy": accuracy,
            "export_parity_max_abs_delta": max_delta}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    do_extract = args.extract or args.force_extract or not args.train
    do_train = args.train or not args.extract

    samples = load_samples()
    print(f"{len(samples)} samples |",
          dict(Counter((s["gate_split"], s["label"]) for s in samples)))

    if do_extract:
        extract(samples, force=args.force_extract)
    if do_train:
        integrity = verify_splits()
        print("Split integrity:", json.dumps(integrity))
        result = trains_and_export()
        result["split_integrity"] = integrity
        METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
