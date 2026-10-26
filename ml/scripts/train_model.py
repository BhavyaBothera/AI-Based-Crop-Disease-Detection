#!/usr/bin/env python3
"""
MobileNetV2 Crop Disease Classification Training Pipeline
=========================================================
Part of the "AI-Based Crop Disease Detection and Treatment Recommendation System"

This script implements Phase 3B training:
  1. Loads and validates the fixed train and validation splits from ml/data/dataset_manifest.csv.
  2. Guarantees strict test set isolation (3,413 test images are never read or decoded).
  3. Prepares streaming tf.data input pipelines with conservative, label-preserving training augmentations.
  4. Builds MobileNetV2 transfer learning architecture:
     - Input (224, 224, 3) raw RGB pixels [0, 255]
     - Embedded Rescaling to [-1, 1] exactly once inside the model
     - Pretrained MobileNetV2 backbone called with training=False to freeze BatchNormalization
     - Classification head: GlobalAveragePooling2D -> BatchNormalization -> Dense(256, ReLU, L2=1e-4) -> Dropout(0.35) -> Dense(15, Softmax)
  5. Executes two-stage training:
     - Stage 1: Frozen backbone, train classification head (Adam, LR=1e-3).
     - Stage 2: Fine-tuning upper backbone layers (100..153) with all BatchNormalization layers strictly frozen (Adam, LR=1e-5).
  6. Implements validation Macro-F1 model selection with deterministic tie-breaking (tolerance 1e-4, lower val_loss tie-breaker).
  7. Supports unweighted and class-weighted ('balanced' computed strictly on training set) experiments with distinct output directories.
  8. Provides experiment comparison utility to select the superior validation model without touching the test set.

Usage:
  py -3.11 ml/scripts/train_model.py --help
  py -3.11 ml/scripts/train_model.py --dry-run
  py -3.11 ml/scripts/train_model.py --class-weighting none
  py -3.11 ml/scripts/train_model.py --class-weighting balanced
  py -3.11 ml/scripts/train_model.py --compare ml/models/unweighted ml/models/balanced
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import PIL
import sklearn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, regularizers

# ---------------------------------------------------------------------------
# Project Paths & Authoritative Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Stable 15 Solanaceae Class Mapping (Class IDs 0 to 14)
# Must strictly match ml/scripts/prepare_dataset.py and ml/data/dataset_manifest.csv
TARGET_CLASS_MAPPING: Dict[str, Dict[str, Any]] = {
    "Pepper,_bell___Bacterial_spot": {
        "id": 0,
        "crop": "Bell Pepper",
        "health": "Diseased",
        "condition": "Bacterial Spot",
    },
    "Pepper,_bell___healthy": {
        "id": 1,
        "crop": "Bell Pepper",
        "health": "Healthy",
        "condition": "Healthy",
    },
    "Potato___Early_blight": {
        "id": 2,
        "crop": "Potato",
        "health": "Diseased",
        "condition": "Early Blight",
    },
    "Potato___Late_blight": {
        "id": 3,
        "crop": "Potato",
        "health": "Diseased",
        "condition": "Late Blight",
    },
    "Potato___healthy": {
        "id": 4,
        "crop": "Potato",
        "health": "Healthy",
        "condition": "Healthy",
    },
    "Tomato___Bacterial_spot": {
        "id": 5,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Bacterial Spot",
    },
    "Tomato___Early_blight": {
        "id": 6,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Early Blight",
    },
    "Tomato___Late_blight": {
        "id": 7,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Late Blight",
    },
    "Tomato___Leaf_Mold": {
        "id": 8,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Leaf Mold",
    },
    "Tomato___Septoria_leaf_spot": {
        "id": 9,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Septoria Leaf Spot",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "id": 10,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Two-spotted Spider Mite",
    },
    "Tomato___Target_Spot": {
        "id": 11,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Target Spot",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "id": 12,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Yellow Leaf Curl Virus",
    },
    "Tomato___Tomato_mosaic_virus": {
        "id": 13,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Mosaic Virus",
    },
    "Tomato___healthy": {
        "id": 14,
        "crop": "Tomato",
        "health": "Healthy",
        "condition": "Healthy",
    },
}

CLASS_ID_TO_NAME: Dict[int, str] = {
    v["id"]: k for k, v in TARGET_CLASS_MAPPING.items()
}

NUM_CLASSES = 15
IMAGE_SHAPE = (224, 224, 3)

EXPECTED_COUNTS = {
    "train": 15942,
    "val": 3418,
    "test": 3413,
    "total_included": 22773,
}


def resolve_path(path_str: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolves relative paths reliably against project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


# ---------------------------------------------------------------------------
# Reproducibility Setup
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """
    Sets deterministic seeds for random, NumPy, and TensorFlow.
    
    Note on determinism limitations:
    On modern multicore CPUs and Intel oneDNN runtime libraries, small non-deterministic
    floating-point variations may occur in parallel reductions, but dataset shuffling,
    augmentation sampling, and initial weight draws will remain reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ---------------------------------------------------------------------------
# Manifest Loading & Validation (Strict Test Set Isolation)
# ---------------------------------------------------------------------------

def load_and_validate_manifest(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
    enforce_exact_counts: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads and validates the dataset manifest.
    
    Enforces:
      - Only 'status == included' rows are processed.
      - Required schema columns exist.
      - class_id is in 0..14 and matches authoritative TARGET_CLASS_MAPPING.
      - Train and validation image paths exist on disk.
      - Fixed split counts (train 15,942; val 3,418; test 3,413).
      - Test split isolation: Test records are verified to exist in the manifest but
        are strictly NEVER loaded into memory for training or validation.
    
    Returns:
      (train_records, val_records)
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    required_columns = {
        "image_id",
        "relative_path",
        "crop",
        "class_name",
        "class_id",
        "split",
        "status",
    }

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_count = 0
    excluded_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing_cols = required_columns - header
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

        for row_idx, row in enumerate(reader, start=2):
            status = row.get("status", "").strip().lower()
            if status != "included":
                excluded_count += 1
                continue

            split = row.get("split", "").strip().lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split '{split}' at row {row_idx}")

            try:
                class_id = int(row["class_id"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid non-integer class_id '{row.get('class_id')}' at row {row_idx}")

            if class_id not in CLASS_ID_TO_NAME:
                raise ValueError(f"class_id {class_id} out of bounds (0..14) at row {row_idx}")

            class_name = row.get("class_name", "").strip()
            if class_name != CLASS_ID_TO_NAME[class_id]:
                raise ValueError(
                    f"Class mapping mismatch at row {row_idx}: "
                    f"class_id {class_id} mapped to '{class_name}', expected '{CLASS_ID_TO_NAME[class_id]}'"
                )

            # Test set isolation: Verify count only, do not load or decode test images
            if split == "test":
                test_count += 1
                continue

            # Resolve image path
            rel_path = row["relative_path"].strip()
            full_path = project_root / "ml" / rel_path
            if not full_path.is_file():
                full_path = project_root / rel_path
            if not full_path.is_file():
                raise FileNotFoundError(
                    f"Image file for row {row_idx} ({row.get('image_id')}) not found on disk: {full_path}"
                )

            record = {
                "image_id": row["image_id"],
                "file_path": str(full_path),
                "class_id": class_id,
                "class_name": class_name,
                "split": split,
            }

            if split == "train":
                train_records.append(record)
            elif split == "val":
                val_records.append(record)

    total_included = len(train_records) + len(val_records) + test_count
    print(f"Manifest Validation Summary:")
    print(f"  * Included Train Images : {len(train_records):,}")
    print(f"  * Included Val Images   : {len(val_records):,}")
    print(f"  * Included Test Images  : {test_count:,} (Isolated, not loaded)")
    print(f"  * Total Included        : {total_included:,}")
    print(f"  * Excluded Rows (other) : {excluded_count:,}")

    if enforce_exact_counts:
        if len(train_records) != EXPECTED_COUNTS["train"]:
            raise ValueError(
                f"Train count mismatch: expected {EXPECTED_COUNTS['train']}, got {len(train_records)}"
            )
        if len(val_records) != EXPECTED_COUNTS["val"]:
            raise ValueError(
                f"Validation count mismatch: expected {EXPECTED_COUNTS['val']}, got {len(val_records)}"
            )
        if test_count != EXPECTED_COUNTS["test"]:
            raise ValueError(
                f"Test count mismatch: expected {EXPECTED_COUNTS['test']}, got {test_count}"
            )

    return train_records, val_records


# ---------------------------------------------------------------------------
# Class Weight Computation (Training Split Only)
# ---------------------------------------------------------------------------

def compute_balanced_class_weights(train_labels: List[int], num_classes: int = NUM_CLASSES) -> Dict[int, float]:
    """
    Computes balanced class weights strictly from the training set labels:
      w_j = N / (K * N_j)
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    return class_weights


# ---------------------------------------------------------------------------
# tf.data Streaming Pipeline
# ---------------------------------------------------------------------------

def _load_and_resize_image(file_path: tf.Tensor) -> tf.Tensor:
    """Reads image file from disk and resizes to 224x224 RGB [0, 255]."""
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3])
    return img


def _augment_training_image(image: tf.Tensor) -> tf.Tensor:
    """
    Conservative, label-preserving augmentations for training images:
      - Random horizontal flip (leaf horizontal orientation invariance)
      - Random vertical flip (leaf vertical orientation invariance)
      - Slight brightness perturbation (+/- 10%)
      - Slight contrast perturbation (0.9 to 1.1)
    Does not apply heavy hue shifts or distortions that alter disease symptoms.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=25.5)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def create_dataset(
    records: List[Dict[str, Any]],
    batch_size: int = 32,
    is_training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Constructs a tf.data.Dataset pipeline.
    
    Training:
      - Deterministic shuffle buffer
      - Conservative label-preserving augmentations
      - Batched and prefetched with AUTOTUNE
    Validation:
      - Deterministic un-shuffled order
      - No augmentation
      - Batched and prefetched with AUTOTUNE
    """
    file_paths = [r["file_path"] for r in records]
    labels = [r["class_id"] for r in records]

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(
            buffer_size=min(len(records), 8192),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(
            lambda p, y: (_augment_training_image(_load_and_resize_image(p)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda p, y: (_load_and_resize_image(p), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# MobileNetV2 Architecture & Two-Stage Freezing Configuration
# ---------------------------------------------------------------------------

def build_mobilenetv2_model(
    num_classes: int = NUM_CLASSES,
    weights: Optional[str] = "imagenet",
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.35,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Builds the MobileNetV2 transfer learning model:
      - Input: (224, 224, 3) raw RGB in [0, 255]
      - Rescaling: Exactly once embedded in model [-1, 1]
      - Backbone: MobileNetV2 without top, called with training=False
      - Head: GlobalAveragePooling2D -> BatchNormalization -> Dense(256, ReLU, L2) -> Dropout -> Dense(15, Softmax)
    
    Returns:
      (model, backbone)
    """
    inputs = layers.Input(shape=IMAGE_SHAPE, name="input_image")
    
    # Embedded normalization to [-1, 1] exactly once inside the model
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)

    backbone_weights = None if (weights is None or str(weights).lower() == "none") else weights
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights=backbone_weights,
    )
    # Stage 1 default: Freeze entire backbone
    backbone.trainable = False

    # Calling backbone with training=False guarantees BatchNormalization layers
    # remain in inference mode using their moving statistics
    x = backbone(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="head_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_crop_disease")
    return model, backbone


def configure_stage2_finetuning(
    backbone: tf.keras.Model,
    unfreeze_from_layer: int = 100,
) -> Dict[str, int]:
    """
    Configures Stage 2 fine-tuning:
      - Unfreezes upper backbone layers (>= unfreeze_from_layer).
      - Strictly freezes ALL BatchNormalization layers inside the backbone.
      - Keeps layers < unfreeze_from_layer frozen.
    
    Returns diagnostic layer counts.
    """
    backbone.trainable = True
    total_layers = len(backbone.layers)
    unfrozen_count = 0
    frozen_bn_count = 0
    frozen_lower_count = 0

    for i, layer in enumerate(backbone.layers):
        if i < unfreeze_from_layer:
            layer.trainable = False
            frozen_lower_count += 1
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn_count += 1
        else:
            layer.trainable = True
            unfrozen_count += 1

    diagnostics = {
        "total_backbone_layers": total_layers,
        "unfrozen_upper_layers": unfrozen_count,
        "frozen_bn_layers": frozen_bn_count,
        "frozen_lower_layers": frozen_lower_count,
    }
    return diagnostics


# ---------------------------------------------------------------------------
# Validation Macro-F1 Metric & Model Selection Callbacks
# ---------------------------------------------------------------------------

class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    """
    Computes full-validation-set Macro-F1 score after every epoch.
    Never averages mini-batch F1 scores.
    Injects 'val_macro_f1' into logs so downstream callbacks can monitor it.
    Must be registered FIRST in the callbacks list.
    """
    def __init__(self, val_dataset: tf.data.Dataset, y_true: np.ndarray):
        super().__init__()
        self.val_dataset = val_dataset
        self.y_true = np.asarray(y_true, dtype=np.int32)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        # Predict on entire validation set
        val_probs = self.model.predict(self.val_dataset, verbose=0)
        val_preds = np.argmax(val_probs, axis=-1)
        val_macro_f1 = f1_score(self.y_true, val_preds, average="macro", zero_division=0)
        logs["val_macro_f1"] = float(val_macro_f1)
        print(f"\n -- val_macro_f1: {val_macro_f1:.4f} (full-validation Macro-F1)")


class BestModelCheckpointTracker(tf.keras.callbacks.Callback):
    """
    Preserves the global best checkpoint across both training stages.
    
    Selection Rule:
      - Selected by highest validation Macro-F1.
      - Differences within tolerance (default 1e-4) are treated as ties.
      - Ties are broken by lowest validation loss (val_loss).
      - Saves checkpoint (.keras) and selection metadata (.json).
    """
    def __init__(
        self,
        checkpoint_path: Path,
        metadata_path: Path,
        f1_tolerance: float = 1e-4,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.metadata_path = metadata_path
        self.f1_tolerance = f1_tolerance

        # Global best tracking across stages
        self.best_f1: Optional[float] = None
        self.best_loss: Optional[float] = None
        self.best_stage: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_reason: Optional[str] = None
        self.current_stage: int = 1

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        val_f1 = logs.get("val_macro_f1")
        val_loss = logs.get("val_loss")

        if val_f1 is None or val_loss is None:
            return

        val_f1 = float(val_f1)
        val_loss = float(val_loss)

        is_better = False
        reason = ""

        if self.best_f1 is None:
            is_better = True
            reason = f"Initial baseline checkpoint (val_macro_f1={val_f1:.6f}, val_loss={val_loss:.6f})"
        elif val_f1 > (self.best_f1 + self.f1_tolerance):
            is_better = True
            reason = f"Higher val_macro_f1 ({val_f1:.6f} > {self.best_f1:.6f}, diff: +{val_f1 - self.best_f1:.6f})"
        elif abs(val_f1 - self.best_f1) <= self.f1_tolerance:
            # Macro-F1 tie within tolerance: use lower val_loss as tie-breaker
            if val_loss < self.best_loss:
                is_better = True
                reason = (
                    f"Macro-F1 tie within tolerance ({val_f1:.6f} vs {self.best_f1:.6f}), "
                    f"broken by lower val_loss ({val_loss:.6f} < {self.best_loss:.6f})"
                )

        if is_better:
            self.best_f1 = val_f1
            self.best_loss = val_loss
            self.best_stage = self.current_stage
            self.best_epoch = epoch + 1
            self.best_reason = reason

            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_path)

            metadata = {
                "checkpoint_file": self.checkpoint_path.name,
                "stage": self.best_stage,
                "epoch": self.best_epoch,
                "val_macro_f1": float(self.best_f1),
                "val_loss": float(self.best_loss),
                "selection_reason": self.best_reason,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f" >>> [NEW GLOBAL BEST] Stage {self.best_stage} Epoch {self.best_epoch}: "
                  f"val_macro_f1={self.best_f1:.4f}, val_loss={self.best_loss:.4f} ({reason})")

    def restore_best_weights(self, target_model: Optional[tf.keras.Model] = None) -> bool:
        """
        Restores the weights from the saved best checkpoint into the model.
        Returns True if weights were loaded, False otherwise.
        """
        m = target_model or self.model
        if self.checkpoint_path.is_file():
            m.load_weights(self.checkpoint_path)
            return True
        return False



# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def run_training(
    manifest_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    epochs_stage1: int = 15,
    epochs_stage2: int = 15,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-5,
    seed: int = 42,
    class_weighting: str = "none",
    weights: Optional[str] = "imagenet",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete training workflow.
    """
    start_time = time.time()
    set_seeds(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AI-BASED CROP DISEASE DETECTION -- MOBILENETV2 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Manifest Path    : {manifest_path}")
    print(f"Output Directory : {output_dir}")
    print(f"Class Weighting  : {class_weighting}")
    print(f"Batch Size       : {batch_size}")
    print(f"Epochs           : Stage 1={epochs_stage1}, Stage 2={epochs_stage2}")
    print(f"Learning Rates   : Stage 1={lr_stage1}, Stage 2={lr_stage2}")
    print(f"Random Seed      : {seed}")
    print(f"Backbone Weights : {weights}")
    print(f"Dry Run Mode     : {dry_run}")
    print("=" * 70)

    # 1. Load & Validate Manifest (Strict Isolation)
    train_records, val_records = load_and_validate_manifest(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        enforce_exact_counts=not dry_run,
    )

    train_labels = [r["class_id"] for r in train_records]
    val_labels = [r["class_id"] for r in val_records]
    val_y_true = np.array(val_labels, dtype=np.int32)

    # 2. Compute Class Weights (Train Only)
    class_weights_dict: Optional[Dict[int, float]] = None
    if class_weighting == "balanced":
        class_weights_dict = compute_balanced_class_weights(train_labels, num_classes=NUM_CLASSES)
        print("\nBalanced Class Weights (computed from training set only):")
        for cid, w in sorted(class_weights_dict.items()):
            print(f"  Class {cid:2d} ({CLASS_ID_TO_NAME[cid]}): {w:.4f}")
    elif class_weighting == "none":
        print("\nClass Weighting: None (unweighted cross-entropy)")
    else:
        raise ValueError(f"Unknown class_weighting option: {class_weighting}")

    # 3. Create tf.data Streaming Pipelines
    train_ds = create_dataset(train_records, batch_size=batch_size, is_training=True, seed=seed)
    val_ds = create_dataset(val_records, batch_size=batch_size, is_training=False, seed=seed)

    # 4. Build Model
    print("\nBuilding MobileNetV2 Architecture...")
    model, backbone = build_mobilenetv2_model(
        num_classes=NUM_CLASSES,
        weights=weights,
        l2_reg=1e-4,
        dropout_rate=0.35,
    )

    # Compile Stage 1
    optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=lr_stage1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    checkpoint_file = output_dir / "best_model.keras"
    metadata_file = output_dir / "best_model_metadata.json"
    history_file = output_dir / "training_history.json"
    config_file = output_dir / "config.json"

    # Save initial run configuration
    run_config = {
        "pipeline": "MobileNetV2 Transfer Learning",
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "class_weighting": class_weighting,
        "class_weights": class_weights_dict,
        "batch_size": batch_size,
        "epochs_stage1": epochs_stage1,
        "epochs_stage2": epochs_stage2,
        "lr_stage1": lr_stage1,
        "lr_stage2": lr_stage2,
        "weights": weights,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test_isolated": EXPECTED_COUNTS["test"],
        },
        "target_class_mapping": TARGET_CLASS_MAPPING,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": getattr(tf.keras, "__version__", "unknown"),
            "numpy": np.__version__,
            "scikit_learn": getattr(sklearn, "__version__", "unknown"),
            "pillow": getattr(PIL, "__version__", "unknown"),
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if dry_run:
        print("\n[DRY RUN] Verification completed successfully without running training epochs:")
        print("  [OK] Manifest schema and exact counts validated.")
        print("  [OK] Train/val tf.data pipelines instantiated with prefetching.")
        print("  [OK] MobileNetV2 architecture constructed and compiled for Stage 1.")
        
        # Test Stage 2 configuration diagnostics
        stage2_diag = configure_stage2_finetuning(backbone, unfreeze_from_layer=100)
        print("  [OK] Stage 2 fine-tuning configuration verified:")
        for k, v in stage2_diag.items():
            print(f"     - {k}: {v}")
        print(f"  [OK] Configuration saved to: {config_file}")
        return {"status": "dry_run_success", "config": run_config}

    # 5. Shared Callbacks Setup
    macro_f1_callback = ValidationMacroF1Callback(val_dataset=val_ds, y_true=val_y_true)
    checkpoint_tracker = BestModelCheckpointTracker(
        checkpoint_path=checkpoint_file,
        metadata_path=metadata_file,
        f1_tolerance=1e-4,
    )

    combined_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Stage 1: Frozen Backbone Training
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD (BACKBONE FROZEN)")
    print("=" * 70)
    checkpoint_tracker.current_stage = 1

    stage1_callbacks = [
        macro_f1_callback,
        checkpoint_tracker,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            min_delta=1e-4,
            patience=5,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        shuffle=False,  # Explicitly False: train_ds handles its own deterministic shuffling; prevents Keras default shuffle=True warning
        class_weight=class_weights_dict,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    for epoch_idx in range(len(h1.history["loss"])):
        entry = {
            "stage": 1,
            "epoch": epoch_idx + 1,
            "loss": float(h1.history["loss"][epoch_idx]),
            "accuracy": float(h1.history["accuracy"][epoch_idx]),
            "val_loss": float(h1.history["val_loss"][epoch_idx]),
            "val_accuracy": float(h1.history["val_accuracy"][epoch_idx]),
            "val_macro_f1": float(h1.history["val_macro_f1"][epoch_idx]),
        }
        combined_history.append(entry)

    # -----------------------------------------------------------------------
    # Stage 2: Fine-Tuning Upper Backbone Layers
    # -----------------------------------------------------------------------
    if epochs_stage2 > 0:
        print("\n" + "=" * 70)
        print("STAGE 2: CONSERVATIVE FINE-TUNING (UPPER BACKBONE, BN FROZEN)")
        print("=" * 70)

        # Ensure Stage 2 starts from the true best Stage 1 weights
        if checkpoint_tracker.best_epoch is not None and checkpoint_file.is_file():
            print(f"\n[STAGE TRANSITION] Restoring Stage 1 best weights (Stage {checkpoint_tracker.best_stage}, "
                  f"Epoch {checkpoint_tracker.best_epoch}, val_macro_f1={checkpoint_tracker.best_f1:.6f}, "
                  f"val_loss={checkpoint_tracker.best_loss:.6f}) before Stage 2 fine-tuning...")
            checkpoint_tracker.restore_best_weights(model)

        checkpoint_tracker.current_stage = 2
        stage2_diag = configure_stage2_finetuning(backbone, unfreeze_from_layer=100)
        print("Stage 2 Backbone Configuration:")
        for k, v in stage2_diag.items():
            print(f"  * {k}: {v}")

        # Recompile with fine-tuning learning rate
        optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=lr_stage2)
        model.compile(
            optimizer=optimizer_stage2,
            loss=loss_fn,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        stage2_callbacks = [
            macro_f1_callback,
            checkpoint_tracker,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                min_delta=1e-4,
                patience=5,
                restore_best_weights=False,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_macro_f1",
                mode="max",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage2,
            shuffle=False,  # Explicitly False: prevents Keras default shuffle=True warning on tf.data.Dataset
            class_weight=class_weights_dict,
            callbacks=stage2_callbacks,
            verbose=1,
        )

        for epoch_idx in range(len(h2.history["loss"])):
            entry = {
                "stage": 2,
                "epoch": epochs_stage1 + epoch_idx + 1,
                "loss": float(h2.history["loss"][epoch_idx]),
                "accuracy": float(h2.history["accuracy"][epoch_idx]),
                "val_loss": float(h2.history["val_loss"][epoch_idx]),
                "val_accuracy": float(h2.history["val_accuracy"][epoch_idx]),
                "val_macro_f1": float(h2.history["val_macro_f1"][epoch_idx]),
            }
            combined_history.append(entry)
    else:
        print("\n" + "=" * 70)
        print("STAGE 2: SKIPPED (epochs_stage2 = 0 requested -- Stage 1 only benchmark)")
        print("=" * 70)

    # Save full combined history
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    # Final model synchronization: Ensure the in-memory model exactly matches the saved global best checkpoint
    if checkpoint_tracker.best_epoch is not None and checkpoint_file.is_file():
        print(f"\n[FINAL SYNCHRONIZATION] Restoring global best weights (Stage {checkpoint_tracker.best_stage}, "
              f"Epoch {checkpoint_tracker.best_epoch}, val_macro_f1={checkpoint_tracker.best_f1:.6f}, "
              f"val_loss={checkpoint_tracker.best_loss:.6f}) into in-memory model...")
        checkpoint_tracker.restore_best_weights(model)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING RUN COMPLETE")
    print("=" * 70)
    print(f"Total Elapsed Time : {total_time / 60:.2f} minutes")
    print(f"Best Stage         : {checkpoint_tracker.best_stage}")
    print(f"Best Epoch         : {checkpoint_tracker.best_epoch}")
    print(f"Best Val Macro-F1  : {checkpoint_tracker.best_f1:.6f}")
    print(f"Best Val Loss      : {checkpoint_tracker.best_loss:.6f}")
    print(f"Selection Reason   : {checkpoint_tracker.best_reason}")
    print(f"Checkpoint File    : {checkpoint_file}")
    print(f"Metadata File      : {metadata_file}")
    print(f"History File       : {history_file}")
    print("=" * 70)

    return {
        "status": "success",
        "best_stage": checkpoint_tracker.best_stage,
        "best_epoch": checkpoint_tracker.best_epoch,
        "best_val_macro_f1": checkpoint_tracker.best_f1,
        "best_val_loss": checkpoint_tracker.best_loss,
        "selection_reason": checkpoint_tracker.best_reason,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Experiment Comparison Utility
# ---------------------------------------------------------------------------

def compare_experiments(
    dir_unweighted: Path,
    dir_weighted: Path,
    tolerance: float = 1e-4,
) -> None:
    """
    Compares the validation results of unweighted vs class-weighted runs.
    Reads best_model_metadata.json from both directories and determines the superior model
    using the documented Macro-F1 priority and validation loss tie-breaker rule.
    Never evaluates on the test set.
    """
    meta_u_path = dir_unweighted / "best_model_metadata.json"
    meta_w_path = dir_weighted / "best_model_metadata.json"

    if not meta_u_path.is_file():
        raise FileNotFoundError(f"Unweighted metadata not found: {meta_u_path}")
    if not meta_w_path.is_file():
        raise FileNotFoundError(f"Weighted metadata not found: {meta_w_path}")

    with open(meta_u_path, "r", encoding="utf-8") as f:
        meta_u = json.load(f)
    with open(meta_w_path, "r", encoding="utf-8") as f:
        meta_w = json.load(f)

    f1_u = float(meta_u["val_macro_f1"])
    loss_u = float(meta_u["val_loss"])
    f1_w = float(meta_w["val_macro_f1"])
    loss_w = float(meta_w["val_loss"])

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPARISON: UNWEIGHTED VS. CLASS-WEIGHTED (VALIDATION SET ONLY)")
    print("=" * 75)
    print(f"{'Experiment':<22} | {'Stage':<6} | {'Epoch':<6} | {'Val Macro-F1':<14} | {'Val Loss':<10}")
    print("-" * 75)
    print(f"{'Unweighted':<22} | {meta_u['stage']:<6} | {meta_u['epoch']:<6} | {f1_u:<14.6f} | {loss_u:<10.6f}")
    print(f"{'Class-Weighted':<22} | {meta_w['stage']:<6} | {meta_w['epoch']:<6} | {f1_w:<14.6f} | {loss_w:<10.6f}")
    print("-" * 75)

    f1_diff = f1_w - f1_u
    if abs(f1_diff) <= tolerance:
        # Tie in Macro-F1
        if loss_w < loss_u:
            winner = "Class-Weighted"
            winner_dir = dir_weighted
            rationale = (
                f"Macro-F1 difference ({abs(f1_diff):.6f}) is within tolerance ({tolerance}). "
                f"Class-weighted won via lower validation loss ({loss_w:.6f} < {loss_u:.6f})."
            )
        else:
            winner = "Unweighted"
            winner_dir = dir_unweighted
            rationale = (
                f"Macro-F1 difference ({abs(f1_diff):.6f}) is within tolerance ({tolerance}). "
                f"Unweighted won via lower validation loss ({loss_u:.6f} <= {loss_w:.6f})."
            )
    elif f1_w > f1_u:
        winner = "Class-Weighted"
        winner_dir = dir_weighted
        rationale = f"Class-weighted achieved higher validation Macro-F1 ({f1_w:.6f} > {f1_u:.6f}, diff: +{f1_diff:.6f})."
    else:
        winner = "Unweighted"
        winner_dir = dir_unweighted
        rationale = f"Unweighted achieved higher validation Macro-F1 ({f1_u:.6f} > {f1_w:.6f}, diff: +{-f1_diff:.6f})."

    model_path = winner_dir / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Recommended candidate model checkpoint not found: {model_path}")

    print(f"\nRecommended Candidate Model: {winner}")
    print(f"Candidate Location         : {model_path}")
    print(f"Selection Rationale        : {rationale}")
    print("=" * 75 + "\n")

    return {
        "winner": winner,
        "winner_dir": str(winner_dir),
        "model_path": str(model_path),
        "metadata_path": str(winner_dir / "best_model_metadata.json"),
        "val_macro_f1": f1_w if winner == "Class-Weighted" else f1_u,
        "val_loss": loss_w if winner == "Class-Weighted" else loss_u,
        "selection_rationale": rationale,
    }


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for Solanaceae Crop Disease Classification (Phase 3B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="ml/data/dataset_manifest.csv",
        help="Path to dataset manifest CSV (relative to project root or absolute)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation streaming pipelines",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=15,
        help="Number of epochs for Stage 1 (classification head with frozen backbone)",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=15,
        help="Number of epochs for Stage 2 (fine-tuning upper backbone layers)",
    )
    parser.add_argument(
        "--learning-rate-stage1",
        type=float,
        default=1e-3,
        help="Initial learning rate for Stage 1 Adam optimizer",
    )
    parser.add_argument(
        "--learning-rate-stage2",
        type=float,
        default=1e-5,
        help="Initial learning rate for Stage 2 Adam optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across random, numpy, and tensorflow",
    )
    parser.add_argument(
        "--class-weighting",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="Class weighting strategy: 'none' (unweighted) or 'balanced' (inverse frequency on train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints and metadata. Defaults to ml/models/<class_weighting>",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="Initial backbone weights. Use 'imagenet' for training or 'none' for testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest, pipelines, and model compilation without running training epochs",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_UNWEIGHTED", "DIR_WEIGHTED"),
        help="Compare two experiment output directories on validation Macro-F1 and select candidate",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    if args.compare:
        dir_u = resolve_path(args.compare[0])
        dir_w = resolve_path(args.compare[1])
        compare_experiments(dir_u, dir_w)
        return

    manifest_path = resolve_path(args.manifest)

    # Determine default output directory based on class weighting if not specified
    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = resolve_path(f"ml/models/{args.class_weighting}")

    weights_param = None if args.weights == "none" else args.weights

    run_training(
        manifest_path=manifest_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_stage1=args.learning_rate_stage1,
        lr_stage2=args.learning_rate_stage2,
        seed=args.seed,
        class_weighting=args.class_weighting,
        weights=weights_param,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
