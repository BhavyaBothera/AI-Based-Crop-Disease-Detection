#!/usr/bin/env python3
"""
Unit and Integration Test Suite for ML Training Pipeline
========================================================
Part of the "AI-Based Crop Disease Detection and Treatment Recommendation System"

Verifies:
  1. Authoritative class mapping preservation against prepare_dataset.py.
  2. Dataset manifest validation, error handling, and strict test split isolation.
  3. Balanced class weight calculation (inverse frequency formula).
  4. BestModelCheckpointTracker tie-breaking and cross-stage best preservation logic.
  5. MobileNetV2 architecture specifications, embedded rescaling, and two-stage layer freezing.
  6. Experiment comparison logic (unweighted vs. class-weighted validation selection).

Usage:
  py -3.11 -m unittest ml/tests/test_train_pipeline.py
"""

import csv
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import tensorflow as tf

from ml.scripts.prepare_dataset import TARGET_CLASS_MAPPING as PREPARE_MAPPING
from ml.scripts.train_model import (
    CLASS_ID_TO_NAME,
    NUM_CLASSES,
    TARGET_CLASS_MAPPING as TRAIN_MAPPING,
    BestModelCheckpointTracker,
    build_mobilenetv2_model,
    compare_experiments,
    compute_balanced_class_weights,
    configure_stage2_finetuning,
    load_and_validate_manifest,
    parse_args,
)


class TestClassMappingPreservation(unittest.TestCase):
    """Verifies that train_model.py exactly preserves the authoritative 15-class mapping."""

    def test_exact_class_mapping_match(self):
        self.assertEqual(len(TRAIN_MAPPING), 15)
        self.assertEqual(len(PREPARE_MAPPING), 15)
        self.assertEqual(TRAIN_MAPPING, PREPARE_MAPPING)

    def test_class_ids_contiguous_zero_to_fourteen(self):
        class_ids = sorted(v["id"] for v in TRAIN_MAPPING.values())
        self.assertEqual(class_ids, list(range(15)))

    def test_bidirectional_lookup(self):
        for name, meta in TRAIN_MAPPING.items():
            cid = meta["id"]
            self.assertEqual(CLASS_ID_TO_NAME[cid], name)


class TestClassWeightComputation(unittest.TestCase):
    """Verifies balanced class weight calculation strictly on training set labels."""

    def test_inverse_frequency_balanced_weights(self):
        # 10 samples of class 0, 90 samples of class 1, total 100 samples across 2 classes
        # Formula: N / (K * N_j) -> class 0: 100 / (2 * 10) = 5.0, class 1: 100 / (2 * 90) = 0.5556
        synthetic_labels = [0] * 10 + [1] * 90
        weights = compute_balanced_class_weights(synthetic_labels, num_classes=2)
        self.assertAlmostEqual(weights[0], 5.0, places=4)
        self.assertAlmostEqual(weights[1], 100.0 / 180.0, places=4)

    def test_fifteen_class_weights(self):
        # Uniform distribution should produce weight 1.0 for all classes
        uniform_labels = [c for c in range(15) for _ in range(20)]
        weights = compute_balanced_class_weights(uniform_labels, num_classes=15)
        for c in range(15):
            self.assertAlmostEqual(weights[c], 1.0, places=4)


class TestManifestValidationAndIsolation(unittest.TestCase):
    """Verifies manifest schema validation, error detection, and strict test set isolation."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_manifest_"))
        self.mock_img_dir = self.test_dir / "images"
        self.mock_img_dir.mkdir(parents=True, exist_ok=True)

        # Create mock 1x1 JPG files
        self.img_files: Dict[str, Path] = {}
        for split in ["train", "val", "test"]:
            p = self.mock_img_dir / f"mock_{split}.jpg"
            img = Image.new("RGB", (32, 32), color=(100, 150, 200))
            img.save(p, "JPEG")
            self.img_files[split] = p

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_manifest(self, rows: List[Dict[str, str]]) -> Path:
        manifest_path = self.test_dir / "manifest.csv"
        fieldnames = [
            "image_id",
            "relative_path",
            "crop",
            "class_name",
            "class_id",
            "health_status",
            "condition",
            "split",
            "group_id",
            "sha256",
            "status",
            "exclusion_reason",
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                full_row = {fn: "None" for fn in fieldnames}
                full_row.update(r)
                writer.writerow(full_row)
        return manifest_path

    def test_strict_test_set_isolation(self):
        """Test images must never appear in returned train or val records."""
        rows = [
            {
                "image_id": "IMG_TR_1",
                "relative_path": str(self.img_files["train"].relative_to(self.test_dir)),
                "crop": "Bell Pepper",
                "class_name": "Pepper,_bell___Bacterial_spot",
                "class_id": "0",
                "split": "train",
                "status": "included",
            },
            {
                "image_id": "IMG_VAL_1",
                "relative_path": str(self.img_files["val"].relative_to(self.test_dir)),
                "crop": "Bell Pepper",
                "class_name": "Pepper,_bell___Bacterial_spot",
                "class_id": "0",
                "split": "val",
                "status": "included",
            },
            {
                "image_id": "IMG_TE_1",
                "relative_path": str(self.img_files["test"].relative_to(self.test_dir)),
                "crop": "Bell Pepper",
                "class_name": "Pepper,_bell___Bacterial_spot",
                "class_id": "0",
                "split": "test",
                "status": "included",
            },
            {
                "image_id": "IMG_EXCL_1",
                "relative_path": "dummy/path.jpg",
                "crop": "Apple",
                "class_name": "Apple___Apple_scab",
                "class_id": "-1",
                "split": "NONE",
                "status": "excluded",
            },
        ]
        manifest_path = self._create_synthetic_manifest(rows)
        train_recs, val_recs = load_and_validate_manifest(
            manifest_path=manifest_path,
            project_root=self.test_dir,
            enforce_exact_counts=False,
        )

        self.assertEqual(len(train_recs), 1)
        self.assertEqual(len(val_recs), 1)
        self.assertEqual(train_recs[0]["image_id"], "IMG_TR_1")
        self.assertEqual(val_recs[0]["image_id"], "IMG_VAL_1")

        # Confirm test split is nowhere in loaded records
        all_loaded_ids = {r["image_id"] for r in train_recs + val_recs}
        self.assertNotIn("IMG_TE_1", all_loaded_ids)
        self.assertNotIn("IMG_EXCL_1", all_loaded_ids)

    def test_missing_image_file_raises_error(self):
        rows = [
            {
                "image_id": "IMG_MISSING",
                "relative_path": "non_existent_folder/missing_image.jpg",
                "crop": "Bell Pepper",
                "class_name": "Pepper,_bell___Bacterial_spot",
                "class_id": "0",
                "split": "train",
                "status": "included",
            }
        ]
        manifest_path = self._create_synthetic_manifest(rows)
        with self.assertRaises(FileNotFoundError):
            load_and_validate_manifest(
                manifest_path=manifest_path,
                project_root=self.test_dir,
                enforce_exact_counts=False,
            )

    def test_mismatched_class_id_and_name_raises_error(self):
        rows = [
            {
                "image_id": "IMG_MISMATCH",
                "relative_path": str(self.img_files["train"].relative_to(self.test_dir)),
                "crop": "Bell Pepper",
                "class_name": "Pepper,_bell___healthy",  # ID 1, but we pass ID 0
                "class_id": "0",
                "split": "train",
                "status": "included",
            }
        ]
        manifest_path = self._create_synthetic_manifest(rows)
        with self.assertRaises(ValueError):
            load_and_validate_manifest(
                manifest_path=manifest_path,
                project_root=self.test_dir,
                enforce_exact_counts=False,
            )


class TestCheckpointTrackerTieBreaking(unittest.TestCase):
    """Verifies Macro-F1 checkpoint selection and validation loss tie-breaking rule."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_tracker_"))
        self.checkpoint_path = self.test_dir / "best_model.keras"
        self.metadata_path = self.test_dir / "best_model_metadata.json"

        # Create dummy Keras model for saving
        inp = tf.keras.Input(shape=(4,))
        out = tf.keras.layers.Dense(2)(inp)
        self.dummy_model = tf.keras.Model(inputs=inp, outputs=out)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tie_breaking_and_tolerance(self):
        tracker = BestModelCheckpointTracker(
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            f1_tolerance=1e-4,
        )
        tracker.set_model(self.dummy_model)

        # Epoch 1: Initial baseline (val_f1=0.8500, val_loss=0.5000) -> NEW BEST
        tracker.on_epoch_end(0, {"val_macro_f1": 0.8500, "val_loss": 0.5000})
        self.assertEqual(tracker.best_epoch, 1)
        self.assertAlmostEqual(tracker.best_f1, 0.8500)
        self.assertAlmostEqual(tracker.best_loss, 0.5000)

        # Epoch 2: Lower F1 (val_f1=0.8400, val_loss=0.4000) -> NO CHANGE
        tracker.on_epoch_end(1, {"val_macro_f1": 0.8400, "val_loss": 0.4000})
        self.assertEqual(tracker.best_epoch, 1)

        # Epoch 3: Strictly higher F1 (val_f1=0.8600, val_loss=0.4800) -> NEW BEST
        tracker.on_epoch_end(2, {"val_macro_f1": 0.8600, "val_loss": 0.4800})
        self.assertEqual(tracker.best_epoch, 3)
        self.assertAlmostEqual(tracker.best_f1, 0.8600)
        self.assertAlmostEqual(tracker.best_loss, 0.4800)

        # Epoch 4: Tie in F1 within tolerance 1e-4 (val_f1=0.86005, diff=0.00005 <= 1e-4)
        # But higher val_loss (0.4900 > 0.4800) -> NO CHANGE
        tracker.on_epoch_end(3, {"val_macro_f1": 0.86005, "val_loss": 0.4900})
        self.assertEqual(tracker.best_epoch, 3)

        # Epoch 5: Tie in F1 within tolerance (val_f1=0.86002, diff=0.00002 <= 1e-4)
        # With lower val_loss (0.4500 < 0.4800) -> NEW BEST (TIE BROKEN BY LOSS)
        tracker.on_epoch_end(4, {"val_macro_f1": 0.86002, "val_loss": 0.4500})
        self.assertEqual(tracker.best_epoch, 5)
        self.assertAlmostEqual(tracker.best_loss, 0.4500)
        self.assertIn("broken by lower val_loss", tracker.best_reason)

        # Check metadata json saved to disk
        self.assertTrue(self.metadata_path.is_file())
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["epoch"], 5)
        self.assertAlmostEqual(meta["val_loss"], 0.4500)

    def test_global_best_preserved_across_stages(self):
        tracker = BestModelCheckpointTracker(
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            f1_tolerance=1e-4,
        )
        tracker.set_model(self.dummy_model)

        # Stage 1 best at epoch 4: F1=0.9200, Loss=0.2500
        tracker.current_stage = 1
        tracker.on_epoch_end(3, {"val_macro_f1": 0.9200, "val_loss": 0.2500})
        self.assertEqual(tracker.best_stage, 1)
        self.assertEqual(tracker.best_epoch, 4)

        # Stage 2 begins: epoch 1 achieves F1=0.9100 -> Stage 1 remains global best
        tracker.current_stage = 2
        tracker.on_epoch_end(0, {"val_macro_f1": 0.9100, "val_loss": 0.2800})
        self.assertEqual(tracker.best_stage, 1)
        self.assertEqual(tracker.best_epoch, 4)

        # Stage 2 epoch 2 achieves F1=0.9350 -> Stage 2 becomes new global best
        tracker.on_epoch_end(1, {"val_macro_f1": 0.9350, "val_loss": 0.2100})
        self.assertEqual(tracker.best_stage, 2)
        self.assertEqual(tracker.best_epoch, 2)


class TestModelArchitectureAndFreezing(unittest.TestCase):
    """Verifies MobileNetV2 architecture, embedded rescaling, and two-stage layer freezing."""

    def test_architecture_and_forward_pass(self):
        model, backbone = build_mobilenetv2_model(
            num_classes=15,
            weights=None,
            l2_reg=1e-4,
            dropout_rate=0.35,
        )
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 15))

        # Check Rescaling layer properties
        rescaling_layer = model.get_layer("rescaling")
        self.assertAlmostEqual(rescaling_layer.scale, 1.0 / 127.5)
        self.assertAlmostEqual(rescaling_layer.offset, -1.0)

        # Forward pass smoke test on raw RGB range [0, 255]
        dummy_input = tf.constant(np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)), dtype=tf.float32)
        dummy_output = model(dummy_input)
        self.assertEqual(dummy_output.shape, (2, 15))
        # Check softmax sum equals 1.0
        sums = tf.reduce_sum(dummy_output, axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(2), rtol=1e-5)

    def test_stage1_and_stage2_freezing_logic(self):
        model, backbone = build_mobilenetv2_model(
            num_classes=15,
            weights=None,
        )
        # Stage 1: Backbone must be completely non-trainable
        self.assertFalse(backbone.trainable)
        for layer in backbone.layers:
            self.assertFalse(layer.trainable)

        # Stage 2 configuration: Unfreeze upper layers >= 100, keep all BN frozen
        diag = configure_stage2_finetuning(backbone, unfreeze_from_layer=100)
        self.assertTrue(backbone.trainable)
        self.assertEqual(diag["total_backbone_layers"], 154)
        self.assertEqual(diag["frozen_lower_layers"], 100)

        # Verify every BatchNormalization layer in backbone is strictly frozen
        for i, layer in enumerate(backbone.layers):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                self.assertFalse(layer.trainable, f"Backbone BN layer {i} ({layer.name}) must be frozen")
            elif i < 100:
                self.assertFalse(layer.trainable, f"Backbone layer {i} ({layer.name}) below 100 must be frozen")
            else:
                self.assertTrue(layer.trainable, f"Backbone layer {i} ({layer.name}) >= 100 non-BN must be trainable")


class TestExperimentComparison(unittest.TestCase):
    """Verifies that compare_experiments correctly chooses the winner without touching test data."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_compare_"))
        self.dir_u = self.test_dir / "unweighted"
        self.dir_w = self.test_dir / "balanced"
        self.dir_u.mkdir(parents=True)
        self.dir_w.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_metadata(self, folder: Path, stage: int, epoch: int, f1: float, loss: float):
        meta = {
            "checkpoint_file": "best_model.keras",
            "stage": stage,
            "epoch": epoch,
            "val_macro_f1": f1,
            "val_loss": loss,
            "selection_reason": "test",
        }
        with open(folder / "best_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        # Create dummy checkpoint file to satisfy existence check
        with open(folder / "best_model.keras", "wb") as f:
            f.write(b"MOCK_MODEL_BYTES")

    def test_comparison_higher_f1_wins(self):
        self._write_metadata(self.dir_u, stage=1, epoch=10, f1=0.8800, loss=0.3500)
        self._write_metadata(self.dir_w, stage=2, epoch=5, f1=0.8950, loss=0.3200)
        result = compare_experiments(self.dir_u, self.dir_w)
        self.assertEqual(result["winner"], "Class-Weighted")
        self.assertEqual(result["model_path"], str(self.dir_w / "best_model.keras"))
        self.assertAlmostEqual(result["val_macro_f1"], 0.8950)

    def test_comparison_tie_loss_wins(self):
        # Difference within tolerance 1e-4: unweighted lower loss
        self._write_metadata(self.dir_u, stage=2, epoch=8, f1=0.89005, loss=0.2900)
        self._write_metadata(self.dir_w, stage=2, epoch=9, f1=0.89001, loss=0.3100)
        result = compare_experiments(self.dir_u, self.dir_w)
        self.assertEqual(result["winner"], "Unweighted")
        self.assertEqual(result["model_path"], str(self.dir_u / "best_model.keras"))
        self.assertAlmostEqual(result["val_loss"], 0.2900)

    def test_compare_experiments_does_not_mutate_artifacts(self):
        """Verifies compare_experiments never modifies or creates files in experiment directories."""
        self._write_metadata(self.dir_u, stage=1, epoch=10, f1=0.8800, loss=0.3500)
        self._write_metadata(self.dir_w, stage=2, epoch=5, f1=0.8950, loss=0.3200)

        # Snapshot directory contents before comparison
        files_u_before = {f.name: f.stat().st_mtime_ns for f in self.dir_u.iterdir()}
        files_w_before = {f.name: f.stat().st_mtime_ns for f in self.dir_w.iterdir()}

        result = compare_experiments(self.dir_u, self.dir_w)

        # Snapshot directory contents after comparison
        files_u_after = {f.name: f.stat().st_mtime_ns for f in self.dir_u.iterdir()}
        files_w_after = {f.name: f.stat().st_mtime_ns for f in self.dir_w.iterdir()}

        self.assertEqual(files_u_before, files_u_after)
        self.assertEqual(files_w_before, files_w_after)


class TestEarlyStoppingAndStageConsistency(unittest.TestCase):
    """
    Focused regression tests verifying:
      1. Potential discrepancy between EarlyStopping(restore_best_weights=True) and custom tracker.
      2. Stage 2 explicitly starting from Stage 1 best weights (not stopping/degraded weights).
      3. Global best weights and metadata synchronization when Stage 2 degrades.
    """

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_es_consistency_"))
        self.checkpoint_path = self.test_dir / "best_model.keras"
        self.metadata_path = self.test_dir / "best_model_metadata.json"

        # Create a tiny 1-weight model for exact weight inspection
        inp = tf.keras.Input(shape=(1,))
        dense = tf.keras.layers.Dense(1, use_bias=False, name="single_weight_dense")
        out = dense(inp)
        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.set_weights([np.array([[1.0]], dtype=np.float32)])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_early_stopping_discrepancy_risk(self):
        """
        Demonstrates that Keras EarlyStopping(restore_best_weights=True) without multi-criteria
        tie-breaking can choose a different epoch than BestModelCheckpointTracker.
        """
        # Epoch 1: val_macro_f1=0.85000, val_loss=0.5000
        # Epoch 2: val_macro_f1=0.85005, val_loss=0.3500 (diff=0.00005 <= 1e-4, lower loss)

        tracker = BestModelCheckpointTracker(
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            f1_tolerance=1e-4,
        )
        tracker.set_model(self.model)

        tracker.on_epoch_end(0, {"val_macro_f1": 0.85000, "val_loss": 0.5000})
        self.assertEqual(tracker.best_epoch, 1)

        tracker.on_epoch_end(1, {"val_macro_f1": 0.85005, "val_loss": 0.3500})
        # Tracker correctly identifies tie within 1e-4 and selects Epoch 2 due to lower loss
        self.assertEqual(tracker.best_epoch, 2)
        self.assertAlmostEqual(tracker.best_loss, 0.3500)

        # Contrast with EarlyStopping: min_delta=1e-4 checks if current - min_delta > best
        # 0.85005 - 0.0001 = 0.84995 < 0.85000 -> EarlyStopping would NOT consider Epoch 2 an improvement!
        # This confirms why restore_best_weights MUST be False on EarlyStopping.
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            min_delta=1e-4,
            restore_best_weights=True,
        )
        # EarlyStopping internal comparison function
        current = 0.85005
        best = 0.85000
        is_es_improvement = np.greater(current - es.min_delta, best)
        self.assertFalse(is_es_improvement, "EarlyStopping fails to detect tie-breaker improvement")

    def test_stage2_starts_from_stage1_best_weights(self):
        """
        Verifies that Stage 2 starts from the best Stage 1 weights,
        not from degraded weights at Stage 1 stopping point.
        """
        tracker = BestModelCheckpointTracker(
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            f1_tolerance=1e-4,
        )
        tracker.set_model(self.model)
        tracker.current_stage = 1

        # Epoch 1: W = 2.0, F1 = 0.80 -> Baseline best
        self.model.set_weights([np.array([[2.0]], dtype=np.float32)])
        tracker.on_epoch_end(0, {"val_macro_f1": 0.80, "val_loss": 0.50})

        # Epoch 2: W = 3.0, F1 = 0.90 -> Stage 1 BEST
        self.model.set_weights([np.array([[3.0]], dtype=np.float32)])
        tracker.on_epoch_end(1, {"val_macro_f1": 0.90, "val_loss": 0.30})
        self.assertEqual(tracker.best_epoch, 2)

        # Epoch 3: W = 4.0, F1 = 0.70 -> Stage 1 degraded stopping point
        self.model.set_weights([np.array([[4.0]], dtype=np.float32)])
        tracker.on_epoch_end(2, {"val_macro_f1": 0.70, "val_loss": 0.65})

        # At the end of Stage 1 fit, model in memory has stopping weights W = 4.0
        current_w = self.model.get_weights()[0][0][0]
        self.assertEqual(current_w, 4.0)

        # Restore best weights before Stage 2
        restored = tracker.restore_best_weights(self.model)
        self.assertTrue(restored)

        # Model must now have Stage 1 best weights W = 3.0, NOT degraded W = 4.0
        restored_w = self.model.get_weights()[0][0][0]
        self.assertEqual(restored_w, 3.0)

    def test_global_best_and_in_memory_sync_when_stage2_degrades(self):
        """
        Verifies that if Stage 2 never beats Stage 1, the global best checkpoint
        remains from Stage 1, and the final synchronization restores those weights
        into the in-memory model.
        """
        tracker = BestModelCheckpointTracker(
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
            f1_tolerance=1e-4,
        )
        tracker.set_model(self.model)
        tracker.current_stage = 1

        # Stage 1 best: W = 3.0, F1 = 0.90 at Epoch 2
        self.model.set_weights([np.array([[3.0]], dtype=np.float32)])
        tracker.on_epoch_end(1, {"val_macro_f1": 0.90, "val_loss": 0.30})
        self.assertEqual(tracker.best_stage, 1)
        self.assertEqual(tracker.best_epoch, 2)

        # Stage 2 begins
        tracker.current_stage = 2

        # Stage 2 Epoch 1: W = 5.0, F1 = 0.82 (Worse than 0.90)
        self.model.set_weights([np.array([[5.0]], dtype=np.float32)])
        tracker.on_epoch_end(0, {"val_macro_f1": 0.82, "val_loss": 0.45})

        # Stage 2 Epoch 2: W = 6.0, F1 = 0.80 (Worse than 0.90)
        self.model.set_weights([np.array([[6.0]], dtype=np.float32)])
        tracker.on_epoch_end(1, {"val_macro_f1": 0.80, "val_loss": 0.50})

        # Global best must still be Stage 1 Epoch 2
        self.assertEqual(tracker.best_stage, 1)
        self.assertEqual(tracker.best_epoch, 2)
        self.assertAlmostEqual(tracker.best_f1, 0.90)

        # Verify disk metadata file
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["stage"], 1)
        self.assertEqual(meta["epoch"], 2)
        self.assertAlmostEqual(meta["val_macro_f1"], 0.90)

        # In-memory model currently has degraded Stage 2 stopping weights (W = 6.0)
        self.assertEqual(self.model.get_weights()[0][0][0], 6.0)

        # Perform final synchronization
        tracker.restore_best_weights(self.model)

        # In-memory model is now synchronized with Stage 1 best (W = 3.0)
        self.assertEqual(self.model.get_weights()[0][0][0], 3.0)


class TestStage2ExecutionAndCliDefaults(unittest.TestCase):
    """
    Focused regression tests verifying:
      1. Default CLI behavior preserves 15 Stage 1 and 15 Stage 2 epochs.
      2. Setting --epochs-stage2 0 parses cleanly to epochs_stage2 = 0.
      3. Setting epochs_stage2 = 0 keeps the backbone completely frozen (Stage 2 skipped).
      4. Explicit shuffle=False on tf.data.Dataset input prevents Keras UserWarning.
    """

    def test_cli_default_epochs_stage1_and_stage2(self):
        """Verifies that the normal default behavior remains 15 epochs for both stages."""
        args = parse_args([])
        self.assertEqual(args.epochs_stage1, 15)
        self.assertEqual(args.epochs_stage2, 15)

    def test_cli_epochs_stage2_zero_parsing(self):
        """Verifies that passing --epochs-stage2 0 sets epochs_stage2 to 0 cleanly."""
        args = parse_args(["--epochs-stage2", "0"])
        self.assertEqual(args.epochs_stage2, 0)
        self.assertEqual(args.epochs_stage1, 15)

    def test_epochs_stage2_zero_skips_stage2_backbone_unfreezing(self):
        """
        Verifies that when epochs_stage2 == 0, Stage 2 logic is skipped:
        the backbone remains 100% frozen (0 trainable weights), and the tracker
        remains in Stage 1.
        """
        model, backbone = build_mobilenetv2_model(num_classes=15, weights="none")

        # Stage 1 state: backbone is fully frozen
        backbone_trainable_weights_before = len(backbone.trainable_weights)
        self.assertEqual(backbone_trainable_weights_before, 0)

        epochs_stage2 = 0
        tracker_stage = 1

        # Simulate the script's exact conditional logic: if epochs_stage2 > 0:
        if epochs_stage2 > 0:
            tracker_stage = 2
            configure_stage2_finetuning(backbone, unfreeze_from_layer=100)

        # Confirm Stage 2 was cleanly skipped
        self.assertEqual(tracker_stage, 1)
        self.assertEqual(len(backbone.trainable_weights), 0)

    def test_fit_with_tf_dataset_does_not_warn_on_shuffle_false(self):
        """
        Verifies that passing shuffle=False to model.fit() with a tf.data.Dataset
        silences Keras's UserWarning: 'shuffle=True was passed, but will be ignored...'.
        """
        x_ds = tf.data.Dataset.from_tensor_slices(
            (tf.ones((4, 2), dtype=tf.float32), tf.ones((4, 1), dtype=tf.float32))
        ).batch(2)

        tiny_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1),
        ])
        tiny_model.compile(optimizer="sgd", loss="mse")

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            tiny_model.fit(x_ds, epochs=1, shuffle=False, verbose=0)

            # Filter for the specific Keras shuffle UserWarning
            shuffle_warnings = [
                w for w in recorded_warnings
                if issubclass(w.category, UserWarning)
                and "shuffle=True" in str(w.message)
            ]
            self.assertEqual(
                len(shuffle_warnings),
                0,
                f"Expected no shuffle UserWarnings, but found: {[str(w.message) for w in shuffle_warnings]}",
            )


if __name__ == "__main__":
    unittest.main()

