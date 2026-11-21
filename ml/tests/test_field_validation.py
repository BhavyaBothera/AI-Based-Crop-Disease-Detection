#!/usr/bin/env python3
"""
Field Validation & Generalization Audit Test Suite
===================================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Automated unit and integration tests verifying:
  1. Authoritative 15-class mapping consistency across backend and ML evaluation.
  2. Strict manifest schema validation, crop-class alignment, and privacy/GPS rejection.
  3. Image integrity, corruption detection, and format verification.
  4. Exact duplicate and perceptual near-duplicate (dHash) detection.
  5. Cross-partition group leakage prevention (field_dev vs field_test).
  6. Inference preprocessing parity ([0.0, 255.0] float32, EXIF transposition).
  7. Deterministic evaluation metrics, 95% Wilson & bootstrap CIs, and low-support flags.
  8. Leaf gate screening, False Rejection Rate (FRR), and False Acceptance Rate (FAR).
  9. Non-destructive output file generation.

Usage:
  .venv/Scripts/python.exe -m unittest ml/tests/test_field_validation.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from backend.app.services.inference import TARGET_CLASS_MAPPING as BACKEND_CLASS_MAPPING
from backend.app.services.inference import preprocess_image as prod_preprocess_image
from ml.evaluation.field_validation.data_auditor import (
    DataQualityAuditor,
)
from ml.evaluation.field_validation.evaluator import (
    FieldEvaluator,
    InferenceAccounting,
    ModelOutputValidationError,
    preprocess_field_image,
    validate_model_output,
)
from ml.evaluation.field_validation.gate_evaluator import (
    GateIntegrityError,
    GateSampleRecord,
    LeafGateEvaluator,
)
from ml.evaluation.field_validation.path_resolver import (
    InvalidImageFileError,
    PathSecurityError,
    resolve_safe_image_path,
)
from ml.evaluation.field_validation.manifest_schema import (
    CROP_TO_CLASSES,
    GATE_CATEGORY_NON_LEAF,
    GATE_CATEGORY_OTHER_PLANT,
    GATE_CATEGORY_SUPPORTED_LEAF,
    TARGET_CLASS_MAPPING,
    VALID_GATE_CATEGORIES,
    FieldSampleRecord,
    ManifestDuplicateIdError,
    ManifestSchemaError,
    ManifestSecurityError,
    ManifestValidationError,
    load_and_validate_manifest,
    map_ground_truth_to_gate_category,
)
from ml.evaluation.field_validation.metrics import (
    calculate_field_metrics,
    wilson_score_interval,
)
from ml.evaluation.field_validation.runner import (
    generate_baseline_comparison,
    parse_controlled_baseline,
    run_field_validation,
)


class TestClassMappingConsistency(unittest.TestCase):
    """Verifies that field validation class mappings strictly match production backend."""

    def test_class_mapping_matches_backend(self):
        self.assertEqual(len(TARGET_CLASS_MAPPING), 15)
        self.assertEqual(len(BACKEND_CLASS_MAPPING), 15)
        for class_name, meta in TARGET_CLASS_MAPPING.items():
            self.assertIn(class_name, BACKEND_CLASS_MAPPING)
            self.assertEqual(meta["id"], BACKEND_CLASS_MAPPING[class_name]["id"])
            self.assertEqual(meta["crop"], BACKEND_CLASS_MAPPING[class_name]["crop"])
            self.assertEqual(meta["condition"], BACKEND_CLASS_MAPPING[class_name]["condition"])
            self.assertEqual(meta["health"], BACKEND_CLASS_MAPPING[class_name]["health"])

    def test_class_ids_contiguous_0_to_14(self):
        ids = sorted(v["id"] for v in TARGET_CLASS_MAPPING.values())
        self.assertEqual(ids, list(range(15)))

    def test_crop_class_distribution(self):
        self.assertEqual(len(CROP_TO_CLASSES["Tomato"]), 10)
        self.assertEqual(len(CROP_TO_CLASSES["Potato"]), 3)
        self.assertEqual(len(CROP_TO_CLASSES["Bell Pepper"]), 2)


class TestManifestSchemaAndValidation(unittest.TestCase):
    """Verifies manifest parsing, schema checks, and privacy safeguards."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.manifest_path = self.test_dir / "test_manifest.csv"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_csv(self, rows: List[Dict[str, Any]]):
        if not rows:
            self.manifest_path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_manifest_record(self):
        row = {
            "image_id": "FLD_TOM_001",
            "file_path": "images/fld_tom_001.jpg",
            "crop": "Tomato",
            "ground_truth_class": "Tomato___Early_blight",
            "collection_source": "pilot_field_trial",
            "capture_device": "Samsung_Galaxy_S21",
            "capture_date": "2026-10-15",
            "collection_location": "Research Farm Block B, State College",
            "lighting_condition": "direct_sun",
            "background_type": "foliage_clutter",
            "disease_severity": "moderate",
            "annotator_status": "expert_pathologist",
            "verification_status": "expert_verified",
            "split": "field_dev",
            "duplicate_group_id": "PLANT_TOM_102",
            "notes": "Single leaflet framed against canopy.",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, "FLD_TOM_001")
        self.assertEqual(records[0].crop, "Tomato")
        self.assertEqual(records[0].ground_truth_class, "Tomato___Early_blight")

    def test_reject_crop_class_mismatch(self):
        # Mismatch: Bell Pepper crop declared, but Potato class given
        row = {
            "image_id": "FLD_ERR_001",
            "file_path": "images/test.jpg",
            "crop": "Bell Pepper",
            "ground_truth_class": "Potato___Late_blight",
            "collection_source": "pilot",
            "split": "field_dev",
            "annotator_status": "expert_pathologist",
            "verification_status": "expert_verified",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("does not belong to crop 'Bell Pepper'" in str(e) for e in errors))

    def test_reject_gps_coordinates_for_privacy(self):
        # Precise GPS coordinates must be strictly rejected
        row = {
            "image_id": "FLD_PRIV_001",
            "file_path": "images/test.jpg",
            "crop": "Tomato",
            "ground_truth_class": "Tomato___healthy",
            "collection_source": "pilot",
            "collection_location": "GPS: 40.7128° N, 74.0060° W, Farm #42",
            "split": "field_dev",
            "annotator_status": "trained_agronomist",
            "verification_status": "expert_verified",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("precise gps coordinates" in str(e).lower() for e in errors))

    def test_reject_invalid_split(self):
        # Only 'field_dev' and 'field_test' are valid. 'train' or 'test' are rejected.
        row = {
            "image_id": "FLD_SPLIT_001",
            "file_path": "images/test.jpg",
            "crop": "Tomato",
            "ground_truth_class": "Tomato___healthy",
            "collection_source": "pilot",
            "split": "training",  # Invalid
            "annotator_status": "trained_agronomist",
            "verification_status": "expert_verified",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("Invalid split 'training'" in str(e) for e in errors))


class TestDataQualityAndLeakageAuditor(unittest.TestCase):
    """Verifies data corruption, exact duplicates, perceptual near-duplicates, and leakage audits."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.img_dir = self.test_dir / "images"
        self.img_dir.mkdir()
        self.manifest_path = self.test_dir / "manifest.csv"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_synthetic_image(self, name: str, color=(100, 150, 50), size=(300, 300)) -> Path:
        img = Image.new("RGB", size, color=color)
        p = self.img_dir / name
        img.save(p, "JPEG")
        return p

    def test_corrupt_and_missing_images_detection(self):
        # 1 valid, 1 missing, 1 corrupt
        p_valid = self._create_synthetic_image("valid.jpg")
        p_corrupt = self.img_dir / "corrupt.jpg"
        p_corrupt.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"randomcorruptedbytes")

        rows = [
            {
                "image_id": "IMG_1",
                "file_path": "images/valid.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "split": "field_dev",
                "annotator_status": "expert_pathologist",
                "verification_status": "expert_verified",
            },
            {
                "image_id": "IMG_2",
                "file_path": "images/missing.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "split": "field_dev",
                "annotator_status": "expert_pathologist",
                "verification_status": "expert_verified",
            },
            {
                "image_id": "IMG_3",
                "file_path": "images/corrupt.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "split": "field_dev",
                "annotator_status": "expert_pathologist",
                "verification_status": "expert_verified",
            },
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = DataQualityAuditor(self.manifest_path, self.test_dir)
        report = auditor.audit_all()

        self.assertEqual(len(report.missing_files), 1)
        self.assertEqual(report.missing_files[0]["image_id"], "IMG_2")
        self.assertEqual(len(report.corrupt_images), 1)
        self.assertEqual(report.corrupt_images[0]["image_id"], "IMG_3")

    def test_exact_and_perceptual_duplicate_detection(self):
        # Create an original image
        p1 = self._create_synthetic_image("orig.jpg", color=(120, 160, 40))
        # Exact copy
        p2 = self.img_dir / "exact_copy.jpg"
        p2.write_bytes(p1.read_bytes())
        # Perceptually near-identical copy (tiny pixel shift)
        img_near = Image.open(p1).copy()
        img_near.putpixel((10, 10), (121, 159, 41))
        p3 = self.img_dir / "near_copy.jpg"
        img_near.save(p3, "JPEG")

        rows = [
            {"image_id": "I1", "file_path": "images/orig.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev", "annotator_status": "e", "verification_status": "expert_verified"},
            {"image_id": "I2", "file_path": "images/exact_copy.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev", "annotator_status": "e", "verification_status": "expert_verified"},
            {"image_id": "I3", "file_path": "images/near_copy.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev", "annotator_status": "e", "verification_status": "expert_verified"},
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = DataQualityAuditor(self.manifest_path, self.test_dir)
        report = auditor.audit_all()

        self.assertEqual(len(report.exact_duplicates), 1)
        self.assertIn("I1", report.exact_duplicates[0]["image_ids"])
        self.assertIn("I2", report.exact_duplicates[0]["image_ids"])

        # Perceptual near duplicates should detect I1/I2 and I3
        self.assertGreaterEqual(len(report.near_duplicates), 1)

    def test_cross_partition_group_leakage(self):
        # Same duplicate_group_id across field_dev and field_test
        p1 = self._create_synthetic_image("leaf1.jpg", color=(50, 120, 60))
        p2 = self._create_synthetic_image("leaf2.jpg", color=(60, 130, 70))

        rows = [
            {"image_id": "DEV_1", "file_path": "images/leaf1.jpg", "crop": "Potato", "ground_truth_class": "Potato___healthy", "collection_source": "p", "split": "field_dev", "duplicate_group_id": "PLANT_POTATO_77", "annotator_status": "e", "verification_status": "expert_verified"},
            {"image_id": "TEST_1", "file_path": "images/leaf2.jpg", "crop": "Potato", "ground_truth_class": "Potato___healthy", "collection_source": "p", "split": "field_test", "duplicate_group_id": "PLANT_POTATO_77", "annotator_status": "e", "verification_status": "expert_verified"},
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        auditor = DataQualityAuditor(self.manifest_path, self.test_dir)
        report = auditor.audit_all()

        self.assertEqual(len(report.group_leaks), 1)
        self.assertEqual(report.group_leaks[0]["group_id"], "PLANT_POTATO_77")
        self.assertIn("DEV_1", report.group_leaks[0]["image_ids"])
        self.assertIn("TEST_1", report.group_leaks[0]["image_ids"])


class TestPreprocessingParity(unittest.TestCase):
    """Verifies that field preprocessing produces raw float32 [0.0, 255.0] matching production backend."""

    def test_preprocessing_output_specs(self):
        # Create synthetic PIL image
        img = Image.new("RGB", (320, 240), color=(140, 190, 80))
        # Add some variation so standard deviation is > 3.0
        for x in range(50):
            for y in range(50):
                img.putpixel((x, y), (20, 50, 10))

        frame = preprocess_field_image(img, enforce_level_a=True)
        self.assertEqual(frame.shape, (1, 224, 224, 3))
        self.assertEqual(frame.dtype, np.float32)

        # Range must strictly be in [0.0, 255.0] (raw RGB pixels)
        self.assertGreaterEqual(float(frame.min()), 0.0)
        self.assertLessEqual(float(frame.max()), 255.0)
        self.assertGreater(float(frame.max()), 1.0)  # Verifies it's NOT normalized to [0, 1]

    def test_uniform_canvas_rejection(self):
        # Completely solid white canvas
        img = Image.new("RGB", (250, 250), color=(255, 255, 255))
        with self.assertRaises(ValueError) as ctx:
            preprocess_field_image(img, enforce_level_a=True)
        self.assertTrue("canvas" in str(ctx.exception).lower() or "blank" in str(ctx.exception).lower())

    def test_undersized_image_rejection(self):
        # Below 64x64 minimum floor
        img = Image.new("RGB", (50, 50), color=(100, 150, 50))
        with self.assertRaises(ValueError) as ctx:
            preprocess_field_image(img, enforce_level_a=True)
        self.assertTrue("small" in str(ctx.exception).lower() or "resolution" in str(ctx.exception).lower())


class TestMetricCalculations(unittest.TestCase):
    """Verifies precision, recall, Macro-F1, Wilson score intervals, and subgroup slicing."""

    def test_wilson_score_interval(self):
        # 100 trials, 90 successes
        ci_low, ci_high = wilson_score_interval(90, 100, confidence=0.95)
        self.assertAlmostEqual(ci_low, 0.8258, places=3)
        self.assertAlmostEqual(ci_high, 0.9452, places=3)

        # Boundary checks
        self.assertEqual(wilson_score_interval(0, 0), (0.0, 0.0))
        ci_zero = wilson_score_interval(0, 50)
        self.assertEqual(ci_zero[0], 0.0)
        self.assertGreater(ci_zero[1], 0.0)

    def test_field_metrics_calculation(self):
        # Synthetic evaluation batch of 10 samples
        predictions = [
            {
                "image_id": f"IMG_{i}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy" if i < 8 else "Tomato___Early_blight",
                "is_correct": i < 8,
                "top_3_predictions": [
                    {"class_name": "Tomato___healthy" if i < 8 else "Tomato___Early_blight"},
                    {"class_name": "Tomato___healthy"},
                    {"class_name": "Tomato___Septoria_leaf_spot"},
                ],
                "raw_scores": {"Tomato___healthy": 0.9 if i < 8 else 0.2, "Tomato___Early_blight": 0.1 if i < 8 else 0.8},
                "lighting_condition": "direct_sun",
                "background_type": "foliage_clutter",
                "disease_severity": "none",
                "capture_device": "Pixel_7",
                "verification_status": "expert_verified",
                "duplicate_group_id": f"PLANT_{i // 2}",
            }
            for i in range(10)
        ]

        summary = calculate_field_metrics(predictions)
        self.assertEqual(summary.total_samples, 10)
        self.assertEqual(summary.correct_samples, 8)
        self.assertEqual(summary.accuracy, 0.8)
        # Top-3 should include both Tomato___healthy and Tomato___Early_blight
        self.assertEqual(summary.top_3_accuracy, 1.0)
        self.assertIn("Tomato", summary.per_crop)
        self.assertEqual(summary.per_crop["Tomato"]["support"], 10)
        self.assertEqual(summary.per_crop["Tomato"]["accuracy"], 0.8)

        # Subgroup low-sample warning (10 samples <= threshold)
        self.assertIn("lighting_condition", summary.subgroup_breakdowns)
        self.assertIn("direct_sun", summary.subgroup_breakdowns["lighting_condition"])


class TestLeafGateEvaluator(unittest.TestCase):
    """Verifies leaf-acceptance gate metrics and False Rejection / False Acceptance calculations."""

    def test_gate_metrics_aggregation(self):
        # Mock LeafGateService that always returns high confidence for supported_leaf
        class MockGateService:
            def predict_proba(self, frame):
                return {"non_leaf": 0.05, "other_plant": 0.05, "supported_leaf": 0.90}

        evaluator = LeafGateEvaluator(leaf_gate_service=MockGateService())

        # Test sample evaluation with mock images
        test_dir = Path(tempfile.mkdtemp())
        try:
            img = Image.new("RGB", (250, 250), color=(80, 160, 40))
            # Put slight variation to pass detail check
            for x in range(30):
                img.putpixel((x, x), (20, 50, 10))
            img_path = test_dir / "leaf.jpg"
            img.save(img_path, "JPEG")

            samples = [
                GateSampleRecord(
                    sample_id="G1",
                    file_path=str(img_path),
                    ground_truth_category="supported_leaf",
                    crop="Tomato",
                ),
                GateSampleRecord(
                    sample_id="G2",
                    file_path=str(img_path),
                    ground_truth_category="non_leaf",
                ),
            ]

            results = evaluator.evaluate_dataset(samples)
            gm = results["gate_metrics"]

            # Supported leaf: accepted => FRR = 0
            self.assertEqual(gm["supported_leaves"]["support"], 1)
            self.assertEqual(gm["supported_leaves"]["accepted"], 1)
            self.assertEqual(gm["supported_leaves"]["false_rejection_rate"], 0.0)

            # Non-leaf: falsely accepted => FAR = 1.0
            self.assertEqual(gm["non_leaf_objects"]["support"], 1)
            self.assertEqual(gm["non_leaf_objects"]["accepted"], 1)
            self.assertEqual(gm["non_leaf_objects"]["false_acceptance_rate"], 1.0)
        finally:
            shutil.rmtree(test_dir)


class TestGateGroundTruthMapping(unittest.TestCase):
    """Bug 1 Regression Tests: Gate Ground-Truth Labels Mapping."""

    def test_supported_leaf_crops_mapping(self):
        # Supported crop leaves must map to supported_leaf
        self.assertEqual(map_ground_truth_to_gate_category("Tomato___healthy", "Tomato"), GATE_CATEGORY_SUPPORTED_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("Tomato___Early_blight", "Tomato"), GATE_CATEGORY_SUPPORTED_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("Potato___Early_blight", "Potato"), GATE_CATEGORY_SUPPORTED_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("Pepper,_bell___Bacterial_spot", "Bell Pepper"), GATE_CATEGORY_SUPPORTED_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("supported_leaf"), GATE_CATEGORY_SUPPORTED_LEAF)

    def test_other_plant_species_mapping(self):
        # Unsupported plant species must map to other_plant
        self.assertEqual(map_ground_truth_to_gate_category("Apple___Apple_scab", "Apple"), GATE_CATEGORY_OTHER_PLANT)
        self.assertEqual(map_ground_truth_to_gate_category("Corn___Common_rust", "Corn"), GATE_CATEGORY_OTHER_PLANT)
        self.assertEqual(map_ground_truth_to_gate_category("Grape___Black_rot", "Grape"), GATE_CATEGORY_OTHER_PLANT)
        self.assertEqual(map_ground_truth_to_gate_category("Weed_Dandelion", "Weed"), GATE_CATEGORY_OTHER_PLANT)
        self.assertEqual(map_ground_truth_to_gate_category("other_plant"), GATE_CATEGORY_OTHER_PLANT)

    def test_non_leaf_mapping(self):
        # Non-leaf objects must map to non_leaf
        self.assertEqual(map_ground_truth_to_gate_category("non_leaf"), GATE_CATEGORY_NON_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("background_soil"), GATE_CATEGORY_NON_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("glove_or_hand"), GATE_CATEGORY_NON_LEAF)
        self.assertEqual(map_ground_truth_to_gate_category("irrigation_pipe"), GATE_CATEGORY_NON_LEAF)

    def test_unknown_and_ambiguous_rejected(self):
        # Unknown/ambiguous records must NOT silently default to supported_leaf
        with self.assertRaises(ValueError):
            map_ground_truth_to_gate_category("completely_unknown_token", "UnknownCrop")
        with self.assertRaises(ValueError):
            map_ground_truth_to_gate_category("", "")

    def test_gate_evaluator_rejects_invalid_category(self):
        class MockService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            def predict_proba(self, frame):
                return {"non_leaf": 0.1, "other_plant": 0.1, "supported_leaf": 0.8}

        evaluator = LeafGateEvaluator(leaf_gate_service=MockService())
        test_dir = Path(tempfile.mkdtemp())
        try:
            img = Image.new("RGB", (200, 200), color=(50, 150, 50))
            for i in range(20):
                img.putpixel((i, i), (10, 30, 10))
            p = test_dir / "t.jpg"
            img.save(p, "JPEG")
            sample = GateSampleRecord("S1", str(p), ground_truth_category="invalid_category")
            with self.assertRaises(ValueError):
                evaluator.evaluate_sample(sample)
        finally:
            shutil.rmtree(test_dir)


class TestPreprocessingParityComprehensive(unittest.TestCase):
    """Bug 2 Regression Tests: Exact preprocessing parity between production backend and evaluation."""

    def _create_synthetic_bytes(self, fmt: str = "JPEG", mode: str = "RGB", size=(250, 250), orientation: int = None) -> bytes:
        img = Image.new(mode, size, color=(100, 160, 50) if mode != "L" else 128)
        # Add texture so it is not a uniform canvas
        for x in range(40):
            for y in range(40):
                if mode == "RGB":
                    img.putpixel((x, y), (20, 50, 10))
                elif mode == "RGBA":
                    img.putpixel((x, y), (20, 50, 10, 255))
                elif mode == "L":
                    img.putpixel((x, y), 35)

        buf = io.BytesIO()
        if orientation is not None:
            exif = img.getexif()
            exif[0x0112] = orientation
            img.save(buf, format=fmt, exif=exif)
        else:
            img.save(buf, format=fmt)
        return buf.getvalue()

    def test_jpeg_parity(self):
        b = self._create_synthetic_bytes("JPEG", "RGB")
        prod_res = prod_preprocess_image(b, enforce_level_a=True)
        eval_res = preprocess_field_image(b, enforce_level_a=True)
        self.assertEqual(prod_res.shape, (1, 224, 224, 3))
        self.assertEqual(prod_res.dtype, np.float32)
        self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))
        self.assertGreater(float(prod_res.max()), 1.0)
        self.assertLessEqual(float(prod_res.max()), 255.0)

    def test_png_parity(self):
        b = self._create_synthetic_bytes("PNG", "RGB")
        prod_res = prod_preprocess_image(b, enforce_level_a=True)
        eval_res = preprocess_field_image(b, enforce_level_a=True)
        self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))

    def test_webp_parity(self):
        b = self._create_synthetic_bytes("WEBP", "RGB")
        prod_res = prod_preprocess_image(b, enforce_level_a=True)
        eval_res = preprocess_field_image(b, enforce_level_a=True)
        self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))

    def test_exif_rotated_parity(self):
        for orient in (3, 6, 8):
            b = self._create_synthetic_bytes("JPEG", "RGB", orientation=orient)
            prod_res = prod_preprocess_image(b, enforce_level_a=True)
            eval_res = preprocess_field_image(b, enforce_level_a=True)
            self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))

    def test_grayscale_parity(self):
        b = self._create_synthetic_bytes("PNG", "L")
        prod_res = prod_preprocess_image(b, enforce_level_a=True)
        eval_res = preprocess_field_image(b, enforce_level_a=True)
        self.assertEqual(prod_res.shape, (1, 224, 224, 3))
        self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))

    def test_rgba_parity(self):
        b = self._create_synthetic_bytes("PNG", "RGBA")
        prod_res = prod_preprocess_image(b, enforce_level_a=True)
        eval_res = preprocess_field_image(b, enforce_level_a=True)
        self.assertEqual(prod_res.shape, (1, 224, 224, 3))
        self.assertTrue(np.allclose(prod_res, eval_res, atol=1e-5))

    def test_small_image_rejection_parity(self):
        b = self._create_synthetic_bytes("JPEG", "RGB", size=(40, 40))
        with self.assertRaises(ValueError):
            prod_preprocess_image(b, enforce_level_a=True)
        with self.assertRaises(ValueError):
            preprocess_field_image(b, enforce_level_a=True)

    def test_corrupted_image_rejection_parity(self):
        corrupt = b"\xFF\xD8\xFF\xE0\x00\x10JFIFbadbytes"
        with self.assertRaises(ValueError):
            prod_preprocess_image(corrupt, enforce_level_a=True)
        with self.assertRaises(ValueError):
            preprocess_field_image(corrupt, enforce_level_a=True)

    def test_uniform_canvas_rejection_parity(self):
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b = buf.getvalue()
        with self.assertRaises(ValueError):
            prod_preprocess_image(b, enforce_level_a=True)
        with self.assertRaises(ValueError):
            preprocess_field_image(b, enforce_level_a=True)


class TestBaselineComparison(unittest.TestCase):
    """Bug 3 Regression Tests: Baseline report parsing and comparative reporting."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_parse_valid_json_baseline(self):
        baseline_file = self.test_dir / "baseline.json"
        data = {
            "evaluation_name": "Test Run",
            "timestamp_utc": "2026-10-20T12:00:00Z",
            "model_checkpoint": "best_model.keras",
            "split_evaluated": "test",
            "metrics": {
                "test_accuracy": 0.955,
                "test_macro_f1": 0.945,
                "test_top_3_accuracy": 0.995,
            },
        }
        baseline_file.write_text(json.dumps(data), encoding="utf-8")
        parsed = parse_controlled_baseline(baseline_file)
        self.assertAlmostEqual(parsed["accuracy"], 0.955)
        self.assertAlmostEqual(parsed["macro_f1"], 0.945)
        self.assertAlmostEqual(parsed["top_3_accuracy"], 0.995)
        self.assertEqual(parsed["model_checkpoint"], "best_model.keras")
        self.assertEqual(parsed["timestamp"], "2026-10-20T12:00:00Z")

    def test_parse_real_json_report(self):
        real_json = Path("docs/audit_reports/final_test_evaluation_report.json")
        if real_json.is_file():
            parsed = parse_controlled_baseline(real_json)
            self.assertAlmostEqual(parsed["accuracy"], 0.968942, places=4)
            self.assertAlmostEqual(parsed["macro_f1"], 0.965288, places=4)
            self.assertIn("best_model.keras", parsed["model_checkpoint"])

    def test_parse_real_md_report(self):
        real_md = Path("docs/audit_reports/final_test_evaluation_report.md")
        if real_md.is_file():
            parsed = parse_controlled_baseline(real_md)
            self.assertAlmostEqual(parsed["accuracy"], 0.968942, places=4)
            self.assertAlmostEqual(parsed["macro_f1"], 0.965288, places=4)

    def test_reject_missing_baseline_file(self):
        with self.assertRaises(FileNotFoundError):
            parse_controlled_baseline(self.test_dir / "nonexistent.json")

    def test_reject_malformed_json_baseline(self):
        # Missing required macro_f1
        f = self.test_dir / "bad.json"
        f.write_text(json.dumps({"split_evaluated": "test", "model_checkpoint": "m.keras", "metrics": {"test_accuracy": 0.90}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            parse_controlled_baseline(f)

    def test_generate_baseline_comparison_no_hardcoding(self):
        f = self.test_dir / "baseline.json"
        data = {
            "split_evaluated": "test",
            "metrics": {"test_accuracy": 0.92, "test_macro_f1": 0.91, "test_top_3_accuracy": 0.98},
            "model_checkpoint": "model_v1.keras",
            "timestamp_utc": "2026-10-20",
        }
        f.write_text(json.dumps(data), encoding="utf-8")

        field_results = {
            "metadata": {
                "evaluation_timestamp": "2026-11-20T00:00:00Z",
                "is_formal_evaluation": True,
                "evaluation_status": "FORMAL_PASSED",
            },
            "metrics": {"accuracy": 0.85, "macro_f1": 0.83, "top_3_accuracy": 0.95},
        }
        md = generate_baseline_comparison(f, field_results)
        self.assertIn("92.00%", md)
        self.assertIn("0.9100", md)
        self.assertIn("-7.00%", md)
        self.assertIn("model_v1.keras", md)


class TestVerificationPolicy(unittest.TestCase):
    """Bug 4 Regression Tests: Verification tiers and default exclusion of unverified samples."""

    def test_verification_status_changes_metric_population(self):
        predictions = [
            {
                "image_id": f"IMG_V_{i}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy",
                "is_correct": True,
                "top_3_predictions": [{"class_name": "Tomato___healthy"}],
                "raw_scores": {"Tomato___healthy": 0.95},
                "verification_status": "expert_verified",
            }
            for i in range(3)
        ] + [
            {
                "image_id": f"IMG_U_{j}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___Early_blight",
                "is_correct": False,
                "top_3_predictions": [{"class_name": "Tomato___Early_blight"}],
                "raw_scores": {"Tomato___Early_blight": 0.85},
                "verification_status": "visually_suspected",
            }
            for j in range(2)
        ]

        # By default exclude_unverified=True
        summary_default = calculate_field_metrics(predictions)
        self.assertEqual(summary_default.total_samples, 3)
        self.assertEqual(summary_default.accuracy, 1.0)
        self.assertEqual(summary_default.verified_sample_count, 3)
        self.assertEqual(summary_default.excluded_unverified_count, 2)
        self.assertEqual(summary_default.unverified_breakdown.get("visually_suspected"), 2)

        # When exclude_unverified=False, all 5 are included
        summary_all = calculate_field_metrics(predictions, exclude_unverified=False)
        self.assertEqual(summary_all.total_samples, 5)
        self.assertAlmostEqual(summary_all.accuracy, 3 / 5)
        self.assertEqual(summary_all.excluded_unverified_count, 0)


class TestDiseaseVsGateSeparation(unittest.TestCase):
    """Bug 5 Regression Tests: Separation of disease vs gate tasks and label spaces."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.manifest_path = self.test_dir / "manifest.csv"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_csv(self, rows: List[Dict[str, Any]]):
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_invalid_evaluation_task_rejected(self):
        row = {
            "image_id": "T1",
            "file_path": "images/t.jpg",
            "crop": "Tomato",
            "ground_truth_class": "Tomato___healthy",
            "collection_source": "pilot",
            "split": "field_dev",
            "evaluation_task": "invalid_task",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("Invalid evaluation_task 'invalid_task'" in str(e) for e in errors))

    def test_unsupported_crop_with_disease_task_rejected(self):
        # Apple is not in Solanaceae 15-class space; cannot have evaluation_task='disease'
        row = {
            "image_id": "T2",
            "file_path": "images/apple.jpg",
            "crop": "Apple",
            "ground_truth_class": "Apple___Apple_scab",
            "collection_source": "pilot",
            "split": "field_dev",
            "evaluation_task": "disease",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("cannot have evaluation_task 'disease'" in str(e) for e in errors))

    def test_unsupported_crop_with_gate_task_accepted(self):
        # Apple with evaluation_task='gate' is completely valid!
        row = {
            "image_id": "T3",
            "file_path": "images/apple.jpg",
            "crop": "Apple",
            "ground_truth_class": "Apple___Apple_scab",
            "collection_source": "pilot",
            "split": "field_dev",
            "evaluation_task": "gate",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].evaluation_task, "gate")


class TestManifestAndDatasetSafety(unittest.TestCase):
    """Bug 6 Regression Tests: Manifest uniqueness, path safety, and duplicate group consistency."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.img_dir = self.test_dir / "images"
        self.img_dir.mkdir()
        self.manifest_path = self.test_dir / "manifest.csv"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_csv(self, rows: List[Dict[str, Any]]):
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_duplicate_image_id_rejected(self):
        rows = [
            {"image_id": "DUP_ID", "file_path": "images/1.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev"},
            {"image_id": "DUP_ID", "file_path": "images/2.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev"},
        ]
        self._write_csv(rows)
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("Duplicate image_id 'DUP_ID'" in str(e) for e in errors))

    def test_path_traversal_rejected_in_manifest(self):
        row = {
            "image_id": "TRAV_1",
            "file_path": "../evil/path.jpg",
            "crop": "Tomato",
            "ground_truth_class": "Tomato___healthy",
            "collection_source": "p",
            "split": "field_dev",
        }
        self._write_csv([row])
        records, errors = load_and_validate_manifest(self.manifest_path)
        self.assertTrue(any("traversal" in str(e).lower() for e in errors))

    def test_duplicate_group_metadata_inconsistency_audited(self):
        # Two images in same plant group but different crop
        img = Image.new("RGB", (100, 100), color=(80, 150, 40))
        (self.img_dir / "1.jpg").write_bytes(b"image1")
        (self.img_dir / "2.jpg").write_bytes(b"image2")
        rows = [
            {"image_id": "S1", "file_path": "images/1.jpg", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "collection_source": "p", "split": "field_dev", "duplicate_group_id": "PLANT_GRP_99"},
            {"image_id": "S2", "file_path": "images/2.jpg", "crop": "Potato", "ground_truth_class": "Potato___healthy", "collection_source": "p", "split": "field_dev", "duplicate_group_id": "PLANT_GRP_99"},
        ]
        self._write_csv(rows)
        auditor = DataQualityAuditor(self.manifest_path, self.test_dir)
        report = auditor.audit_all()
        self.assertGreaterEqual(len(report.duplicate_group_inconsistencies), 1)
        self.assertEqual(report.duplicate_group_inconsistencies[0]["group_id"], "PLANT_GRP_99")
        self.assertFalse(report.is_clean)


class TestMetricReliability(unittest.TestCase):
    """Bug 7 Regression Tests: Metric calculations reliability, missing classes, empty datasets, and cluster bootstrap."""

    def test_empty_dataset_handling(self):
        summary = calculate_field_metrics([])
        self.assertEqual(summary.total_samples, 0)
        self.assertEqual(summary.accuracy, 0.0)
        self.assertEqual(summary.macro_f1, 0.0)
        self.assertEqual(summary.accuracy_ci_95, (0.0, 0.0))
        self.assertEqual(len(summary.unobserved_classes), 15)

    def test_single_class_dataset_handling(self):
        predictions = [
            {
                "image_id": f"IMG_{i}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy",
                "is_correct": True,
                "top_3_predictions": [{"class_name": "Tomato___healthy"}],
                "raw_scores": {"Tomato___healthy": 0.95},
                "duplicate_group_id": f"PLANT_{i % 2}",
            }
            for i in range(10)
        ]
        summary = calculate_field_metrics(predictions)
        self.assertEqual(summary.total_samples, 10)
        self.assertEqual(summary.observed_classes_count, 1)
        self.assertEqual(len(summary.unobserved_classes), 14)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.macro_f1, 1.0)

    def test_nan_infinite_scores_safe_handling(self):
        predictions = [
            {
                "image_id": "IMG_NAN",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy",
                "is_correct": True,
                "top_3_predictions": [{"class_name": "Tomato___healthy"}],
                "raw_scores": {"Tomato___healthy": float("nan"), "Tomato___Early_blight": float("inf")},
            }
        ]
        summary = calculate_field_metrics(predictions)
        self.assertEqual(summary.total_samples, 1)
        self.assertFalse(math.isnan(summary.accuracy))

    def test_cluster_aware_metrics(self):
        predictions = [
            {"image_id": "1", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "predicted_class": "Tomato___healthy", "is_correct": True, "top_3_predictions": [{"class_name": "Tomato___healthy"}], "duplicate_group_id": "PLANT_A"},
            {"image_id": "2", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "predicted_class": "Tomato___healthy", "is_correct": True, "top_3_predictions": [{"class_name": "Tomato___healthy"}], "duplicate_group_id": "PLANT_A"},
            {"image_id": "3", "crop": "Tomato", "ground_truth_class": "Tomato___healthy", "predicted_class": "Tomato___Early_blight", "is_correct": False, "top_3_predictions": [{"class_name": "Tomato___Early_blight"}], "duplicate_group_id": "PLANT_B"},
        ]
        summary = calculate_field_metrics(predictions)
        self.assertIn("cluster_accuracy", summary.cluster_metrics)
        self.assertEqual(summary.cluster_metrics["cluster_count"], 2)


class TestGateFallbackTransparency(unittest.TestCase):
    """Bug 8 Regression Tests: Fallback transparency and sidecar validation."""

    def test_deterministic_fallback_identification_and_warning(self):
        evaluator = LeafGateEvaluator(gate_model_path="non_existent_gate_model.keras")
        self.assertFalse(evaluator.is_trained_model)
        self.assertEqual(evaluator.gate_backend, "deterministic_fallback")

        results = evaluator.evaluate_dataset([])
        self.assertEqual(results["metadata"]["gate_backend"], "deterministic_fallback")
        self.assertIn("warning", results["metadata"])
        self.assertIn("DETERMINISTIC_FALLBACK", results["metadata"]["warning"])

        md = evaluator.generate_markdown_summary(results)
        self.assertIn("Deterministic Heuristic Fallback", md)
        self.assertIn("DETERMINISTIC FALLBACK EVALUATION NOTICE", md)

    def test_sidecar_validation(self):
        evaluator = LeafGateEvaluator(gate_model_path="non_existent_gate_model.keras")
        sidecar_info = evaluator.sidecar_validation
        self.assertIn("classes", sidecar_info)
        self.assertEqual(sidecar_info["output_dim"], 3)


class TestFailClosedManifestValidation(unittest.TestCase):
    """Phase 1: Fail-closed manifest validation and error propagation."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reject_unknown_manifest_fields_csv(self):
        p = self.temp_dir / "manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "file_path",
                    "crop",
                    "ground_truth_class",
                    "collection_source",
                    "evaluation_task",
                    "unknown_typo_column",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "FLD_001",
                "file_path": "images/test.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "evaluation_task": "disease",
                "unknown_typo_column": "bad_data",
            })
        records, errors = load_and_validate_manifest(p)
        self.assertEqual(len(records), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Row 2", errors[0])
        self.assertIn("unknown_typo_column", errors[0])

    def test_reject_unknown_manifest_fields_json(self):
        p = self.temp_dir / "manifest.json"
        data = {
            "records": [
                {
                    "image_id": "FLD_002",
                    "file_path": "images/test.jpg",
                    "crop": "Tomato",
                    "ground_truth_class": "Tomato___healthy",
                    "collection_source": "pilot",
                    "evaluation_task": "disease",
                    "unexpected_meta": 1234,
                }
            ]
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        records, errors = load_and_validate_manifest(p)
        self.assertEqual(len(records), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Row 1", errors[0])
        self.assertIn("unexpected_meta", errors[0])

    def test_explicit_evaluation_task_requirements(self):
        # Unsupported crop (Apple) without explicit task='gate' must fail
        rec_data = {
            "image_id": "OOD_001",
            "file_path": "images/apple.jpg",
            "crop": "Apple",
            "ground_truth_class": "Apple___scab",
            "collection_source": "field",
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            FieldSampleRecord.from_dict(rec_data, row_idx=5)
        self.assertIn("Row 5", str(ctx.exception))
        self.assertIn("Missing required field 'evaluation_task'", str(ctx.exception))

        # Explicit task='gate' for unsupported crop succeeds
        rec_data["evaluation_task"] = "gate"
        rec = FieldSampleRecord.from_dict(rec_data, row_idx=5)
        self.assertEqual(rec.evaluation_task, "gate")

        # Invalid task string fails
        rec_data["evaluation_task"] = "unsupported_task"
        with self.assertRaises(ManifestValidationError) as ctx:
            FieldSampleRecord.from_dict(rec_data, row_idx=5)
        self.assertIn("Invalid evaluation_task", str(ctx.exception))

    def test_propagate_manifest_errors_fails_formal_evaluation(self):
        # Create manifest with 1 valid and 1 invalid row
        p = self.temp_dir / "manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "file_path",
                    "crop",
                    "ground_truth_class",
                    "collection_source",
                    "evaluation_task",
                    "typo_field",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "FLD_001",
                "file_path": "images/img1.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "evaluation_task": "disease",
                "typo_field": "invalid",
            })

        out_dir = self.temp_dir / "out"
        import argparse
        args = argparse.Namespace(
            manifest=p,
            dataset_root=self.temp_dir,
            output_dir=out_dir,
            model_path=Path("non_existent_model.keras"),
            gate_path=Path("non_existent_gate.keras"),
            split=None,
            audit_only=False,
            eval_gate=False,
            exclude_unverified=True,
            baseline_comparison=None,
            allow_exploratory_subset=False,
        )
        exit_code = run_field_validation(args)
        # Must fail closed with exit code 1
        self.assertEqual(exit_code, 1)

        # When allow_exploratory_subset=True, it doesn't fail closed at manifest check
        args.allow_exploratory_subset = True
        args.audit_only = True
        exit_code_exploratory = run_field_validation(args)
        self.assertEqual(exit_code_exploratory, 0)


class TestSafeImagePathResolution(unittest.TestCase):
    """Phase 2: Single authoritative safe image path resolution tests."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.img_dir = self.temp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.valid_img = self.img_dir / "leaf.jpg"
        self.valid_img.write_bytes(b"mock_jpeg_bytes")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_valid_nested_file_resolved(self):
        resolved = resolve_safe_image_path("images/leaf.jpg", dataset_root=self.temp_dir)
        self.assertEqual(resolved, self.valid_img)
        self.assertTrue(resolved.is_file())

    def test_parent_directory_traversal_rejected(self):
        with self.assertRaises(PathSecurityError) as ctx:
            resolve_safe_image_path("images/../../secret.txt", dataset_root=self.temp_dir)
        self.assertIn("traversal detected", str(ctx.exception).lower())

    def test_absolute_path_outside_root_rejected(self):
        outside_file = Path(tempfile.gettempdir()).resolve() / "outside.jpg"
        outside_file.write_bytes(b"data")
        try:
            with self.assertRaises(PathSecurityError) as ctx:
                resolve_safe_image_path(str(outside_file), dataset_root=self.temp_dir, allow_absolute=False)
            self.assertIn("absolute path", str(ctx.exception).lower())

            # Even with allow_absolute=True, escaping dataset root is strictly rejected
            with self.assertRaises(PathSecurityError) as ctx:
                resolve_safe_image_path(str(outside_file), dataset_root=self.temp_dir, allow_absolute=True)
            self.assertIn("containment violation", str(ctx.exception).lower())
        finally:
            if outside_file.exists():
                outside_file.unlink()

    def test_missing_file_rejected(self):
        with self.assertRaises(FileNotFoundError):
            resolve_safe_image_path("images/non_existent.jpg", dataset_root=self.temp_dir)

    def test_directory_path_rejected(self):
        with self.assertRaises(InvalidImageFileError) as ctx:
            resolve_safe_image_path("images", dataset_root=self.temp_dir)
        self.assertIn("directory", str(ctx.exception).lower())

    def test_symlink_escape_rejected(self):
        outside_target = self.temp_dir.parent / f"outside_target_{self.temp_dir.name}.jpg"
        outside_target.write_bytes(b"outside")
        symlink_path = self.img_dir / "symlink_escape.jpg"
        try:
            try:
                symlink_path.symlink_to(outside_target)
            except OSError:
                # Symlink creation may require admin privileges on Windows; skip if OS restricts
                self.skipTest("Symlink creation not permitted by OS without elevation.")

            with self.assertRaises(PathSecurityError) as ctx:
                resolve_safe_image_path("images/symlink_escape.jpg", dataset_root=self.temp_dir)
            self.assertIn("containment violation", str(ctx.exception).lower())
        finally:
            if symlink_path.is_symlink():
                symlink_path.unlink()
            if outside_target.exists():
                outside_target.unlink()


class TestGateModelAndSidecarIntegrity(unittest.TestCase):
    """Phase 3: Gate model and sidecar integrity validation."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_corrupt_sidecar_raises_gate_integrity_error(self):
        model_file = self.temp_dir / "mock_gate.keras"
        model_file.write_bytes(b"fake_keras_binary")
        sidecar_file = self.temp_dir / "gate_model_config.json"
        sidecar_file.write_text("{corrupt_json: invalid", encoding="utf-8")

        class MockLoadedService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            _model_path = model_file

        with self.assertRaises(GateIntegrityError) as ctx:
            LeafGateEvaluator(leaf_gate_service=MockLoadedService())
        self.assertIn("Failed to parse sidecar", str(ctx.exception))

    def test_sidecar_missing_required_fields_raises_gate_integrity_error(self):
        model_file = self.temp_dir / "mock_gate.keras"
        model_file.write_bytes(b"fake_keras_binary")
        sidecar_file = self.temp_dir / "gate_model_config.json"
        # Missing 'feature_backbone' and 'input'
        sidecar_file.write_text(json.dumps({"classes": ["non_leaf", "other_plant", "supported_leaf"]}), encoding="utf-8")

        class MockLoadedService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            _model_path = model_file

        with self.assertRaises(GateIntegrityError) as ctx:
            LeafGateEvaluator(leaf_gate_service=MockLoadedService())
        self.assertIn("missing required fields", str(ctx.exception))

    def test_sidecar_wrong_class_order_raises_gate_integrity_error(self):
        model_file = self.temp_dir / "mock_gate.keras"
        model_file.write_bytes(b"fake_keras_binary")
        sidecar_file = self.temp_dir / "gate_model_config.json"
        # Wrong order: supported_leaf first
        sidecar_file.write_text(
            json.dumps({
                "classes": ["supported_leaf", "other_plant", "non_leaf"],
                "feature_backbone": "MobileNetV2",
                "input": "224x224 RGB in [0, 255]",
            }),
            encoding="utf-8",
        )

        class MockLoadedService:
            is_loaded = True
            classes = ["supported_leaf", "other_plant", "non_leaf"]
            _model_path = model_file

        with self.assertRaises(GateIntegrityError) as ctx:
            LeafGateEvaluator(leaf_gate_service=MockLoadedService())
        self.assertIn("exact ordering required", str(ctx.exception))

    def test_missing_sidecar_when_model_exists_raises_gate_integrity_error(self):
        model_file = self.temp_dir / "mock_gate.keras"
        model_file.write_bytes(b"fake_keras_binary")
        # No gate_model_config.json created

        class MockLoadedService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            _model_path = model_file

        with self.assertRaises(GateIntegrityError) as ctx:
            LeafGateEvaluator(leaf_gate_service=MockLoadedService())
        self.assertIn("sidecar config is missing", str(ctx.exception))


class TestControlledBaselineValidation(unittest.TestCase):
    """Phase 4: Controlled baseline validation and rejection of validation metrics."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reject_validation_split_json_report(self):
        report_file = self.temp_dir / "val_baseline.json"
        data = {
            "split_evaluated": "validation",
            "metrics": {
                "test_accuracy": 0.95,
                "test_macro_f1": 0.94,
            },
        }
        report_file.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("held-out 'test' split required", str(ctx.exception))

    def test_reject_validation_only_metrics(self):
        report_file = self.temp_dir / "val_metrics.json"
        data = {
            "metrics": {
                "val_accuracy": 0.975,
                "val_macro_f1": 0.968,
            },
        }
        report_file.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("only validation metrics", str(ctx.exception))

    def test_accept_genuine_held_out_test_split(self):
        report_file = self.temp_dir / "genuine_test.json"
        data = {
            "split_evaluated": "test",
            "model_checkpoint": "best_model.keras",
            "timestamp_utc": "2026-10-20T12:00:00Z",
            "metrics": {
                "test_accuracy": 0.9689,
                "test_macro_f1": 0.9652,
                "test_top_3_accuracy": 0.995,
            },
        }
        report_file.write_text(json.dumps(data), encoding="utf-8")
        res = parse_controlled_baseline(report_file)
        self.assertEqual(res["accuracy"], 0.9689)
        self.assertEqual(res["macro_f1"], 0.9652)
        self.assertEqual(res["top_3_accuracy"], 0.995)

    def test_reject_out_of_range_metrics(self):
        report_file = self.temp_dir / "bad_range.json"
        data = {
            "split_evaluated": "test",
            "metrics": {
                "test_accuracy": 1.25,
                "test_macro_f1": 0.95,
            },
        }
        report_file.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("outside valid range [0.0, 1.0]", str(ctx.exception))


class TestDiseaseModelOutputValidation(unittest.TestCase):
    """Phase 5: Disease model output mathematical and structural validation."""

    def test_valid_model_output(self):
        probs = np.full(15, 1.0 / 15.0, dtype=np.float32)
        validated = validate_model_output(probs, expected_classes=15)
        self.assertEqual(validated.shape, (15,))
        self.assertAlmostEqual(float(np.sum(validated)), 1.0, places=4)

    def test_reject_nan_output(self):
        probs = np.full(15, 1.0 / 15.0)
        probs[3] = float("nan")
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_reject_infinite_output(self):
        probs = np.full(15, 1.0 / 15.0)
        probs[0] = float("inf")
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_reject_negative_probability(self):
        probs = np.full(15, 1.0 / 14.0)
        probs[0] = -0.1
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("out of bounds", str(ctx.exception))

    def test_reject_invalid_sum(self):
        probs = np.full(15, 0.02)  # sums to 0.30
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("must sum to 1.0", str(ctx.exception))

    def test_reject_wrong_dimension(self):
        probs = np.full(14, 1.0 / 14.0)  # 14 instead of 15
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("invalid", str(ctx.exception).lower())


class TestMetricStatisticalReliability(unittest.TestCase):
    """Phase 6: Metric and statistical reliability, macro F1, and cluster bootstrap."""

    def test_report_both_fixed_and_observed_macro_f1(self):
        # 10 samples only covering Tomato___healthy and Tomato___Early_blight
        predictions = [
            {
                "image_id": f"IMG_{i}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy" if i < 5 else "Tomato___Early_blight",
                "predicted_class": "Tomato___healthy" if i < 5 else "Tomato___Early_blight",
                "is_correct": True,
                "top_3_predictions": [{"class_name": "Tomato___healthy"}],
                "verification_status": "expert_verified",
            }
            for i in range(10)
        ]
        summary = calculate_field_metrics(predictions)
        # Observed classes (2 classes): F1 = 1.0
        self.assertEqual(summary.observed_macro_f1, 1.0)
        # Fixed classes (15 classes, 13 have 0 F1): 2/15 = 0.1333
        self.assertAlmostEqual(summary.fixed_macro_f1, 2.0 / 15.0, places=3)
        self.assertEqual(summary.observed_classes_count, 2)
        self.assertEqual(len(summary.unobserved_classes), 13)

    def test_cluster_bootstrap_reports_unavailable_when_clusters_under_5(self):
        # 4 unique clusters (under minimum 5)
        predictions = [
            {
                "image_id": f"IMG_{i}",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy",
                "is_correct": True,
                "top_3_predictions": [{"class_name": "Tomato___healthy"}],
                "duplicate_group_id": f"GROUP_{i % 4}",
                "verification_status": "expert_verified",
            }
            for i in range(12)
        ]
        summary = calculate_field_metrics(predictions)
        self.assertFalse(summary.macro_f1_ci_available)
        self.assertIsNone(summary.macro_f1_ci_95)
        self.assertIn("Insufficient unique clusters", summary.macro_f1_ci_reason)

    def test_group_majority_voting_tie_breaking(self):
        # Group with a tie: 1 vote for Class A (confidence 0.60), 1 vote for Class B (confidence 0.80)
        predictions = [
            {
                "image_id": "1",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___healthy",
                "raw_scores": {"Tomato___healthy": 0.60},
                "duplicate_group_id": "GRP_TIE",
                "verification_status": "expert_verified",
            },
            {
                "image_id": "2",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "predicted_class": "Tomato___Early_blight",
                "raw_scores": {"Tomato___Early_blight": 0.80},
                "duplicate_group_id": "GRP_TIE",
                "verification_status": "expert_verified",
            },
        ]
        summary = calculate_field_metrics(predictions)
        # Tie broken by higher confidence -> Early blight chosen -> Group is incorrect (ground truth is healthy)
        self.assertEqual(summary.cluster_metrics["cluster_accuracy"], 0.0)
        self.assertEqual(summary.cluster_metrics["aggregation_method"], "plurality_with_confidence_tie_breaking")


class TestGateVerificationPolicy(unittest.TestCase):
    """Phase 7: Gate verification policy and zero-denominator safety."""

    def test_zero_denominator_far_frr_returns_none(self):
        class MockService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            def predict_proba(self, frame):
                return {"non_leaf": 0.1, "other_plant": 0.1, "supported_leaf": 0.8}

        evaluator = LeafGateEvaluator(leaf_gate_service=MockService())
        # Empty dataset evaluation
        results = evaluator.evaluate_dataset([])
        gm = results["gate_metrics"]
        # Rates with 0 support MUST be None, never 0.0
        self.assertIsNone(gm["supported_leaves"]["false_rejection_rate"])
        self.assertIsNone(gm["other_plant_species"]["false_acceptance_rate"])
        self.assertIsNone(gm["non_leaf_objects"]["false_acceptance_rate"])
        self.assertIsNone(gm["overall_ood_summary"]["combined_false_acceptance_rate"])

        # Markdown summary must render N/A without crashing
        md = evaluator.generate_markdown_summary(results)
        self.assertIn("N/A (support=0)", md)


class TestExploratoryModeIsolation(unittest.TestCase):
    """P0 Regression Tests: Exploratory mode isolation, path containment, status tracking, and baseline rejection."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.img_dir = self.temp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = self.temp_dir / "out"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_exploratory_mode_cannot_bypass_path_traversal(self):
        # Manifest has a path traversal attack
        p = self.temp_dir / "manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "file_path",
                    "crop",
                    "ground_truth_class",
                    "collection_source",
                    "evaluation_task",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "FLD_EVIL",
                "file_path": "../evil.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "evaluation_task": "disease",
            })

        import argparse
        args = argparse.Namespace(
            manifest=p,
            dataset_root=self.temp_dir,
            output_dir=self.out_dir,
            model_path=Path("ml/models/experiment_weighted/best_model.keras"),
            gate_path=Path("ml/models/input_gate/leaf_gate.keras"),
            split=None,
            audit_only=True,
            eval_gate=False,
            exclude_unverified=True,
            baseline_comparison=None,
            allow_exploratory_subset=True,  # Even with this flag, traversal MUST abort!
        )
        exit_code = run_field_validation(args)
        self.assertEqual(exit_code, 1)

    def test_exploratory_mode_labels_outputs_explicitly(self):
        # Create 1 valid non-uniform image and a manifest
        img_p = self.img_dir / "valid_leaf.jpg"
        np_img = np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(np_img).save(img_p, format="JPEG")

        p = self.temp_dir / "manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "file_path",
                    "crop",
                    "ground_truth_class",
                    "collection_source",
                    "evaluation_task",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "FLD_001",
                "file_path": "images/valid_leaf.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "evaluation_task": "disease",
            })

        class MockEvaluator:
            def __init__(self, model_path):
                self.model_path = model_path
                self.model_sha256 = "dummy_sha"
            def predict_single(self, frame):
                from ml.evaluation.field_validation.evaluator import TopPrediction
                return "Tomato___healthy", 14, [TopPrediction(14, "Tomato___healthy", "Tomato", "Healthy", "Healthy", 1.0/15.0)], {"Tomato___healthy": 1.0/15.0}

        evaluator = FieldEvaluator.__new__(FieldEvaluator)
        evaluator.model_path = Path("mock_model.keras")
        evaluator.model_sha256 = "dummy_hash"
        evaluator.predict_single = MockEvaluator(evaluator.model_path).predict_single

        results = evaluator.evaluate_manifest(
            manifest_path=p,
            dataset_root=self.temp_dir,
            allow_exploratory_subset=True,
        )

        self.assertEqual(results["metadata"]["evaluation_status"], "EXPLORATORY_SUBSET")
        self.assertFalse(results["metadata"]["is_formal_evaluation"])
        self.assertTrue(results["metadata"]["allow_exploratory_subset"])

        md = evaluator.generate_markdown_summary(results)
        self.assertIn("EXPLORATORY — NOT FOR FORMAL PERFORMANCE CLAIMS", md)
        self.assertIn("EXPLORATORY EVALUATION ACTIVE", md)

    def test_exploratory_record_accounting_and_exclusion_reasons(self):
        # 1 valid row, 1 schema-rejected row
        img_p = self.img_dir / "leaf.jpg"
        Image.new("RGB", (224, 224), color=(60, 140, 50)).save(img_p, format="JPEG")

        p = self.temp_dir / "manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "file_path",
                    "crop",
                    "ground_truth_class",
                    "collection_source",
                    "evaluation_task",
                    "invalid_column",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "FLD_001",
                "file_path": "images/leaf.jpg",
                "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy",
                "collection_source": "pilot",
                "evaluation_task": "disease",
                "invalid_column": "typo",
            })

        evaluator = FieldEvaluator.__new__(FieldEvaluator)
        evaluator.model_path = Path("mock_model.keras")
        evaluator.model_sha256 = "dummy_hash"
        def dummy_predict(frame):
            from ml.evaluation.field_validation.evaluator import TopPrediction
            return "Tomato___healthy", 14, [TopPrediction(14, "Tomato___healthy", "Tomato", "Healthy", "Healthy", 1.0)], {"Tomato___healthy": 1.0}
        evaluator.predict_single = dummy_predict

        results = evaluator.evaluate_manifest(
            manifest_path=p,
            dataset_root=self.temp_dir,
            allow_exploratory_subset=True,
        )

        meta = results["metadata"]
        self.assertEqual(meta["submitted_records"], 1)
        self.assertEqual(meta["valid_records"], 0)
        self.assertEqual(meta["rejected_records"], 1)
        self.assertEqual(meta["evaluated_records"], 0)
        self.assertIn("exclusion_reasons", meta)
        self.assertEqual(meta["exclusion_reasons"]["schema_validation_rejected_count"], 1)

    def test_baseline_comparison_rejects_exploratory_results(self):
        dummy_field_results = {
            "metadata": {
                "evaluation_status": "EXPLORATORY_SUBSET",
                "is_formal_evaluation": False,
                "evaluation_timestamp": "2026-11-20T12:00:00Z",
            },
            "metrics": {
                "accuracy": 0.85,
                "macro_f1": 0.82,
            },
        }
        baseline_file = self.temp_dir / "valid_baseline.json"
        baseline_file.write_text(
            json.dumps({
                "split_evaluated": "test",
                "model_checkpoint": "best_model.keras",
                "metrics": {"test_accuracy": 0.95, "test_macro_f1": 0.94},
            }),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            generate_baseline_comparison(baseline_file, dummy_field_results)
        self.assertIn("cannot be performed on exploratory evaluation results", str(ctx.exception))

    def test_exploratory_mode_cannot_bypass_cross_split_leakage(self):
        img1 = self.img_dir / "leaf1.jpg"
        img2 = self.img_dir / "leaf2.jpg"
        Image.fromarray(np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)).save(img1, format="JPEG")
        Image.fromarray(np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)).save(img2, format="JPEG")

        p = self.temp_dir / "manifest_leak.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id", "file_path", "crop", "ground_truth_class",
                    "collection_source", "split", "duplicate_group_id", "evaluation_task"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "L1", "file_path": "images/leaf1.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_dev", "duplicate_group_id": "LEAK_GRP_1", "evaluation_task": "disease"
            })
            writer.writerow({
                "image_id": "L2", "file_path": "images/leaf2.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_test", "duplicate_group_id": "LEAK_GRP_1", "evaluation_task": "disease"
            })

        import argparse
        args_formal = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=False,
        )
        self.assertEqual(run_field_validation(args_formal), 1)

        args_exploratory = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=True,
        )
        self.assertEqual(run_field_validation(args_exploratory), 1)

        # Confirm no evaluation JSON artifact created
        json_artifacts = list(self.out_dir.glob("field_disease_eval_*.json"))
        self.assertEqual(len(json_artifacts), 0)

    def test_exploratory_mode_cannot_bypass_controlled_dataset_leakage(self):
        img1 = self.img_dir / "leaf_ctrl.jpg"
        np_img = np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(np_img).save(img1, format="JPEG")
        img_sha = hashlib.sha256(img1.read_bytes()).hexdigest()

        # Write controlled manifest
        ctrl_p = self.temp_dir / "controlled_manifest.csv"
        with ctrl_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "sha256"])
            w.writeheader()
            w.writerow({"image_id": "CTRL_001", "sha256": img_sha})

        p = self.temp_dir / "manifest_ctrl_leak.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id", "file_path", "crop", "ground_truth_class",
                    "collection_source", "split", "evaluation_task"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "C_LEAK", "file_path": "images/leaf_ctrl.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_test", "evaluation_task": "disease"
            })

        auditor = DataQualityAuditor(manifest_path=p, dataset_root=self.temp_dir, controlled_manifest_path=ctrl_p)
        rep = auditor.audit_all()
        self.assertEqual(len(rep.controlled_dataset_leaks), 1)

        import argparse
        args_formal = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=False,
            controlled_manifest=ctrl_p,
        )
        self.assertEqual(run_field_validation(args_formal), 1)

        args_exploratory = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=True,
            controlled_manifest=ctrl_p,
        )
        self.assertEqual(run_field_validation(args_exploratory), 1)

        json_artifacts = list(self.out_dir.glob("field_disease_eval_*.json"))
        self.assertEqual(len(json_artifacts), 0)

    def test_exploratory_mode_cannot_bypass_duplicate_image_id(self):
        img1 = self.img_dir / "leaf_dup.jpg"
        Image.fromarray(np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)).save(img1, format="JPEG")

        p = self.temp_dir / "manifest_dup_id.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id", "file_path", "crop", "ground_truth_class",
                    "collection_source", "split", "evaluation_task"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "DUP_ID_1", "file_path": "images/leaf_dup.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_test", "evaluation_task": "disease"
            })
            writer.writerow({
                "image_id": "DUP_ID_1", "file_path": "images/leaf_dup.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_test", "evaluation_task": "disease"
            })

        import argparse
        args_formal = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=False,
        )
        self.assertEqual(run_field_validation(args_formal), 1)

        args_exploratory = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=True,
        )
        self.assertEqual(run_field_validation(args_exploratory), 1)

        json_artifacts = list(self.out_dir.glob("field_disease_eval_*.json"))
        self.assertEqual(len(json_artifacts), 0)

    def test_exploratory_mode_cannot_bypass_duplicate_group_metadata_conflict(self):
        img1 = self.img_dir / "leaf_c1.jpg"
        img2 = self.img_dir / "leaf_c2.jpg"
        Image.fromarray(np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)).save(img1, format="JPEG")
        Image.fromarray(np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)).save(img2, format="JPEG")

        p = self.temp_dir / "manifest_grp_conflict.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id", "file_path", "crop", "ground_truth_class",
                    "collection_source", "split", "duplicate_group_id", "evaluation_task"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "image_id": "C1", "file_path": "images/leaf_c1.jpg", "crop": "Tomato",
                "ground_truth_class": "Tomato___healthy", "collection_source": "pilot",
                "split": "field_test", "duplicate_group_id": "GRP_CONFLICT", "evaluation_task": "disease"
            })
            writer.writerow({
                "image_id": "C2", "file_path": "images/leaf_c2.jpg", "crop": "Potato",
                "ground_truth_class": "Potato___healthy", "collection_source": "pilot",
                "split": "field_test", "duplicate_group_id": "GRP_CONFLICT", "evaluation_task": "disease"
            })

        import argparse
        args_formal = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=False,
        )
        self.assertEqual(run_field_validation(args_formal), 1)

        args_exploratory = argparse.Namespace(
            manifest=p, dataset_root=self.temp_dir, output_dir=self.out_dir,
            model_path=Path("mock.keras"), gate_path=Path("mock_gate.keras"),
            split=None, audit_only=True, eval_gate=False, exclude_unverified=True,
            baseline_comparison=None, allow_exploratory_subset=True,
        )
        self.assertEqual(run_field_validation(args_exploratory), 1)

        json_artifacts = list(self.out_dir.glob("field_disease_eval_*.json"))
        self.assertEqual(len(json_artifacts), 0)

    def test_wording_independent_security_enforcement(self):
        # Error text contains NEITHER 'traversal', 'containment', NOR '..'
        custom_sec_error = ManifestSchemaError(
            "Row 2: Restricted operational security boundary failure - perimeter breach detected.",
            category="security",
            row=2,
            field_name="file_path",
        )
        self.assertEqual(custom_sec_error.category, "security")
        self.assertNotIn("traversal", str(custom_sec_error).lower())
        self.assertNotIn("containment", str(custom_sec_error).lower())
        self.assertNotIn("..", str(custom_sec_error))

        # Test auditor properly marks path_traversal_errors via structured category
        auditor = DataQualityAuditor(dataset_root=self.temp_dir)
        report = auditor.audit(records=[], schema_errors=[custom_sec_error])
        self.assertGreaterEqual(len(report.path_traversal_errors), 1)
        self.assertFalse(report.is_clean)


class TestGateFailureAccounting(unittest.TestCase):
    """P0 Regression Tests: Gate evaluation failure accounting, path traversal re-raising, and denominator safety."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.img_dir = self.temp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_gate_evaluator_aborts_on_path_traversal(self):
        class MockGateService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            def predict_proba(self, frame):
                return {"non_leaf": 0.1, "other_plant": 0.1, "supported_leaf": 0.8}

        evaluator = LeafGateEvaluator(leaf_gate_service=MockGateService())
        samples = [
            GateSampleRecord(
                sample_id="G_EVIL",
                file_path="../outside.jpg",
                ground_truth_category="supported_leaf",
            )
        ]
        # Must re-raise PathSecurityError rather than catching into error ledger
        with self.assertRaises(PathSecurityError):
            evaluator.evaluate_dataset(samples, dataset_root=self.temp_dir)

    def test_gate_evaluator_failure_accounting_mixed_samples(self):
        valid_img = self.img_dir / "valid.jpg"
        np_img = np.random.randint(30, 200, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(np_img).save(valid_img, format="JPEG")
        corrupt_img = self.img_dir / "corrupt.jpg"
        corrupt_img.write_bytes(b"not an image file")

        class MockGateService:
            is_loaded = True
            classes = ["non_leaf", "other_plant", "supported_leaf"]
            def predict_proba(self, frame):
                return {"non_leaf": 0.05, "other_plant": 0.05, "supported_leaf": 0.90}

        evaluator = LeafGateEvaluator(leaf_gate_service=MockGateService())
        samples = [
            GateSampleRecord(
                sample_id="G_1",
                file_path="images/valid.jpg",
                ground_truth_category="supported_leaf",
            ),
            GateSampleRecord(
                sample_id="G_2",
                file_path="images/corrupt.jpg",
                ground_truth_category="supported_leaf",
            ),
        ]
        results = evaluator.evaluate_dataset(
            samples,
            dataset_root=self.temp_dir,
            allow_exploratory_subset=True,
        )

        supp = results["gate_metrics"]["supported_leaves"]
        self.assertEqual(supp["submitted"], 2)
        self.assertEqual(supp["support"], 1)
        self.assertEqual(supp["failed"], 1)
        self.assertEqual(supp["accepted"], 1)
        self.assertEqual(supp["acceptance_rate"], 1.0)  # conditional on successful: 1/1
        self.assertEqual(supp["failure_adjusted_acceptance_rate"], 0.5)  # failure-adjusted: 1/2
        self.assertEqual(results["metadata"]["gate_inference_accounting"]["failed"], 1)


class TestStrictControlledBaselineProvenance(unittest.TestCase):
    """P0 Regression Tests: Strict controlled baseline provenance checks."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_baseline_rejects_missing_split_metadata(self):
        # Report with generic accuracy and macro_f1 but no split_evaluated
        report_file = self.temp_dir / "generic_report.json"
        report_file.write_text(
            json.dumps({
                "model_checkpoint": "best_model.keras",
                "metrics": {"accuracy": 0.95, "macro_f1": 0.94},
            }),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("lacks explicit test split provenance", str(ctx.exception))

    def test_baseline_rejects_exploratory_report(self):
        report_file = self.temp_dir / "exploratory_report.json"
        report_file.write_text(
            json.dumps({
                "split_evaluated": "test",
                "model_checkpoint": "best_model.keras",
                "evaluation_status": "EXPLORATORY_SUBSET",
                "metrics": {"test_accuracy": 0.95, "test_macro_f1": 0.94},
            }),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("exploratory results are strictly prohibited", str(ctx.exception).lower())

    def test_baseline_rejects_missing_model_checkpoint(self):
        report_file = self.temp_dir / "no_model.json"
        report_file.write_text(
            json.dumps({
                "split_evaluated": "test",
                "metrics": {"test_accuracy": 0.95, "test_macro_f1": 0.94},
            }),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("lacks explicit model/checkpoint identity", str(ctx.exception))

    def test_baseline_markdown_rejects_ambiguous_report(self):
        report_file = self.temp_dir / "ambiguous.md"
        report_file.write_text(
            "# General Model Report\n| **Accuracy** | 0.95 |\n| **Macro-F1** | 0.94 |\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            parse_controlled_baseline(report_file)
        self.assertIn("lacks explicit test split provenance", str(ctx.exception))


class TestProbabilityContractBoundaries(unittest.TestCase):
    """P1 Regression Tests: Softmax probability contract and tolerance boundaries."""

    def test_probability_contract_rejects_logits(self):
        # Raw logits with large positive values summing to >> 1.0
        logits = np.array([2.5, 1.2, 0.8, -0.4, 3.1, 0.0, 1.1, 0.5, -1.2, 0.2, 0.9, 1.5, 0.3, -0.1, 2.0])
        with self.assertRaises(ModelOutputValidationError):
            validate_model_output(logits, expected_classes=15)

    def test_probability_contract_rejects_negative_value(self):
        # Probabilities summing to ~1.0 but containing a negative value
        probs = np.full(15, 1.0 / 14.0)
        probs[0] = -0.05
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs, expected_classes=15)
        self.assertIn("negative probabilities", str(ctx.exception))

    def test_probability_contract_boundary_sums(self):
        # Sum = 1.008 is within 0.01 tolerance -> passes
        probs_pass_high = np.full(15, 1.008 / 15.0)
        validated_high = validate_model_output(probs_pass_high, expected_classes=15)
        self.assertEqual(len(validated_high), 15)

        # Sum = 0.992 is within 0.01 tolerance -> passes
        probs_pass_low = np.full(15, 0.992 / 15.0)
        validated_low = validate_model_output(probs_pass_low, expected_classes=15)
        self.assertEqual(len(validated_low), 15)

        # Sum = 1.02 exceeds 0.01 tolerance -> raises
        probs_fail_high = np.full(15, 1.02 / 15.0)
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs_fail_high, expected_classes=15)
        self.assertIn("must sum to 1.0 within tolerance", str(ctx.exception))

        # Sum = 0.98 exceeds 0.01 tolerance -> raises
        probs_fail_low = np.full(15, 0.98 / 15.0)
        with self.assertRaises(ModelOutputValidationError) as ctx:
            validate_model_output(probs_fail_low, expected_classes=15)
        self.assertIn("must sum to 1.0 within tolerance", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
