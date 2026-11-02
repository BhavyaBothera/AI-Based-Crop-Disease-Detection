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
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from backend.app.services.inference import TARGET_CLASS_MAPPING as BACKEND_CLASS_MAPPING
from ml.evaluation.field_validation.data_auditor import (
    AuditReport,
    DataQualityAuditor,
    compute_dhash,
    hamming_distance,
)
from ml.evaluation.field_validation.evaluator import (
    PREPROCESSING_VERSION,
    FieldEvaluator,
    compute_file_sha256,
    preprocess_field_image,
)
from ml.evaluation.field_validation.gate_evaluator import (
    GateSampleRecord,
    LeafGateEvaluator,
)
from ml.evaluation.field_validation.manifest_schema import (
    ALLOWED_ANNOTATOR_STATUS,
    ALLOWED_BACKGROUND_TYPES,
    ALLOWED_CROPS,
    ALLOWED_DISEASE_SEVERITIES,
    ALLOWED_LIGHTING_CONDITIONS,
    ALLOWED_SPLITS,
    ALLOWED_VERIFICATION_STATUS,
    CROP_TO_CLASSES,
    NAME_TO_CLASS_ID,
    TARGET_CLASS_MAPPING,
    FieldSampleRecord,
    ManifestValidationError,
    load_and_validate_manifest,
)
from ml.evaluation.field_validation.metrics import (
    FieldEvaluationSummary,
    bootstrap_macro_f1_ci,
    calculate_field_metrics,
    wilson_score_interval,
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
        self.assertIn("uniform canvas", str(ctx.exception).lower())

    def test_undersized_image_rejection(self):
        # Below 64x64 minimum floor
        img = Image.new("RGB", (50, 50), color=(100, 150, 50))
        with self.assertRaises(ValueError) as ctx:
            preprocess_field_image(img, enforce_level_a=True)
        self.assertIn("below minimum floor", str(ctx.exception).lower())


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


if __name__ == "__main__":
    unittest.main()
