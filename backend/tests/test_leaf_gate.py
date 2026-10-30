"""
Tests for the trained leaf-acceptance gate: service behaviour, decision mapping and the
POST /api/predict contract.

The gate's *accuracy* is not asserted here - that is measured by
``ml/scripts/evaluate_input_gate.py`` on the held-out corpus and reported in
``docs/audit_reports/non_leaf_input_safety_report.md``. These tests pin the decision logic,
the fallback behaviour and the API safety contract, which are what regressions break.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services import input_gate
from backend.app.services.inference import model_service, preprocess_image
from backend.app.services.leaf_gate import LeafGateService, leaf_gate_service
from backend.tests.non_leaf_fixtures import (
    create_dark_leaf_bytes,
    create_leaf_closeup_bytes,
    create_portrait_bytes,
)

DIAGNOSIS_FIELDS = (
    "predicted_class_id",
    "raw_class",
    "crop",
    "condition",
    "health",
    "score",
    "top_predictions",
    "all_scores",
)


class StubGateModel:
    """Minimal stand-in for the exported Keras gate."""

    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32).reshape(-1)

    def predict(self, batch, verbose=0):
        return np.stack([self.output] * len(batch))


class CountingModel:
    """Disease-model stand-in that records whether inference was invoked."""

    def __init__(self, class_id: int = 7, score: float = 0.93):
        self.class_id = class_id
        self.score = score
        self.calls = 0

    def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.calls += 1
        scores = np.full(15, (1.0 - self.score) / 14.0, dtype=np.float32)
        scores[self.class_id] = self.score
        return np.expand_dims(scores, axis=0)


def frame_of(payload: bytes) -> np.ndarray:
    return preprocess_image(payload)[0]


def probs(non_leaf: float, other_plant: float, supported_leaf: float) -> dict:
    return {
        "non_leaf": non_leaf,
        "other_plant": other_plant,
        "supported_leaf": supported_leaf,
    }


class TestGateService(unittest.TestCase):
    """The service must fail closed: no artefact means no gate verdict, not a free pass."""

    def test_unloaded_service_returns_none(self):
        service = LeafGateService()
        self.assertIsNone(service.predict_proba(np.zeros((224, 224, 3), dtype=np.float32)))

    def test_loaded_service_maps_output_to_class_names(self):
        service = LeafGateService()
        service.set_model(StubGateModel([0.10, 0.15, 0.75]),
                          classes=["non_leaf", "other_plant", "supported_leaf"])
        result = service.predict_proba(frame_of(create_leaf_closeup_bytes()))
        self.assertEqual(set(result), {"non_leaf", "other_plant", "supported_leaf"})
        self.assertAlmostEqual(result["supported_leaf"], 0.75, places=5)

    def test_wrong_output_dimension_raises(self):
        service = LeafGateService()
        service.set_model(StubGateModel([0.5, 0.5]))
        with self.assertRaises(ValueError):
            service.predict_proba(frame_of(create_leaf_closeup_bytes()))

    def test_wrong_frame_shape_raises(self):
        service = LeafGateService()
        service.set_model(StubGateModel([0.1, 0.1, 0.8]))
        with self.assertRaises(ValueError):
            service.predict_proba(np.zeros((10, 10, 3), dtype=np.float32))

    def test_classes_are_read_from_sidecar_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "gate_model_config.json").write_text(
                json.dumps({"classes": ["supported_leaf", "non_leaf", "other_plant"]}),
                encoding="utf-8",
            )
            classes = LeafGateService._read_classes(directory / "leaf_gate.keras")
        self.assertEqual(classes, ["supported_leaf", "non_leaf", "other_plant"])

    def test_missing_sidecar_falls_back_to_default_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            classes = LeafGateService._read_classes(Path(tmp) / "leaf_gate.keras")
        self.assertEqual(classes, ["non_leaf", "other_plant", "supported_leaf"])


class TestGateDecisions(unittest.TestCase):
    """Decision mapping from gate probabilities to an InputAssessment."""

    def assess(self, payload: bytes, probabilities) -> input_gate.InputAssessment:
        return input_gate.assess_image(frame_of(payload), probabilities)

    def test_confident_non_leaf_is_rejected(self):
        result = self.assess(create_portrait_bytes(), probs(0.97, 0.02, 0.01))
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)
        self.assertEqual(result.category, "non_leaf_image")

    def test_confident_other_plant_is_rejected_as_unsupported_species(self):
        result = self.assess(create_leaf_closeup_bytes(), probs(0.02, 0.91, 0.07))
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_UNSUPPORTED_SPECIES)
        self.assertEqual(result.category, "unsupported_species")

    def test_confident_supported_leaf_is_accepted(self):
        result = self.assess(create_leaf_closeup_bytes(), probs(0.01, 0.04, 0.95))
        self.assertEqual(result.decision, input_gate.ACCEPTED)
        self.assertEqual(result.code, input_gate.CODE_ACCEPTED)

    def test_low_confidence_abstains_instead_of_deciding(self):
        result = self.assess(create_leaf_closeup_bytes(), probs(0.38, 0.32, 0.30))
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_AMBIGUOUS)
        self.assertEqual(result.category, "ambiguous_image")

    def test_low_confidence_with_supported_top_class_still_abstains(self):
        # A bare plurality is not evidence; the gate must not turn it into a diagnosis.
        result = self.assess(create_leaf_closeup_bytes(), probs(0.20, 0.20, 0.45))
        self.assertEqual(result.decision, input_gate.ABSTAINED)

    def test_signals_expose_probabilities_but_never_a_class(self):
        result = self.assess(create_portrait_bytes(), probs(0.97, 0.02, 0.01))
        self.assertEqual(result.signals["gate_label"], "non_leaf")
        self.assertAlmostEqual(result.signals["gate_probabilities"]["non_leaf"], 0.97, places=4)
        self.assertIn("gate_threshold", result.signals)
        for field in DIAGNOSIS_FIELDS:
            self.assertNotIn(field, result.signals)

    def test_malformed_gate_payload_falls_back_to_deterministic_screen(self):
        for payload in (None, {}, {"non_leaf": 0.5}, {"a": 1, "b": 2, "c": 3},
                        {"non_leaf": "x", "other_plant": 0.1, "supported_leaf": 0.1}):
            with self.subTest(payload=payload):
                result = self.assess(create_portrait_bytes(), payload)
                self.assertEqual(result.decision, input_gate.REJECTED)
                self.assertNotIn("gate_label", result.signals)

    def test_quality_abstention_precedes_the_gate(self):
        result = self.assess(create_dark_leaf_bytes(), probs(0.0, 0.0, 1.0))
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_UNDEREXPOSED)


class TestPredictEndpointWithGate(unittest.TestCase):
    """The endpoint must never diagnose an input the gate declines, and never lose the gate."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def setUp(self):
        self.model = CountingModel()
        model_service.set_model(self.model)

    def tearDown(self):
        model_service.unload_model()

    def post(self, name: str, payload: bytes):
        return self.client.post("/api/predict", files={"file": (name, payload, "image/png")})

    def assert_rejection(self, response, status: str, code: str, category: str):
        self.assertEqual(response.status_code, 422)
        body = response.json()
        detail = body["detail"]
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail["status"], status)
        self.assertEqual(detail["code"], code)
        self.assertEqual(detail["category"], category)
        self.assertTrue(detail["message"].startswith("We couldn't verify"))
        for field in DIAGNOSIS_FIELDS:
            self.assertNotIn(field, body)
            self.assertNotIn(field, detail)
        return detail

    def test_non_leaf_gate_decision_returns_structured_422(self):
        with patch.object(leaf_gate_service, "predict_proba", return_value=probs(0.96, 0.03, 0.01)):
            response = self.post("portrait.png", create_portrait_bytes())
        self.assert_rejection(response, "rejected", "NON_LEAF_IMAGE", "non_leaf_image")
        self.assertEqual(self.model.calls, 0)

    def test_unsupported_species_returns_its_own_rejection(self):
        with patch.object(leaf_gate_service, "predict_proba", return_value=probs(0.02, 0.93, 0.05)):
            response = self.post("apple.png", create_leaf_closeup_bytes())
        detail = self.assert_rejection(response, "rejected", "UNSUPPORTED_SPECIES", "unsupported_species")
        self.assertIn("tomato, potato and bell pepper", detail["detail"])
        self.assertEqual(self.model.calls, 0)

    def test_ambiguous_gate_decision_abstains(self):
        with patch.object(leaf_gate_service, "predict_proba", return_value=probs(0.40, 0.31, 0.29)):
            response = self.post("scene.png", create_leaf_closeup_bytes())
        self.assert_rejection(response, "abstained", "AMBIGUOUS_IMAGE", "ambiguous_image")
        self.assertEqual(self.model.calls, 0)

    def test_supported_leaf_from_gate_still_predicts(self):
        with patch.object(leaf_gate_service, "predict_proba", return_value=probs(0.01, 0.02, 0.97)):
            response = self.post("leaf.png", create_leaf_closeup_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_class_id"], 7)
        self.assertEqual(self.model.calls, 1)

    def test_gate_unavailable_falls_back_to_deterministic_screen(self):
        with patch.object(leaf_gate_service, "predict_proba", return_value=None):
            rejected = self.post("portrait.png", create_portrait_bytes())
        self.assert_rejection(rejected, "rejected", "NON_LEAF_IMAGE", "non_leaf_image")
        self.assertEqual(self.model.calls, 0)

        with patch.object(leaf_gate_service, "predict_proba", return_value=None):
            accepted = self.post("leaf.png", create_leaf_closeup_bytes())
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(self.model.calls, 1)


if __name__ == "__main__":
    unittest.main()
