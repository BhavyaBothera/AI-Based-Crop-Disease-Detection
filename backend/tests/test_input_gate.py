"""
Tests for the content-level input suitability gate and its API contract.

Coverage:
- Non-leaf probes never receive a diagnosis and never invoke the model.
- Ambiguous, sparse, scattered and unassessable probes abstain.
- Legitimate leaf photographs (single and multiple leaves) still predict normally.
- Rejections carry a machine-readable status/code/category and no diagnosis fields.
- Admission-control bookkeeping is unchanged by a rejection.
"""
import unittest

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services import input_gate
from backend.app.services.inference import (
    UniformCanvasError,
    model_service,
    preprocess_image,
)
from backend.tests.non_leaf_fixtures import (
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

# Response fields that constitute a diagnosis and must never accompany a rejection.
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


class CountingModel:
    """Mock model that records whether inference was invoked at all."""

    def __init__(self, class_id: int = 7, score: float = 0.93):
        self.class_id = class_id
        self.score = score
        self.calls = 0

    def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.calls += 1
        scores = np.full(15, (1.0 - self.score) / 14.0, dtype=np.float32)
        scores[self.class_id] = self.score
        return np.expand_dims(scores, axis=0)


def assess_bytes(payload: bytes) -> input_gate.InputAssessment:
    tensor = preprocess_image(payload)
    return input_gate.assess_image(tensor[0])


class TestInputGateDecisions(unittest.TestCase):
    """Unit tests for the deterministic gate on synthetic probes."""

    def test_portrait_is_rejected_as_non_leaf(self):
        result = assess_bytes(create_portrait_bytes())
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)
        self.assertEqual(result.category, "non_leaf_image")
        self.assertFalse(result.is_accepted)

    def test_animal_is_rejected_as_non_leaf(self):
        result = assess_bytes(create_animal_bytes())
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)

    def test_building_is_rejected_as_non_leaf(self):
        result = assess_bytes(create_building_bytes())
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)

    def test_document_is_rejected_as_non_leaf(self):
        result = assess_bytes(create_document_bytes())
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)

    def test_random_object_is_rejected_as_non_leaf(self):
        result = assess_bytes(create_object_bytes())
        self.assertEqual(result.decision, input_gate.REJECTED)
        self.assertEqual(result.code, input_gate.CODE_NON_LEAF)

    def test_screenshot_is_never_accepted(self):
        # A screenshot with a small green accent is colour-ambiguous; it must abstain
        # rather than be diagnosed. It is not claimed to be a confident non-leaf.
        result = assess_bytes(create_screenshot_bytes())
        self.assertNotEqual(result.decision, input_gate.ACCEPTED)
        self.assertIn(result.category, ("non_leaf_image", "ambiguous_image"))

    def test_uniform_canvas_is_rejected_before_the_gate(self):
        with self.assertRaises(UniformCanvasError):
            preprocess_image(create_uniform_bytes())

    def test_leaf_closeup_is_accepted(self):
        result = assess_bytes(create_leaf_closeup_bytes())
        self.assertEqual(result.decision, input_gate.ACCEPTED)
        self.assertEqual(result.code, input_gate.CODE_ACCEPTED)
        self.assertTrue(result.is_accepted)

    def test_multiple_leaves_are_accepted(self):
        result = assess_bytes(create_multiple_leaves_bytes())
        self.assertEqual(result.decision, input_gate.ACCEPTED)

    def test_complex_background_leaf_is_accepted(self):
        result = assess_bytes(create_complex_background_bytes())
        self.assertEqual(result.decision, input_gate.ACCEPTED)

    def test_sparse_vegetation_abstains(self):
        result = assess_bytes(create_sparse_vegetation_bytes())
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_SPARSE)
        self.assertEqual(result.category, "ambiguous_image")

    def test_tiny_leaf_abstains(self):
        result = assess_bytes(create_tiny_leaf_bytes())
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_SPARSE)

    def test_scattered_vegetation_abstains(self):
        result = assess_bytes(create_scattered_vegetation_bytes())
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_SCATTERED)

    def test_dark_leaf_abstains_as_low_quality(self):
        result = assess_bytes(create_dark_leaf_bytes())
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_UNDEREXPOSED)
        self.assertEqual(result.category, "low_quality_image")

    def test_blurry_leaf_abstains_as_low_detail(self):
        result = assess_bytes(create_blurry_leaf_bytes())
        self.assertEqual(result.decision, input_gate.ABSTAINED)
        self.assertEqual(result.code, input_gate.CODE_LOW_DETAIL)

    def test_signals_are_reported_and_bounded(self):
        result = assess_bytes(create_leaf_closeup_bytes())
        self.assertEqual(
            set(result.signals),
            {"plant_evidence", "dominant_region", "region_count", "mean_luminance", "detail_variance"},
        )
        self.assertGreaterEqual(result.signals["plant_evidence"], 0.0)
        self.assertLessEqual(result.signals["plant_evidence"], 1.0)
        self.assertGreaterEqual(result.signals["dominant_region"], 0.0)
        self.assertLessEqual(result.signals["dominant_region"], 1.0)

    def test_error_detail_contains_no_diagnosis_fields(self):
        detail = assess_bytes(create_portrait_bytes()).to_error_detail()
        for field in DIAGNOSIS_FIELDS:
            self.assertNotIn(field, detail)
        self.assertEqual(detail["status"], "rejected")
        self.assertEqual(detail["code"], input_gate.CODE_NON_LEAF)
        self.assertTrue(detail["retryable"])
        self.assertIn("supported crop leaf", detail["message"])

    def test_malformed_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            input_gate.assess_image(np.zeros((10, 10), dtype=np.float32))
        with self.assertRaises(ValueError):
            input_gate.assess_image(np.zeros((0, 10, 3), dtype=np.float32))

    def test_gate_can_be_disabled_for_rollback(self):
        original = input_gate.ENABLE_INPUT_GATE
        input_gate.ENABLE_INPUT_GATE = False
        try:
            result = assess_bytes(create_portrait_bytes())
        finally:
            input_gate.ENABLE_INPUT_GATE = original
        self.assertEqual(result.decision, input_gate.ACCEPTED)
        self.assertIn("disabled", f"{result.headline} {result.detail}".lower())


class TestPredictContentGateEndpoint(unittest.TestCase):
    """Endpoint-level safety tests for POST /api/predict."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def setUp(self):
        self.model = CountingModel()
        model_service.set_model(self.model)

    def tearDown(self):
        model_service.unload_model()

    def _post(self, name: str, payload: bytes):
        return self.client.post(
            "/api/predict",
            files={"file": (name, payload, "image/png")},
        )

    def _assert_no_diagnosis(self, response):
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIsInstance(body.get("detail"), dict)
        detail = body["detail"]
        for field in DIAGNOSIS_FIELDS:
            self.assertNotIn(field, body)
            self.assertNotIn(field, detail)
        self.assertIn(detail["status"], ("rejected", "abstained"))
        self.assertTrue(detail["message"].startswith("We couldn't verify"))
        return detail

    def test_non_leaf_categories_never_reach_the_model(self):
        probes = {
            "person.png": create_portrait_bytes(),
            "animal.png": create_animal_bytes(),
            "building.png": create_building_bytes(),
            "document.png": create_document_bytes(),
            "object.png": create_object_bytes(),
            "screenshot.png": create_screenshot_bytes(),
        }
        for name, payload in probes.items():
            with self.subTest(probe=name):
                response = self._post(name, payload)
                detail = self._assert_no_diagnosis(response)
                if name != "screenshot.png":
                    self.assertEqual(detail["code"], input_gate.CODE_NON_LEAF)
                self.assertEqual(self.model.calls, 0)

    def test_ambiguous_image_abstains_without_diagnosis(self):
        response = self._post("sparse.png", create_sparse_vegetation_bytes())
        detail = self._assert_no_diagnosis(response)
        self.assertEqual(detail["status"], "abstained")
        self.assertEqual(detail["category"], "ambiguous_image")
        self.assertEqual(self.model.calls, 0)

    def test_low_quality_image_abstains_without_diagnosis(self):
        response = self._post("dark.png", create_dark_leaf_bytes())
        detail = self._assert_no_diagnosis(response)
        self.assertEqual(detail["category"], "low_quality_image")
        self.assertEqual(self.model.calls, 0)

    def test_supported_leaf_still_predicts(self):
        response = self._post("leaf.png", create_leaf_closeup_bytes())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_class_id"], 7)
        self.assertEqual(data["condition"], "Late Blight")
        self.assertEqual(len(data["top_predictions"]), 3)
        self.assertEqual(len(data["all_scores"]), 15)
        self.assertEqual(self.model.calls, 1)

    def test_multiple_leaves_still_predict(self):
        response = self._post("leaves.png", create_multiple_leaves_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.model.calls, 1)

    def test_rejection_keeps_admission_control_bookkeeping_clean(self):
        import backend.app.api.predict as predict_module

        response = self._post("person.png", create_portrait_bytes())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], input_gate.CODE_NON_LEAF)
        with predict_module._decode_waiters_lock:
            self.assertEqual(predict_module._decode_waiters, 0)

        follow_up = self._post("leaf.png", create_leaf_closeup_bytes())
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(self.model.calls, 1)

    def test_rejection_body_does_not_leak_paths_or_tracebacks(self):
        response = self._post("person.png", create_portrait_bytes())
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("C:\\", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("keras", response.text.lower())


if __name__ == "__main__":
    unittest.main()
