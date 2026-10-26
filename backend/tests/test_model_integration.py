"""
Integration test verifying actual Keras model loading and end-to-end inference.

REQUIREMENTS & CONSTRAINTS:
- Requires TensorFlow, Keras, and the trained model file:
    ml/models/experiment_weighted/best_model.keras
- DOES NOT load or access the final test dataset (manifest split='test').
- Uses an in-memory synthetic image to verify model input/output tensor shapes and scoring.
- The synthetic image contains macro-scale texture, because a solid-colour canvas
  is rejected by the Level A blank-canvas guard before it reaches the model.
"""
import unittest
from pathlib import Path
import numpy as np

from backend.app.config import DEFAULT_MODEL_PATH
from backend.app.services.inference import (
    ModelService,
    preprocess_image,
)
from backend.tests.image_fixtures import create_textured_image_bytes


class TestRealModelIntegration(unittest.TestCase):
    """End-to-end integration test with the actual trained best_model.keras."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEFAULT_MODEL_PATH
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"Model file not found at {cls.model_path}; skipping integration test.")
        cls.service = ModelService(model_path=cls.model_path)
        cls.service.load_model()

    @classmethod
    def tearDownClass(cls):
        cls.service.unload_model()

    def test_model_loaded_successfully(self):
        """Model must be loaded and report is_loaded True."""
        self.assertTrue(self.service.is_loaded)
        self.assertIsNone(self.service.load_error)

    def test_end_to_end_inference_on_synthetic_image(self):
        """Inference with actual weights must return 15 valid probabilities summing to ~1.0."""
        # Create a synthetic 224x224 leaf-like image with spatial variation so it
        # passes the Level A uniformity guard and exercises real inference.
        raw_bytes = create_textured_image_bytes(format="JPEG", size=(224, 224), color=(45, 120, 45))

        # Preprocess
        input_tensor = preprocess_image(raw_bytes)
        self.assertEqual(input_tensor.shape, (1, 224, 224, 3))
        self.assertEqual(input_tensor.dtype, np.float32)

        # Run inference
        result = self.service.predict(input_tensor)

        # Validate predicted output
        self.assertIn("predicted_class_id", result)
        self.assertIn(result["predicted_class_id"], range(15))
        self.assertIn("raw_class", result)
        self.assertIn("crop", result)
        self.assertIn("condition", result)
        self.assertIn("health", result)
        self.assertIn(result["health"], ["Healthy", "Diseased"])

        # Check score validity
        score = result["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Check top 3 predictions
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(result["top_predictions"][0]["class_id"], result["predicted_class_id"])

        # Check all 15 scores sum to approximately 1.0
        all_scores = result["all_scores"]
        self.assertEqual(len(all_scores), 15)
        total_prob = sum(all_scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
