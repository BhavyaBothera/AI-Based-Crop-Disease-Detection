"""
Unit tests for image preprocessing, validation safeguards, and prediction endpoints.
Uses mock/injected models for fast, deterministic unit test execution.
"""
import asyncio
import threading
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport

from backend.app.main import create_app
from backend.app.services.inference import (
    CLASS_ID_TO_METADATA,
    TARGET_CLASS_MAPPING,
    model_service,
    preprocess_image,
)
from backend.app.config import MAX_UPLOAD_SIZE_BYTES
from backend.tests.image_fixtures import (
    create_textured_image_bytes as create_synthetic_image_bytes,
)


class MockCropModel:
    """Mock model that returns fixed or customizable 15-class softmax distribution."""

    def __init__(self, target_class_id: int = 2, target_score: float = 0.95):
        self.target_class_id = target_class_id
        self.target_score = target_score

    def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        # Generate 15-class scores summing to 1.0
        scores = np.full(15, (1.0 - self.target_score) / 14.0, dtype=np.float32)
        scores[self.target_class_id] = self.target_score
        return np.expand_dims(scores, axis=0)


class TestPredictEndpoint(unittest.TestCase):
    """Automated tests for prediction API and preprocessing pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self):
        # Reset model service state after each test
        model_service.unload_model()

    def test_authoritative_mapping_parity(self):
        """Verify that backend TARGET_CLASS_MAPPING exactly matches authoritative ML mapping."""
        from ml.scripts.prepare_dataset import TARGET_CLASS_MAPPING as ML_MAPPING

        self.assertEqual(len(TARGET_CLASS_MAPPING), 15)
        self.assertEqual(len(ML_MAPPING), 15)
        self.assertEqual(TARGET_CLASS_MAPPING, ML_MAPPING)

        # Check all class IDs from 0 to 14 are present
        class_ids = {v["id"] for v in TARGET_CLASS_MAPPING.values()}
        self.assertEqual(class_ids, set(range(15)))

    def test_preprocess_image_valid_jpeg(self):
        """Preprocessing must output (1, 224, 224, 3) float32 in raw [0, 255] range."""
        raw_bytes = create_synthetic_image_bytes(format="JPEG", size=(320, 240))
        tensor = preprocess_image(raw_bytes)

        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        # Verify values remain in raw [0.0, 255.0] range (NOT pre-normalized to [-1, 1] or [0, 1])
        self.assertGreater(tensor.max(), 1.0)
        self.assertLessEqual(tensor.max(), 255.0)
        self.assertGreaterEqual(tensor.min(), 0.0)

    def test_preprocess_image_rgba_png_conversion(self):
        """PNG with Alpha channel must be safely converted to 3-channel RGB."""
        raw_bytes = create_synthetic_image_bytes(format="PNG", mode="RGBA", color=(50, 150, 50, 255))
        tensor = preprocess_image(raw_bytes)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_preprocess_image_webp_format(self):
        """WEBP format images must decode properly."""
        raw_bytes = create_synthetic_image_bytes(format="WEBP", size=(150, 150))
        tensor = preprocess_image(raw_bytes)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_valid_image_prediction_response(self):
        """POST /api/predict with valid image returns 200 and schema-compliant response."""
        # Inject mock model predicting class 2 (Potato___Early_blight)
        model_service.set_model(MockCropModel(target_class_id=2, target_score=0.94))

        img_bytes = create_synthetic_image_bytes(format="JPEG", size=(256, 256))
        response = self.client.post(
            "/api/predict",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check top prediction
        self.assertEqual(data["predicted_class_id"], 2)
        self.assertEqual(data["raw_class"], "Potato___Early_blight")
        self.assertEqual(data["crop"], "Potato")
        self.assertEqual(data["condition"], "Early Blight")
        self.assertEqual(data["health"], "Diseased")
        self.assertAlmostEqual(data["score"], 0.94, places=4)

        # Check top 3 predictions list
        top3 = data["top_predictions"]
        self.assertEqual(len(top3), 3)
        self.assertEqual(top3[0]["rank"], 1)
        self.assertEqual(top3[0]["class_id"], 2)
        self.assertEqual(top3[0]["crop"], "Potato")
        self.assertEqual(top3[0]["condition"], "Early Blight")
        self.assertGreaterEqual(top3[0]["score"], top3[1]["score"])
        self.assertGreaterEqual(top3[1]["score"], top3[2]["score"])

        # Check all_scores contains all 15 classes
        self.assertEqual(len(data["all_scores"]), 15)
        for class_name in TARGET_CLASS_MAPPING:
            self.assertIn(class_name, data["all_scores"])

        # Check advisory disclaimer is included
        self.assertIn("disclaimer", data)
        self.assertIn("informational", data["disclaimer"].lower())

    def test_alternative_image_field_name(self):
        """POST /api/predict accepts upload under 'image' field as well as 'file'."""
        model_service.set_model(MockCropModel(target_class_id=14, target_score=0.98))

        img_bytes = create_synthetic_image_bytes(format="PNG", size=(128, 128))
        response = self.client.post(
            "/api/predict",
            files={"image": ("tomato_healthy.png", img_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_class_id"], 14)
        self.assertEqual(data["raw_class"], "Tomato___healthy")
        self.assertEqual(data["health"], "Healthy")

    def test_empty_upload_returns_400(self):
        """Empty uploaded file (0 bytes) must return HTTP 400."""
        model_service.set_model(MockCropModel())
        response = self.client.post(
            "/api/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_malformed_image_returns_415(self):
        """Payload matching no supported file signature must return HTTP 415."""
        model_service.set_model(MockCropModel())
        response = self.client.post(
            "/api/predict",
            files={"file": ("corrupt.jpg", b"NOT_A_VALID_IMAGE_DATA_HEADER_12345", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("does not match detected file signature", response.json()["detail"].lower())

    def test_unsupported_format_returns_415(self):
        """Unsupported declared MIME type (e.g., GIF) must return HTTP 415."""
        model_service.set_model(MockCropModel())
        gif_bytes = create_synthetic_image_bytes(format="GIF", size=(50, 50))
        response = self.client.post(
            "/api/predict",
            files={"file": ("animation.gif", gif_bytes, "image/gif")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("unsupported media type", response.json()["detail"].lower())

    def test_oversized_upload_returns_413(self):
        """Uploads exceeding 10 MB must return HTTP 413 without loading into memory."""
        model_service.set_model(MockCropModel())
        # Generate payload larger than 10 MB
        oversized_bytes = b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024)
        response = self.client.post(
            "/api/predict",
            files={"file": ("huge_image.jpg", oversized_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds", response.json()["detail"].lower())

    def test_model_not_ready_returns_503(self):
        """POST /api/predict must return HTTP 503 if model is not loaded."""
        model_service.unload_model()
        img_bytes = create_synthetic_image_bytes(format="JPEG")
        response = self.client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json()["detail"].lower())

    def test_overload_rejection_returns_429_without_leaking_counter(self):
        """Decode-queue admission control must reject with 429 and not leak waiter accounting."""
        import backend.app.api.predict as predict_module

        model_service.set_model(MockCropModel())
        img_bytes = create_synthetic_image_bytes(format="JPEG", size=(100, 100))

        # Simulate a full admission queue without needing timing-dependent concurrency.
        with predict_module._decode_waiters_lock:
            predict_module._decode_waiters = predict_module.MAX_DECODE_QUEUE_SIZE
        try:
            response = self.client.post(
                "/api/predict",
                files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
            )
        finally:
            with predict_module._decode_waiters_lock:
                predict_module._decode_waiters = 0

        self.assertEqual(response.status_code, 429)
        self.assertIn("peak request volume", response.json()["detail"].lower())

        # A subsequent request must be admitted and leave the counter clean.
        follow_up = self.client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(follow_up.status_code, 200)
        with predict_module._decode_waiters_lock:
            self.assertEqual(predict_module._decode_waiters, 0)

    def test_no_file_field_returns_400(self):
        """Request without 'file' or 'image' multipart field returns HTTP 400."""
        model_service.set_model(MockCropModel())
        response = self.client.post(
            "/api/predict",
            data={"some_other_field": "value"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("no image uploaded", response.json()["detail"].lower())

    def test_model_unready_does_not_leak_internal_paths_or_tracebacks(self):
        """POST /api/predict must return generic 503 error without leaking host paths."""
        model_service.unload_model()
        model_service._load_error = "Model loading error at: C:\\Users\\secret\\internal\\deep\\path\\model.keras"
        img_bytes = create_synthetic_image_bytes(format="JPEG")
        response = self.client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["detail"], "Model inference service is currently unavailable.")
        self.assertNotIn("secret", response.text)
        self.assertNotIn("C:\\", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_oversized_pixel_dimensions_rejected_422(self):
        """Images with decoded dimensions exceeding MAX_IMAGE_PIXELS (20 MP) must return HTTP 422."""
        model_service.set_model(MockCropModel())
        # Create a synthetic image of 5000 x 5000 (25 MP > 20.97 MP)
        # Note: In-memory solid JPEG encoding of 5000x5000 takes only ~120 KB, staying well under 10 MB upload limit
        huge_dim_bytes = create_synthetic_image_bytes(format="JPEG", size=(5000, 5000))
        response = self.client.post(
            "/api/predict",
            files={"file": ("huge_leaf.jpg", huge_dim_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("exceeds maximum permitted limit", response.json()["detail"].lower())

    def test_pillow_decompression_bomb_safeguards(self):
        """Direct unit test that DecompressionBombWarning and DecompressionBombError are caught as ValueError."""
        from unittest.mock import patch
        img_bytes = create_synthetic_image_bytes(format="JPEG", size=(100, 100))

        # Test DecompressionBombError is caught and converted to safe ValueError
        with patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = Image.DecompressionBombError("Simulated decompression bomb")
            with self.assertRaises(ValueError) as ctx:
                preprocess_image(img_bytes)
            self.assertIn("exceeds safe decompression limits", str(ctx.exception).lower())

        # Test DecompressionBombWarning is caught and converted to safe ValueError
        with patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = Image.DecompressionBombWarning("Simulated warning bomb")
            with self.assertRaises(ValueError) as ctx:
                preprocess_image(img_bytes)
            self.assertIn("exceeds safe decompression limits", str(ctx.exception).lower())

    def test_inference_does_not_block_event_loop(self):
        """
        Verify CPU inference offloading does not block FastAPI's async event loop.
        An unrelated async endpoint (/health) must execute and return 200 while
        model inference is in-flight on a worker thread.
        Uses deterministic threading events rather than flaky wall-clock assertions.
        """
        class SlowMockModel:
            def __init__(self, started_evt: threading.Event, release_evt: threading.Event):
                self.started_evt = started_evt
                self.release_evt = release_evt

            def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
                self.started_evt.set()
                # Block in worker thread until released by the test
                self.release_evt.wait(timeout=5.0)
                scores = np.full(15, 0.05 / 14.0, dtype=np.float32)
                scores[2] = 0.95
                return np.expand_dims(scores, axis=0)

        async def run_verification():
            started_evt = threading.Event()
            release_evt = threading.Event()
            model_service.set_model(SlowMockModel(started_evt, release_evt))

            transport = ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                img_bytes = create_synthetic_image_bytes(format="JPEG", size=(100, 100))

                # Launch inference request asynchronously
                predict_task = asyncio.create_task(
                    ac.post("/api/predict", files={"file": ("leaf.jpg", img_bytes, "image/jpeg")})
                )

                # Wait until inference starts running in its worker thread.
                # Bounded polling: never spin forever if the request is rejected
                # before reaching the model (e.g. by an input-validation guard).
                for _ in range(1000):  # ~5 seconds at 5 ms per poll
                    if started_evt.is_set() or predict_task.done():
                        break
                    await asyncio.sleep(0.005)

                if not started_evt.is_set():
                    release_evt.set()  # unblock the worker thread before failing
                    if predict_task.done():
                        early = predict_task.result()
                        self.fail(
                            "Inference never started; the request was rejected during "
                            f"validation (HTTP {early.status_code}: {early.text})."
                        )
                    self.fail("Inference did not start within the 5 second window.")

                # Inference is now running in background thread.
                # If the event loop were blocked, this GET /health request would hang.
                health_response = await ac.get("/health")
                self.assertEqual(health_response.status_code, 200)
                self.assertEqual(health_response.json()["status"], "ok")

                # Release the worker thread to finish inference
                release_evt.set()

                # Await prediction completion
                predict_response = await predict_task
                self.assertEqual(predict_response.status_code, 200)
                data = predict_response.json()
                self.assertEqual(data["predicted_class_id"], 2)

        asyncio.run(run_verification())

    def test_concurrent_prediction_requests_safety(self):
        """Verify concurrent prediction requests execute safely under the bounded strategy."""
        async def run_concurrent():
            model_service.set_model(MockCropModel(target_class_id=2, target_score=0.92))
            transport = ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                img_bytes = create_synthetic_image_bytes(format="JPEG", size=(100, 100))

                async def make_request():
                    return await ac.post("/api/predict", files={"file": ("leaf.jpg", img_bytes, "image/jpeg")})

                responses = await asyncio.gather(*(make_request() for _ in range(4)))
                for resp in responses:
                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    self.assertEqual(data["predicted_class_id"], 2)
                    self.assertEqual(len(data["top_predictions"]), 3)

        asyncio.run(run_concurrent())

    def test_model_runtime_error_offloaded_returns_503(self):
        """Verify RuntimeError raised during offloaded inference is caught and returns 503."""
        class ErrorMockModel:
            def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
                raise RuntimeError("Simulated inference engine failure")

        model_service.set_model(ErrorMockModel())
        img_bytes = create_synthetic_image_bytes(format="JPEG", size=(100, 100))
        response = self.client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Model inference service is currently unavailable.")


if __name__ == "__main__":
    unittest.main()
