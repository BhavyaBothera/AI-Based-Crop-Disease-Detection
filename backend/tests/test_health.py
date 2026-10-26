"""
Unit tests for health, readiness, and root endpoints.
"""
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.config import KNOWLEDGE_BASE_PATH
from backend.app.main import create_app
from backend.app.services.inference import model_service
from backend.app.services.knowledge import knowledge_service


class MockModel:
    """Mock object to simulate a loaded Keras model."""
    pass


class TestHealthEndpoints(unittest.TestCase):
    """Tests for system health and readiness probes."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self):
        # Reset services state between tests
        model_service.unload_model()
        knowledge_service.unload()

    def test_root_endpoint(self):
        """GET / should return 200 with API metadata and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("docs_url", data)
        self.assertIn("health_url", data)
        self.assertIn("ready_url", data)
        self.assertIn("predict_url", data)
        self.assertIn("knowledge_url", data)

    def test_health_endpoint(self):
        """GET /health should return 200 with basic process operational status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("app", data)
        self.assertIn("version", data)

    def test_ready_endpoint_both_services_ready_returns_200(self):
        """GET /ready should return 200 when both model and knowledge base are loaded."""
        model_service.set_model(MockModel())
        knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)

        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["knowledge_loaded"])
        self.assertIn("model_path", data)
        self.assertNotIn(":\\", data["model_path"])

    def test_ready_endpoint_model_unavailable_returns_503(self):
        """GET /ready should return 503 when model is not loaded even if knowledge base is ready."""
        model_service.unload_model()
        knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)

        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["model_loaded"])
        self.assertTrue(data["knowledge_loaded"])
        self.assertEqual(data["detail"], "Model inference service is not ready.")

    def test_ready_endpoint_knowledge_unavailable_returns_503(self):
        """GET /ready should return 503 when knowledge base is not loaded even if model is ready."""
        model_service.set_model(MockModel())
        knowledge_service.unload()

        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertTrue(data["model_loaded"])
        self.assertFalse(data["knowledge_loaded"])
        self.assertEqual(data["detail"], "Knowledge base service is not ready.")

    def test_ready_endpoint_both_unavailable_returns_503(self):
        """GET /ready should return 503 when both services are not loaded."""
        model_service.unload_model()
        knowledge_service.unload()

        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["model_loaded"])
        self.assertFalse(data["knowledge_loaded"])
        self.assertEqual(data["detail"], "Model inference and knowledge base services are not ready.")

    def test_ready_endpoint_does_not_leak_paths_on_failure(self):
        """GET /ready should never leak absolute paths, tracebacks, or raw load_error strings."""
        model_service.unload_model()
        model_service._load_error = "Model file not found at: C:\\Users\\secret\\internal\\deep\\path\\best_model.keras"
        knowledge_service.unload()
        knowledge_service._load_error = "Knowledge base file not found at: C:\\Users\\secret\\internal\\deep\\path\\disease_knowledge_base.json"

        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["model_loaded"])
        self.assertFalse(data["knowledge_loaded"])
        self.assertEqual(data["detail"], "Model inference and knowledge base services are not ready.")

        # Verify no sensitive path tokens or drives are leaked
        self.assertNotIn("secret", response.text)
        self.assertNotIn("C:\\", response.text)
        self.assertNotIn("internal", response.text)
        self.assertNotIn("Traceback", response.text)
        # model_path should be sanitized relative path
        self.assertNotIn(":\\", data["model_path"])


if __name__ == "__main__":
    unittest.main()
