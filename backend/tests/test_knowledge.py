"""
Tests for the Disease Knowledge Base service, Pydantic schemas, and API endpoint.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.app.config import KNOWLEDGE_BASE_PATH
from backend.app.main import create_app
from backend.app.schemas.knowledge import DiseaseKnowledgeRecord
from backend.app.services.inference import CLASS_ID_TO_METADATA, TARGET_CLASS_MAPPING
from backend.app.services.knowledge import KnowledgeBaseService, knowledge_service


class TestKnowledgeBaseService(unittest.TestCase):
    """Unit tests for the KnowledgeBaseService lifecycle and validation logic."""

    @classmethod
    def setUpClass(cls):
        # Ensure knowledge base is loaded from the authoritative path
        knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)

    def test_service_loads_production_data_successfully(self):
        """Service should load all 15 records and report is_loaded=True."""
        self.assertTrue(knowledge_service.is_loaded)
        self.assertIsNone(knowledge_service.load_error)
        all_records = knowledge_service.get_all_records()
        self.assertEqual(len(all_records), 15)
        for cid in range(15):
            self.assertIn(cid, all_records)

    def test_class_mapping_parity_all_15_classes(self):
        """All 15 knowledge records must strictly match inference TARGET_CLASS_MAPPING."""
        for cid, expected_meta in CLASS_ID_TO_METADATA.items():
            record = knowledge_service.get_disease_record(cid)
            self.assertIsNotNone(record, f"Record for class_id {cid} should exist")
            self.assertEqual(record["class_id"], cid)
            self.assertEqual(record["raw_class"], expected_meta["raw_class"])
            self.assertEqual(record["crop"], expected_meta["crop"])
            self.assertEqual(record["condition"], expected_meta["condition"])
            self.assertEqual(record["health_status"], expected_meta["health"])

    def test_has_class(self):
        """has_class should return True for 0..14 and False for out-of-range IDs."""
        for cid in range(15):
            self.assertTrue(knowledge_service.has_class(cid))
        self.assertFalse(knowledge_service.has_class(-1))
        self.assertFalse(knowledge_service.has_class(15))
        self.assertFalse(knowledge_service.has_class(99))

    def test_defensive_copies_immutability(self):
        """Modifying a retrieved record must not alter the internal service state."""
        record = knowledge_service.get_disease_record(0)
        self.assertEqual(record["crop"], "Bell Pepper")
        # Mutate the copy
        record["crop"] = "MUTATED_CROP"
        record["symptoms"]["summary"] = "MUTATED_SUMMARY"

        # Fetch again and confirm original values remain intact
        fresh_record = knowledge_service.get_disease_record(0)
        self.assertEqual(fresh_record["crop"], "Bell Pepper")
        self.assertNotEqual(fresh_record["symptoms"]["summary"], "MUTATED_SUMMARY")

    def test_get_all_records_defensive_copy(self):
        """Modifying the dictionary from get_all_records must not alter internal service state."""
        all_recs = knowledge_service.get_all_records()
        all_recs[0]["crop"] = "MUTATED_CROP"

        fresh_recs = knowledge_service.get_all_records()
        self.assertEqual(fresh_recs[0]["crop"], "Bell Pepper")

    def test_unloaded_service_raises_runtime_error(self):
        """Querying an unloaded service instance must raise RuntimeError."""
        unloaded_service = KnowledgeBaseService()
        self.assertFalse(unloaded_service.is_loaded)
        with self.assertRaises(RuntimeError):
            unloaded_service.get_disease_record(0)
        with self.assertRaises(RuntimeError):
            unloaded_service.get_all_records()

    def test_missing_file_raises_file_not_found(self):
        """Loading a non-existent file path must raise FileNotFoundError and clear records."""
        service = KnowledgeBaseService()
        non_existent = Path("non_existent_knowledge_base_file_xyz.json")
        with self.assertRaises(FileNotFoundError):
            service.load_knowledge_base(non_existent)
        self.assertFalse(service.is_loaded)
        self.assertIn("not found", service.load_error)

    def test_corrupt_json_raises_value_error(self):
        """Loading malformed JSON syntax must raise ValueError."""
        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            tf.write("INVALID JSON CONTENT { [")
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("malformed JSON", str(ctx.exception))
            self.assertFalse(service.is_loaded)
        finally:
            tf_path.unlink(missing_ok=True)

    def test_non_list_root_raises_value_error(self):
        """JSON root that is an object instead of a list must raise ValueError."""
        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump({"class_id": 0}, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("JSON array", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_record_count_mismatch_raises_value_error(self):
        """Loading fewer or more than 15 records must raise ValueError."""
        all_records = list(knowledge_service.get_all_records().values())
        truncated_records = all_records[:14]  # only 14 records

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(truncated_records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("record count mismatch", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_raw_class_mismatch_raises_value_error(self):
        """Record with mismatched raw_class must fail validation."""
        records = copy.deepcopy(list(knowledge_service.get_all_records().values()))
        records[0]["raw_class"] = "Invalid___Raw_Class"

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("raw_class mismatch", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_crop_rotation_as_list_raises_value_error(self):
        """Record with crop_rotation as a list instead of str must fail validation."""
        records = copy.deepcopy(list(knowledge_service.get_all_records().values()))
        records[0]["prevention_and_cultural"]["crop_rotation"] = ["Rotate 1 year", "Rotate 2 years"]

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("crop_rotation' must be a non-empty string", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_healthy_class_with_causal_agent_raises_value_error(self):
        """Healthy class (e.g. class_id 1) having a causal_agent string must fail validation."""
        records = copy.deepcopy(list(knowledge_service.get_all_records().values()))
        records[1]["causal_agent"] = "Xanthomonas campestris"

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("causal_agent=null", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_healthy_class_recommending_pesticides_raises_value_error(self):
        """Healthy class recommending chemical spraying must fail validation."""
        records = copy.deepcopy(list(knowledge_service.get_all_records().values()))
        records[1]["treatment"]["ipm_chemical_principles"] = [
            "Apply fungicide spray every 7 to 10 days"
        ]

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("cannot recommend pesticide/fungicide application", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_invalid_source_url_scheme_raises_value_error(self):
        """Source reference with a non-HTTP/HTTPS URL scheme must fail validation."""
        records = copy.deepcopy(list(knowledge_service.get_all_records().values()))
        records[0]["sources"][0]["url"] = "ftp://invalid-server/file.pdf"

        service = KnowledgeBaseService()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(records, tf)
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                service.load_knowledge_base(tf_path)
            self.assertIn("is not a valid HTTP/HTTPS URL", str(ctx.exception))
        finally:
            tf_path.unlink(missing_ok=True)


class TestKnowledgeAPIEndpoint(unittest.TestCase):
    """Integration tests for the GET /api/knowledge/{class_id} endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        # Ensure knowledge service is loaded
        knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)

    def test_all_15_class_ids_return_200_and_validate_schema(self):
        """GET /api/knowledge/{class_id} must return 200 and valid schema for all 15 classes."""
        for cid in range(15):
            response = self.client.get(f"/api/knowledge/{cid}")
            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 for class_id {cid}, got {response.status_code}: {response.text}",
            )
            data = response.json()

            # Validate against Pydantic schema
            record = DiseaseKnowledgeRecord(**data)
            self.assertEqual(record.class_id, cid)

            # Validate metadata matching
            expected_meta = CLASS_ID_TO_METADATA[cid]
            self.assertEqual(record.raw_class, expected_meta["raw_class"])
            self.assertEqual(record.crop, expected_meta["crop"])
            self.assertEqual(record.condition, expected_meta["condition"])
            self.assertEqual(record.health_status, expected_meta["health"])

            # Validate non-empty sub-objects
            self.assertTrue(record.symptoms.summary)
            self.assertTrue(len(record.symptoms.foliar_symptoms) > 0)
            self.assertTrue(record.favorable_conditions.environmental_factors)
            self.assertIsInstance(record.prevention_and_cultural.crop_rotation, str)
            self.assertTrue(len(record.prevention_and_cultural.sanitation) > 0)
            self.assertTrue(record.treatment.urgency)
            self.assertTrue(len(record.treatment.ipm_chemical_principles) > 0)
            self.assertTrue(record.extension_consultation.local_specialist)
            self.assertTrue(len(record.sources) >= 1)

    def test_healthy_classes_safety_restrictions(self):
        """Classes 1, 4, 14 must have causal_agent=null and no chemical spray recommendations."""
        healthy_cids = [1, 4, 14]
        for cid in healthy_cids:
            response = self.client.get(f"/api/knowledge/{cid}")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["health_status"], "Healthy")
            self.assertIsNone(data["causal_agent"])

            # IPM principles should explicitly note no chemical spray needed
            ipm_principles = " ".join(data["treatment"]["ipm_chemical_principles"]).lower()
            self.assertTrue(
                "no pesticide" in ipm_principles or "no ... application" in ipm_principles or "warranted" in ipm_principles,
                f"Healthy class {cid} must disclaim chemical application",
            )

    def test_diseased_classes_have_causal_agent(self):
        """All 12 diseased classes must have a non-empty causal_agent string."""
        diseased_cids = [c for c in range(15) if c not in (1, 4, 14)]
        for cid in diseased_cids:
            response = self.client.get(f"/api/knowledge/{cid}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["health_status"], "Diseased")
            self.assertIsNotNone(data["causal_agent"])
            self.assertTrue(len(data["causal_agent"].strip()) > 0)

    def test_unknown_class_ids_return_404(self):
        """Out-of-range integer class IDs must return 404 with a sanitized message."""
        test_ids = [-1, 15, 99, 100]
        for cid in test_ids:
            response = self.client.get(f"/api/knowledge/{cid}")
            self.assertEqual(
                response.status_code,
                404,
                f"Expected 404 for unknown class_id {cid}, got {response.status_code}",
            )
            data = response.json()
            self.assertIn("detail", data)
            self.assertIn(f"not found for class_id {cid}", data["detail"])
            # Ensure no filesystem path or internal traceback is leaked
            self.assertNotIn("backend", data["detail"].lower())
            self.assertNotIn(".json", data["detail"].lower())
            self.assertNotIn("traceback", response.text.lower())

    def test_non_integer_class_id_returns_422(self):
        """Passing non-integer string path parameter must return 422 validation error."""
        response = self.client.get("/api/knowledge/invalid_class")
        self.assertEqual(response.status_code, 422)

    def test_service_unavailable_returns_503_without_leakage(self):
        """When knowledge base service is unavailable, endpoint must return 503 without leaking paths."""
        # Unload knowledge service and inject a mock sensitive error string
        knowledge_service.unload()
        knowledge_service._load_error = "Mock internal error at C:\\Users\\secret\\path\\disease_knowledge_base.json"

        try:
            response = self.client.get("/api/knowledge/0")
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertEqual(data["detail"], "Knowledge base service is currently unavailable.")
            # Verify no secret internal path or filename is leaked in the public response
            self.assertNotIn("secret", response.text)
            self.assertNotIn("disease_knowledge_base.json", response.text)
            self.assertNotIn("C:\\", response.text)
        finally:
            # Restore knowledge service for subsequent tests
            knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)


if __name__ == "__main__":
    unittest.main()
