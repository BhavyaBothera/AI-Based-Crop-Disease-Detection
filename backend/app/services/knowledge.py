"""
Knowledge base service for crop disease information.

Loads, validates, and serves validated agronomic knowledge base records
for the 15 supported Solanaceae model classes.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import json
import logging
from urllib.parse import urlparse

from ..config import (
    DEFAULT_KNOWLEDGE_BASE_REL_PATH,
    KNOWLEDGE_BASE_PATH,
    PROJECT_ROOT,
)
from ..schemas.knowledge import DiseaseKnowledgeRecord
from .inference import CLASS_ID_TO_METADATA, TARGET_CLASS_MAPPING

logger = logging.getLogger(__name__)

# Expected healthy class IDs
HEALTHY_CLASS_IDS = {1, 4, 14}


class KnowledgeBaseService:
    """
    Thread-safe knowledge base service managing disease knowledge records.
    Loads once on startup, validates all records against authoritative mappings,
    and returns defensive copies to ensure immutability.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self._data_path: Path = data_path or KNOWLEDGE_BASE_PATH
        self._records: Dict[int, Dict[str, Any]] = {}
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the knowledge base is loaded and validated."""
        return len(self._records) == len(TARGET_CLASS_MAPPING) and self._load_error is None

    @property
    def data_path(self) -> Path:
        """Configured knowledge base path (internal server use only)."""
        return self._data_path

    @property
    def safe_data_identifier(self) -> str:
        """
        Returns a sanitized data path identifier (project-relative or basename),
        never leaking host filesystem paths in public responses.
        """
        if self._data_path is None:
            return DEFAULT_KNOWLEDGE_BASE_REL_PATH
        try:
            return self._data_path.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            return self._data_path.name

    @property
    def load_error(self) -> Optional[str]:
        """Returns the last loading error message for internal server logging."""
        return self._load_error

    def unload(self) -> None:
        """Unloads the knowledge base and resets service state."""
        self._records.clear()
        self._load_error = None
        logger.info("Knowledge base unloaded.")

    def set_records(self, records: Dict[int, Dict[str, Any]]) -> None:
        """Injects pre-validated records (used for testing/mocking)."""
        self._records = copy.deepcopy(records)
        self._load_error = None

    def load_knowledge_base(self, path: Optional[Path] = None) -> None:
        """
        Loads the knowledge base JSON from disk and validates all 15 records.

        Validates:
        - Exact JSON format and file existence.
        - Exactly 15 unique records with class IDs 0 through 14.
        - 1:1 parity with authoritative model mapping (raw_class, crop, condition, health_status).
        - Healthy-class safety rules (null causal agent, no chemical pesticide/fungicide recommendations).
        - Prevention field types, strictly enforcing crop_rotation as a string.
        - Source references structure and URL schemes (offline, no network calls).
        - Pydantic schema conformance.

        Args:
            path: Optional override path to the knowledge base JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If JSON content or any record violates validation rules.
        """
        target_path = path or self._data_path
        logger.info("Loading disease knowledge base from %s...", target_path)

        if not target_path.is_file():
            self._records.clear()
            self._load_error = f"Knowledge base file not found at: {target_path}"
            logger.error(self._load_error)
            raise FileNotFoundError(self._load_error)

        try:
            raw_text = target_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self._records.clear()
            self._load_error = f"Knowledge base contains malformed JSON: {str(exc)}"
            logger.error(self._load_error, exc_info=True)
            raise ValueError(self._load_error) from exc
        except Exception as exc:
            self._records.clear()
            self._load_error = f"Failed to read knowledge base file: {str(exc)}"
            logger.error(self._load_error, exc_info=True)
            raise ValueError(self._load_error) from exc

        if not isinstance(data, list):
            self._records.clear()
            self._load_error = "Knowledge base root must be a JSON array of records."
            logger.error(self._load_error)
            raise ValueError(self._load_error)

        expected_count = len(TARGET_CLASS_MAPPING)
        if len(data) != expected_count:
            self._records.clear()
            self._load_error = (
                f"Knowledge base record count mismatch: expected exactly {expected_count} records, "
                f"got {len(data)}."
            )
            logger.error(self._load_error)
            raise ValueError(self._load_error)

        validated_records: Dict[int, Dict[str, Any]] = {}
        seen_class_ids = set()

        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                self._records.clear()
                self._load_error = f"Record at index {idx} is not a JSON object."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            # 1. Validate class_id
            if "class_id" not in record:
                self._records.clear()
                self._load_error = f"Record at index {idx} missing required field 'class_id'."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            cid = record["class_id"]
            if not isinstance(cid, int) or isinstance(cid, bool):
                self._records.clear()
                self._load_error = f"Record at index {idx} has non-integer class_id: {cid}."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if cid < 0 or cid >= expected_count:
                self._records.clear()
                self._load_error = f"Record has out-of-range class_id: {cid} (expected 0 to {expected_count - 1})."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if cid in seen_class_ids:
                self._records.clear()
                self._load_error = f"Duplicate class_id detected: {cid}."
                logger.error(self._load_error)
                raise ValueError(self._load_error)
            seen_class_ids.add(cid)

            # 2. Authoritative model mapping parity
            expected_meta = CLASS_ID_TO_METADATA.get(cid)
            if expected_meta is None:
                self._records.clear()
                self._load_error = f"No authoritative metadata found for class_id {cid}."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if record.get("raw_class") != expected_meta["raw_class"]:
                self._records.clear()
                self._load_error = (
                    f"Class ID {cid} raw_class mismatch: expected '{expected_meta['raw_class']}', "
                    f"got '{record.get('raw_class')}'."
                )
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if record.get("crop") != expected_meta["crop"]:
                self._records.clear()
                self._load_error = (
                    f"Class ID {cid} crop mismatch: expected '{expected_meta['crop']}', "
                    f"got '{record.get('crop')}'."
                )
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if record.get("condition") != expected_meta["condition"]:
                self._records.clear()
                self._load_error = (
                    f"Class ID {cid} condition mismatch: expected '{expected_meta['condition']}', "
                    f"got '{record.get('condition')}'."
                )
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if record.get("health_status") != expected_meta["health"]:
                self._records.clear()
                self._load_error = (
                    f"Class ID {cid} health_status mismatch: expected '{expected_meta['health']}', "
                    f"got '{record.get('health_status')}'."
                )
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            # 3. Healthy vs Diseased rules & Causal agent
            is_healthy = cid in HEALTHY_CLASS_IDS
            causal_agent = record.get("causal_agent")

            if is_healthy:
                if causal_agent is not None:
                    self._records.clear()
                    self._load_error = (
                        f"Healthy class ID {cid} must have causal_agent=null, got: '{causal_agent}'."
                    )
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)
            else:
                if not causal_agent or not isinstance(causal_agent, str) or not causal_agent.strip():
                    self._records.clear()
                    self._load_error = (
                        f"Diseased class ID {cid} must have a non-empty causal_agent string."
                    )
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # 4. Prevention and cultural management: crop_rotation MUST be string
            prev = record.get("prevention_and_cultural")
            if not isinstance(prev, dict):
                self._records.clear()
                self._load_error = f"Class ID {cid} missing 'prevention_and_cultural' dict."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            crop_rot = prev.get("crop_rotation")
            if not isinstance(crop_rot, str) or isinstance(crop_rot, (list, tuple)) or not crop_rot.strip():
                self._records.clear()
                self._load_error = (
                    f"Class ID {cid} 'prevention_and_cultural.crop_rotation' must be a non-empty string, "
                    f"got: {type(crop_rot).__name__}."
                )
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            for list_field in ["sanitation", "irrigation_and_airflow", "resistant_cultivars"]:
                val = prev.get(list_field)
                if not isinstance(val, list) or not val:
                    self._records.clear()
                    self._load_error = f"Class ID {cid} 'prevention_and_cultural.{list_field}' must be a non-empty list."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)
                if not all(isinstance(item, str) and item.strip() for item in val):
                    self._records.clear()
                    self._load_error = f"Class ID {cid} 'prevention_and_cultural.{list_field}' contains non-string or empty elements."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # 5. Treatment validation & Healthy chemical restrictions
            treat = record.get("treatment")
            if not isinstance(treat, dict):
                self._records.clear()
                self._load_error = f"Class ID {cid} missing 'treatment' dict."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            for t_field in ["urgency", "non_chemical_controls", "ipm_chemical_principles", "safety_and_regulatory_constraints"]:
                if t_field not in treat:
                    self._records.clear()
                    self._load_error = f"Class ID {cid} missing 'treatment.{t_field}'."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            if not isinstance(treat["urgency"], str) or not treat["urgency"].strip():
                self._records.clear()
                self._load_error = f"Class ID {cid} 'treatment.urgency' must be a non-empty string."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            for list_field in ["non_chemical_controls", "ipm_chemical_principles", "safety_and_regulatory_constraints"]:
                val = treat[list_field]
                if not isinstance(val, list) or not val:
                    self._records.clear()
                    self._load_error = f"Class ID {cid} 'treatment.{list_field}' must be a non-empty list."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)
                if not all(isinstance(item, str) and item.strip() for item in val):
                    self._records.clear()
                    self._load_error = f"Class ID {cid} 'treatment.{list_field}' contains non-string or empty items."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # Enforce that Healthy classes have no pesticide/fungicide spray recommendations
            if is_healthy:
                chem_text = " ".join(treat["ipm_chemical_principles"]).lower()
                forbidden_terms = ["apply fungicide", "apply pesticide", "spray fungicide", "spray copper"]
                if any(term in chem_text for term in forbidden_terms):
                    self._records.clear()
                    self._load_error = f"Healthy class ID {cid} cannot recommend pesticide/fungicide application."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # 6. Symptoms validation
            symp = record.get("symptoms")
            if not isinstance(symp, dict):
                self._records.clear()
                self._load_error = f"Class ID {cid} missing 'symptoms' dict."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if not isinstance(symp.get("summary"), str) or not symp["summary"].strip():
                self._records.clear()
                self._load_error = f"Class ID {cid} 'symptoms.summary' must be a non-empty string."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            for s_field in ["foliar_symptoms", "stem_and_fruit_symptoms", "look_alike_conditions"]:
                val = symp.get(s_field)
                if not isinstance(val, list) or not val or not all(isinstance(x, str) and x.strip() for x in val):
                    self._records.clear()
                    self._load_error = f"Class ID {cid} 'symptoms.{s_field}' must be a non-empty list of strings."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # 7. Favorable conditions validation
            fav = record.get("favorable_conditions")
            if not isinstance(fav, dict):
                self._records.clear()
                self._load_error = f"Class ID {cid} missing 'favorable_conditions' dict."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if not isinstance(fav.get("environmental_factors"), str) or not fav["environmental_factors"].strip():
                self._records.clear()
                self._load_error = f"Class ID {cid} 'favorable_conditions.environmental_factors' must be a non-empty string."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if not isinstance(fav.get("transmission_routes"), list) or not fav["transmission_routes"] or not all(isinstance(x, str) and x.strip() for x in fav["transmission_routes"]):
                self._records.clear()
                self._load_error = f"Class ID {cid} 'favorable_conditions.transmission_routes' must be a non-empty list of strings."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            # 8. Extension consultation validation
            ext = record.get("extension_consultation")
            if not isinstance(ext, dict):
                self._records.clear()
                self._load_error = f"Class ID {cid} missing 'extension_consultation' dict."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if not isinstance(ext.get("local_specialist"), str) or not ext["local_specialist"].strip():
                self._records.clear()
                self._load_error = f"Class ID {cid} 'extension_consultation.local_specialist' must be a non-empty string."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            if not isinstance(ext.get("escalation_triggers"), list) or not ext["escalation_triggers"] or not all(isinstance(x, str) and x.strip() for x in ext["escalation_triggers"]):
                self._records.clear()
                self._load_error = f"Class ID {cid} 'extension_consultation.escalation_triggers' must be a non-empty list of strings."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            # 9. Offline source references validation
            sources = record.get("sources")
            if not isinstance(sources, list) or not sources:
                self._records.clear()
                self._load_error = f"Class ID {cid} 'sources' must be a non-empty list."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            for s_idx, src in enumerate(sources):
                if not isinstance(src, dict):
                    self._records.clear()
                    self._load_error = f"Class ID {cid} source {s_idx} is not a dictionary."
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

                for src_field in ["organization", "title", "url", "claims_supported", "reference_url_or_publication"]:
                    val = src.get(src_field)
                    if not isinstance(val, str) or not val.strip():
                        self._records.clear()
                        self._load_error = f"Class ID {cid} source {s_idx} has invalid '{src_field}'."
                        logger.error(self._load_error)
                        raise ValueError(self._load_error)

                # Offline URL scheme validation (no network calls)
                parsed_url = urlparse(src["url"])
                if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                    self._records.clear()
                    self._load_error = (
                        f"Class ID {cid} source {s_idx} URL '{src['url']}' is not a valid HTTP/HTTPS URL."
                    )
                    logger.error(self._load_error)
                    raise ValueError(self._load_error)

            # 10. Safety and uncertainty notes
            sec_notes = record.get("safety_and_uncertainty_notes")
            if not isinstance(sec_notes, str) or not sec_notes.strip():
                self._records.clear()
                self._load_error = f"Class ID {cid} missing non-empty 'safety_and_uncertainty_notes'."
                logger.error(self._load_error)
                raise ValueError(self._load_error)

            # 11. Complete Pydantic schema validation
            try:
                DiseaseKnowledgeRecord(**record)
            except Exception as exc:
                self._records.clear()
                self._load_error = f"Class ID {cid} failed Pydantic schema validation: {str(exc)}"
                logger.error(self._load_error, exc_info=True)
                raise ValueError(self._load_error) from exc

            validated_records[cid] = record

        # Ensure all IDs 0 through expected_count - 1 are present
        if seen_class_ids != set(range(expected_count)):
            missing = sorted(set(range(expected_count)) - seen_class_ids)
            self._records.clear()
            self._load_error = f"Knowledge base missing class IDs: {missing}."
            logger.error(self._load_error)
            raise ValueError(self._load_error)

        self._records = validated_records
        self._data_path = target_path
        self._load_error = None
        logger.info("Successfully loaded and validated %d disease knowledge records.", len(self._records))

    def has_class(self, class_id: int) -> bool:
        """Checks if a class ID exists in the loaded knowledge base."""
        return self.is_loaded and class_id in self._records

    def get_disease_record(self, class_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a defensive deepcopy of the knowledge record for the given class_id.

        Args:
            class_id: Authoritative class index (0-14).

        Returns:
            Optional[Dict[str, Any]]: Deep copy of the knowledge record, or None if not found.

        Raises:
            RuntimeError: If the knowledge base is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Knowledge base service is not loaded.")

        record = self._records.get(class_id)
        if record is None:
            return None

        # Return defensive copy to guarantee caller cannot mutate internal service state
        return copy.deepcopy(record)

    def get_all_records(self) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves defensive deepcopies of all 15 knowledge records.

        Returns:
            Dict[int, Dict[str, Any]]: Map of class_id to deepcopied records.

        Raises:
            RuntimeError: If the knowledge base is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Knowledge base service is not loaded.")

        return copy.deepcopy(self._records)


# Global singleton instance for the FastAPI application lifespan
knowledge_service = KnowledgeBaseService()
