"""
Field Dataset Manifest Schema & Validation Rules
================================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Defines the authoritative schema for PHYTOSIGNAL field validation sample records,
environmental stratification metadata, privacy protection rules, and manifest loaders.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Authoritative 15-class Solanaceae mapping matching backend.app.services.inference
TARGET_CLASS_MAPPING: Dict[str, Dict[str, Any]] = {
    "Pepper,_bell___Bacterial_spot": {
        "id": 0,
        "crop": "Bell Pepper",
        "health": "Diseased",
        "condition": "Bacterial Spot",
    },
    "Pepper,_bell___healthy": {
        "id": 1,
        "crop": "Bell Pepper",
        "health": "Healthy",
        "condition": "Healthy",
    },
    "Potato___Early_blight": {
        "id": 2,
        "crop": "Potato",
        "health": "Diseased",
        "condition": "Early Blight",
    },
    "Potato___Late_blight": {
        "id": 3,
        "crop": "Potato",
        "health": "Diseased",
        "condition": "Late Blight",
    },
    "Potato___healthy": {
        "id": 4,
        "crop": "Potato",
        "health": "Healthy",
        "condition": "Healthy",
    },
    "Tomato___Bacterial_spot": {
        "id": 5,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Bacterial Spot",
    },
    "Tomato___Early_blight": {
        "id": 6,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Early Blight",
    },
    "Tomato___Late_blight": {
        "id": 7,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Late Blight",
    },
    "Tomato___Leaf_Mold": {
        "id": 8,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Leaf Mold",
    },
    "Tomato___Septoria_leaf_spot": {
        "id": 9,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Septoria Leaf Spot",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "id": 10,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Two-spotted Spider Mite",
    },
    "Tomato___Target_Spot": {
        "id": 11,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Target Spot",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "id": 12,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Yellow Leaf Curl Virus",
    },
    "Tomato___Tomato_mosaic_virus": {
        "id": 13,
        "crop": "Tomato",
        "health": "Diseased",
        "condition": "Mosaic Virus",
    },
    "Tomato___healthy": {
        "id": 14,
        "crop": "Tomato",
        "health": "Healthy",
        "condition": "Healthy",
    },
}

NAME_TO_CLASS_ID: Dict[str, int] = {k: v["id"] for k, v in TARGET_CLASS_MAPPING.items()}
CLASS_ID_TO_NAME: Dict[int, str] = {v["id"]: k for k, v in TARGET_CLASS_MAPPING.items()}

# Mapping of crop to classes
CROP_TO_CLASSES: Dict[str, List[str]] = {
    "Tomato": [k for k, v in TARGET_CLASS_MAPPING.items() if v["crop"] == "Tomato"],
    "Potato": [k for k, v in TARGET_CLASS_MAPPING.items() if v["crop"] == "Potato"],
    "Bell Pepper": [k for k, v in TARGET_CLASS_MAPPING.items() if v["crop"] == "Bell Pepper"],
}

RAW_CLASS_TO_CROP: Dict[str, str] = {k: v["crop"] for k, v in TARGET_CLASS_MAPPING.items()}

# Allowed domain value sets
ALLOWED_CROPS: Set[str] = {"Tomato", "Potato", "Bell Pepper"}
VALID_CROPS = ALLOWED_CROPS

ALLOWED_SPLITS: Set[str] = {"field_dev", "field_test"}
VALID_SPLITS = ALLOWED_SPLITS

ALLOWED_VERIFICATION_STATUS: Set[str] = {
    "visually_suspected",
    "expert_verified",
    "laboratory_confirmed",
    "ambiguous_unresolved",
}
VALID_VERIFICATION_STATUSES = ALLOWED_VERIFICATION_STATUS

ALLOWED_ANNOTATOR_STATUS: Set[str] = {
    "farmer_self_report",
    "trained_scout",
    "trained_agronomist",
    "expert_pathologist",
    "molecular_diagnostic_lab",
}

ALLOWED_LIGHTING_CONDITIONS: Set[str] = {
    "direct_sun",
    "diffuse_overcast",
    "dappled_shade",
    "golden_hour",
    "backlit",
    "indoor_ambient",
    "unknown",
}
VALID_LIGHTING_CONDITIONS = ALLOWED_LIGHTING_CONDITIONS

ALLOWED_BACKGROUND_TYPES: Set[str] = {
    "bare_soil",
    "foliage_clutter",
    "canopy",
    "mulch_plastic",
    "weed_understory",
    "greenhouse",
    "hand",
    "complex_mixed",
    "unknown",
}
VALID_BACKGROUND_TYPES = ALLOWED_BACKGROUND_TYPES

ALLOWED_DISEASE_SEVERITIES: Set[str] = {
    "none",
    "mild",
    "moderate",
    "severe",
    "asymptomatic",
    "unknown",
}
VALID_DISEASE_SEVERITIES = ALLOWED_DISEASE_SEVERITIES

# Non-leaf / out-of-distribution classes for leaf gate evaluation
GATE_OOD_CLASSES: Set[str] = {"non_leaf", "other_plant"}


class ManifestValidationError(ValueError):
    """Raised when a field manifest entry fails validation."""
    pass


@dataclass
class FieldSampleRecord:
    """
    Structured record representing a single field photograph in the evaluation manifest.
    """
    image_id: str
    file_path: str
    crop: str
    ground_truth_class: str
    collection_source: str
    split: str = "field_test"
    capture_device: Optional[str] = None
    capture_date: Optional[str] = None
    collection_location: Optional[str] = None
    lighting_condition: str = "unknown"
    background_type: str = "unknown"
    disease_severity: str = "unknown"
    annotator_status: str = "trained_scout"
    verification_status: str = "visually_suspected"
    duplicate_group_id: str = "NONE"
    notes: Optional[str] = None
    sha256: Optional[str] = None
    dhash: Optional[str] = None

    def validate(self) -> None:
        """
        Validates internal consistency, schemas, and privacy rules.
        Raises ManifestValidationError on any infraction.
        """
        if not self.image_id or not isinstance(self.image_id, str):
            raise ManifestValidationError("Field 'image_id' must be a non-empty string.")

        if not self.file_path or not isinstance(self.file_path, str):
            raise ManifestValidationError("Field 'file_path' must be a non-empty string.")

        if self.crop not in ALLOWED_CROPS and self.ground_truth_class not in GATE_OOD_CLASSES:
            raise ManifestValidationError(
                f"Invalid crop '{self.crop}' for image '{self.image_id}'. Allowed crops: {ALLOWED_CROPS}."
            )

        # Target Class & Crop Alignment
        if self.ground_truth_class in RAW_CLASS_TO_CROP:
            expected_crop = RAW_CLASS_TO_CROP[self.ground_truth_class]
            if self.crop != expected_crop:
                raise ManifestValidationError(
                    f"Crop mismatch for image '{self.image_id}': declared class '{self.ground_truth_class}' "
                    f"does not belong to crop '{self.crop}' (expected '{expected_crop}')."
                )
        elif self.ground_truth_class in GATE_OOD_CLASSES:
            pass
        else:
            valid_targets = ", ".join(sorted(RAW_CLASS_TO_CROP.keys()))
            raise ManifestValidationError(
                f"Invalid ground_truth_class '{self.ground_truth_class}' for image '{self.image_id}'. "
                f"Must be one of: {valid_targets} or gate classes {GATE_OOD_CLASSES}."
            )

        # Split validation
        if self.split not in ALLOWED_SPLITS:
            raise ManifestValidationError(
                f"Invalid split '{self.split}' for image '{self.image_id}'. "
                f"Allowed splits: {ALLOWED_SPLITS}."
            )

        # Verification status
        if self.verification_status not in ALLOWED_VERIFICATION_STATUS:
            raise ManifestValidationError(
                f"Invalid verification_status '{self.verification_status}' for image '{self.image_id}'. "
                f"Allowed statuses: {ALLOWED_VERIFICATION_STATUS}."
            )

        # Environmental Metadata Validation
        if self.lighting_condition not in ALLOWED_LIGHTING_CONDITIONS:
            raise ManifestValidationError(
                f"Invalid lighting_condition '{self.lighting_condition}' for image '{self.image_id}'. "
                f"Allowed: {ALLOWED_LIGHTING_CONDITIONS}."
            )

        if self.background_type not in ALLOWED_BACKGROUND_TYPES:
            raise ManifestValidationError(
                f"Invalid background_type '{self.background_type}' for image '{self.image_id}'. "
                f"Allowed: {ALLOWED_BACKGROUND_TYPES}."
            )

        if self.disease_severity not in ALLOWED_DISEASE_SEVERITIES:
            raise ManifestValidationError(
                f"Invalid disease_severity '{self.disease_severity}' for image '{self.image_id}'. "
                f"Allowed: {ALLOWED_DISEASE_SEVERITIES}."
            )

        # Privacy Protection: Disallow GPS coordinates or precise lat/long patterns
        if self.collection_location:
            gps_pattern = re.compile(
                r"(\bGPS\b|\d{1,3}\.\d{3,}[°\s]*[NSEW]?\s*,\s*[-+]?\d{1,3}\.\d{3,}[°\s]*[NSEW]?)",
                re.IGNORECASE,
            )
            if gps_pattern.search(self.collection_location):
                raise ManifestValidationError(
                    f"Privacy violation in image '{self.image_id}': collection_location "
                    f"'{self.collection_location}' contains precise GPS coordinates. "
                    f"Use regional or district codes only (e.g. US_CA_CentralValley)."
                )

        # Date format check (YYYY-MM-DD) if provided
        if self.capture_date:
            try:
                date.fromisoformat(self.capture_date)
            except ValueError:
                raise ManifestValidationError(
                    f"Invalid capture_date format '{self.capture_date}' for image '{self.image_id}'. "
                    f"Must be ISO format YYYY-MM-DD."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FieldSampleRecord:
        """Constructs and validates a record from dictionary."""
        known_keys = {
            "image_id", "file_path", "crop", "ground_truth_class",
            "collection_source", "split", "capture_device", "capture_date",
            "collection_location", "lighting_condition", "background_type",
            "disease_severity", "annotator_status", "verification_status",
            "duplicate_group_id", "notes", "sha256", "dhash",
        }
        filtered = {k: v for k, v in data.items() if k in known_keys and v is not None}
        record = cls(**filtered)
        record.validate()
        return record


def load_and_validate_manifest(
    manifest_path: Union[Path, str],
) -> Tuple[List[FieldSampleRecord], List[str]]:
    """
    Loads, parses, and validates all records from a CSV or JSON field manifest.
    Returns:
        (valid_records_list, validation_error_strings_list)
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    records: List[FieldSampleRecord] = []
    errors: List[str] = []

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict) and "records" in raw_data:
                raw_data = raw_data["records"]
            for idx, item in enumerate(raw_data, start=1):
                try:
                    records.append(FieldSampleRecord.from_dict(item))
                except Exception as exc:
                    errors.append(f"Row {idx}: {str(exc)}")
    else:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):
                try:
                    records.append(FieldSampleRecord.from_dict(row))
                except Exception as exc:
                    errors.append(f"Row {idx}: {str(exc)}")

    return records, errors
