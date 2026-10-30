"""
Integration tests for the real exported leaf-acceptance gate.

These load ``ml/models/input_gate/leaf_gate.keras`` and real photographs, so they verify the
artefact that ships - not a mock. They are skipped when the trained artefact or the local
(git-ignored) data are absent, because both are produced by the ML pipeline rather than
committed:

    .venv/Scripts/python.exe ml/scripts/build_ood_corpus.py
    .venv/Scripts/python.exe ml/scripts/train_input_gate.py

Statistical accuracy is NOT asserted here; ``ml/scripts/evaluate_input_gate.py`` measures it
on the held-out corpus and writes docs/audit_reports/non_leaf_input_safety_eval.json.
"""
import csv
import hashlib
import unittest

from backend.app.config import LEAF_GATE_MODEL_PATH, PROJECT_ROOT
from backend.app.services.input_gate import (
    ACCEPTED,
    CODE_UNSUPPORTED_SPECIES,
    assess_image,
)
from backend.app.services.inference import preprocess_image
from backend.app.services.leaf_gate import LeafGateService

MANIFEST = PROJECT_ROOT / "ml" / "data" / "dataset_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "ml"
OOD_ROOT = PROJECT_ROOT / "ml" / "data" / "ood_corpus"

PORTRAITS = [
    "person/elderly-gambian-woman-face-portrait.jpg",
    "person/laughing-woman-with-teeth.jpg",
    "person/woman-portrait-fortepan-78431.jpg",
]
UNSUPPORTED_LEAVES = [
    "data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Apple___Apple_scab/image (1).JPG",
    "data/raw_dataset/Plant_leave_diseases_dataset_without_augmentation/Corn___Common_rust/image (1).JPG",
]

ARTEFACTS_READY = LEAF_GATE_MODEL_PATH.is_file() and (
    (OOD_ROOT / PORTRAITS[0]).is_file() and (DATA_ROOT / UNSUPPORTED_LEAVES[0]).is_file()
)


def bucket(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:2], "big") % 100


def supported_eval_leaves(limit: int = 3) -> list[str]:
    """Real supported leaves from the gate's evaluation half (never trained on)."""
    with open(MANIFEST, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    leaves = [
        row["relative_path"].replace("\\", "/")
        for row in rows
        if row["split"] == "val" and bucket(row["relative_path"]) >= 50
    ]
    return leaves[:limit]


@unittest.skipUnless(ARTEFACTS_READY, "Gate artefact or local corpus not available")
class TestRealGateArtefact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = LeafGateService()
        cls.gate.load_model(LEAF_GATE_MODEL_PATH)

    def assess(self, path):
        frame = preprocess_image(path.read_bytes())[0]
        return assess_image(frame, self.gate.predict_proba(frame))

    def test_real_portraits_are_not_accepted(self):
        for relative in PORTRAITS:
            with self.subTest(image=relative):
                result = self.assess(OOD_ROOT / relative)
                self.assertNotEqual(result.decision, ACCEPTED)
                self.assertGreaterEqual(result.signals["gate_probabilities"]["non_leaf"], 0.5)

    def test_real_supported_leaves_are_accepted(self):
        for relative in supported_eval_leaves():
            with self.subTest(image=relative):
                result = self.assess(DATA_ROOT / relative)
                self.assertEqual(result.decision, ACCEPTED, result.signals)

    def test_unsupported_species_leaf_is_not_diagnosed_as_a_supported_crop(self):
        for relative in UNSUPPORTED_LEAVES:
            with self.subTest(image=relative):
                result = self.assess(DATA_ROOT / relative)
                self.assertNotEqual(result.decision, ACCEPTED, result.signals)
                self.assertIn(result.code, (CODE_UNSUPPORTED_SPECIES, "AMBIGUOUS_IMAGE"))

    def test_load_error_is_reported_for_a_missing_artefact(self):
        service = LeafGateService()
        with self.assertRaises(FileNotFoundError):
            service.load_model(PROJECT_ROOT / "ml" / "models" / "input_gate" / "does_not_exist.keras")
        self.assertFalse(service.is_loaded)


if __name__ == "__main__":
    unittest.main()
