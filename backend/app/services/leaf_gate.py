"""
Leaf-acceptance gate model service.

WHAT IT IS
----------
A 3-class classifier (`non_leaf`, `other_plant`, `supported_leaf`) whose inputs are the same
224x224 RGB frames the disease model receives. It is a linear head on frozen ImageNet
MobileNetV2 features, trained by ``ml/scripts/train_input_gate.py`` and exported as a single
self-contained Keras model, so serving needs no second download and no extra dependency.

WHO DECIDES
-----------
``backend.app.services.input_gate.assess_image`` consumes this service's probabilities. The
gate is the *content* authority when it is loaded; ``input_gate`` keeps its deterministic
plant-pigmentation screen as a fallback for the case where the model file is missing or
disabled, so the product never silently loses its input screening.

CONCURRENCY
-----------
Keras models are not guaranteed to be reentrant across threads. Inference is serialised by a
lock, exactly like ``ModelService``, and callers already run it inside ``asyncio.to_thread``
so the event loop is never blocked.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ENABLE_LEAF_GATE, LEAF_GATE_MODEL_PATH

logger = logging.getLogger(__name__)

NON_LEAF = "non_leaf"
OTHER_PLANT = "other_plant"
SUPPORTED_LEAF = "supported_leaf"

# sklearn orders classes alphabetically; used only when the sidecar config is absent.
DEFAULT_CLASSES = [NON_LEAF, OTHER_PLANT, SUPPORTED_LEAF]


class LeafGateService:
    """Thread-safe wrapper around the exported leaf-acceptance model."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path: Path = model_path or LEAF_GATE_MODEL_PATH
        self._model: Any = None
        self._classes: list[str] = list(DEFAULT_CLASSES)
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def load_model(self, path: Optional[Path] = None) -> None:
        """Loads the gate artefact once during application startup."""
        target = path or self._model_path
        if not ENABLE_LEAF_GATE:
            logger.info("Leaf gate disabled by configuration; deterministic screen only.")
            return
        if not target.is_file():
            self._load_error = f"Leaf gate model not found at: {target}"
            logger.warning(self._load_error)
            raise FileNotFoundError(self._load_error)

        import keras

        with self._lock:
            try:
                self._model = keras.models.load_model(str(target), compile=False)
                self._model_path = target
                self._classes = self._read_classes(target)
                self._load_error = None
                logger.info("Leaf-acceptance gate loaded (%s).", ", ".join(self._classes))
            except Exception as exc:
                self._model = None
                self._load_error = f"Failed to load leaf gate: {exc}"
                logger.error(self._load_error, exc_info=True)
                raise RuntimeError(self._load_error) from exc

    @staticmethod
    def _read_classes(model_path: Path) -> list[str]:
        """Reads the trained class order from the sidecar config written at export time."""
        sidecar = model_path.parent / "gate_model_config.json"
        try:
            classes = json.loads(sidecar.read_text(encoding="utf-8"))["classes"]
            if isinstance(classes, list) and len(classes) == 3 and all(isinstance(c, str) for c in classes):
                return classes
        except Exception:
            pass
        return list(DEFAULT_CLASSES)

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            self._load_error = None
            logger.info("Leaf-acceptance gate unloaded.")

    def set_model(self, model: Any, classes: Optional[list[str]] = None) -> None:
        """Injects a model (used by tests)."""
        with self._lock:
            self._model = model
            self._classes = list(classes or DEFAULT_CLASSES)
            self._load_error = None

    def predict_proba(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Returns class probabilities for one (224, 224, 3) RGB frame in [0, 255].

        Returns None when the gate is unavailable, which makes the caller fall back to the
        deterministic input screen rather than skipping input validation entirely.
        """
        model = self._model
        if model is None or not ENABLE_LEAF_GATE:
            return None

        array = np.asarray(frame, dtype=np.float32)
        if array.shape != (224, 224, 3):
            raise ValueError(f"Expected a (224, 224, 3) frame, got {array.shape}.")

        with self._lock:
            output = model.predict(np.expand_dims(array, axis=0), verbose=0)

        scores = np.asarray(output, dtype=np.float64).reshape(-1)
        if len(scores) != len(self._classes):
            raise ValueError(
                f"Leaf gate output dimension mismatch: expected {len(self._classes)}, got {len(scores)}."
            )
        return {name: float(value) for name, value in zip(self._classes, scores)}


leaf_gate_service = LeafGateService()
