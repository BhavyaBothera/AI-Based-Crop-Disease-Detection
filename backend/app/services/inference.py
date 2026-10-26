"""
Inference service for crop disease classification.

Handles model lifecycle management, safe Pillow image decoding,
RGB conversion, resizing to 224x224, raw [0, 255] float32 array creation,
and output formatting with the authoritative 15-class mapping.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import logging
import threading
import warnings

import numpy as np
from PIL import Image, ImageOps

from ..config import (
    ALLOWED_PIL_FORMATS,
    DEFAULT_MODEL_REL_PATH,
    DISCLAIMER_TEXT,
    ENABLE_LEVEL_A_PREVALIDATION,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_DIMENSION,
    MODEL_PATH,
    PROJECT_ROOT,
    TARGET_IMAGE_SIZE,
    UNIFORM_CANVAS_STD_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Configure Pillow's decompression-bomb protection deliberately (never disabled).
# Set threshold so warnings escalate at MAX_IMAGE_PIXELS and errors at 2 * MAX_IMAGE_PIXELS.
Image.MAX_IMAGE_PIXELS = 2 * MAX_IMAGE_PIXELS

# Authoritative 15-class Solanaceae mapping.
# Must strictly match ml/scripts/train_model.py TARGET_CLASS_MAPPING and dataset manifest.
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

CLASS_ID_TO_METADATA: Dict[int, Dict[str, Any]] = {
    v["id"]: {"raw_class": k, **v} for k, v in TARGET_CLASS_MAPPING.items()
}


class ImageValidationError(ValueError):
    """Base exception for all image validation and integrity failures."""
    status_code: int = 400


class EmptyImageError(ImageValidationError):
    """Raised when uploaded image payload is empty (0 bytes)."""
    status_code: int = 400


class CorruptImageError(ImageValidationError):
    """Raised when image container is truncated, malformed, or unreadable."""
    status_code: int = 400


class UnsupportedFormatError(ImageValidationError):
    """Raised when image format or declared MIME type is not supported."""
    status_code: int = 415


class InvalidImageDimensionsError(ImageValidationError):
    """Raised when image dimensions are below minimum floor or exceed pixel limit."""
    status_code: int = 422


class UniformCanvasError(ImageValidationError):
    """Raised when image appears to be a blank or solid-color canvas."""
    status_code: int = 422


def sniff_image_format(image_bytes: bytes) -> Optional[str]:
    """
    Sniffs binary magic bytes from the start of the image payload.
    Returns 'JPEG', 'PNG', 'WEBP', or None if unsupported or too short.
    """
    if len(image_bytes) < 12:
        return None
    if image_bytes.startswith(b"\xFF\xD8\xFF"):
        return "JPEG"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "WEBP"
    return None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes an image from bytes, verifies format, checks pixel dimensions,
    converts to RGB, resizes to 224x224, and returns a float32 array in [0.0, 255.0].

    SECURITY & DECOMPRESSION SAFEGUARDS:
    - Fast magic-byte sniffing pre-filter for JPEG, PNG, and WebP signatures.
    - Scoped context escalates Image.DecompressionBombWarning into an exception.
    - Explicitly inspects (width * height) BEFORE raster decompression to reject
      oversized images (> MAX_IMAGE_PIXELS) and prevent memory exhaustion.
    - Handles EXIF orientation transposition prior to dimension verification.
    - Enforces 64x64 minimum dimension floor (operational sanity floor).
    - Normalizes color spaces (Palette, Grayscale, CMYK, RGBA) safely to 3-channel RGB.
    - Explicit raster load to catch corrupted scanlines and truncated files.
    - Computes population standard deviation on 224x224 luminance to reject uniform canvases.

    CRITICAL PREPROCESSING NOTE:
    The saved Keras model includes an internal `layers.Rescaling(scale=1.0/127.5, offset=-1.0)`
    as Layer 1. Therefore, this function strictly outputs raw RGB pixel values
    in [0.0, 255.0]. Normalizing here to [0, 1] or [-1, 1] would double-rescale
    and corrupt predictions.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        np.ndarray: Batch tensor of shape (1, 224, 224, 3) and dtype float32.

    Raises:
        EmptyImageError: If file is empty (0 bytes) [HTTP 400].
        UnsupportedFormatError: If magic bytes or format are unsupported [HTTP 415].
        CorruptImageError: If container or scanlines are corrupt/truncated [HTTP 400].
        InvalidImageDimensionsError: If dimensions < 64x64 or > 20 MP [HTTP 422].
        UniformCanvasError: If canvas standard deviation <= threshold [HTTP 422].
    """
    if not image_bytes:
        raise EmptyImageError("Uploaded file is empty (0 bytes).")

    # Layer 1: Fast binary magic-byte signature pre-filter
    sniffed_fmt = sniff_image_format(image_bytes)
    if sniffed_fmt is None or sniffed_fmt not in ALLOWED_PIL_FORMATS:
        supported = ", ".join(sorted(ALLOWED_PIL_FORMATS))
        raise UnsupportedFormatError(
            f"Unsupported image format. Supported formats: {supported}."
        )

    with warnings.catch_warnings():
        # Escalate Pillow decompression-bomb warnings into exceptions within this scoped context
        warnings.filterwarnings("error", category=Image.DecompressionBombWarning)

        try:
            # Step 1: Open header and verify container format matches sniffed signature
            buf = io.BytesIO(image_bytes)
            img = Image.open(buf)
            detected_format = img.format
            if detected_format not in ALLOWED_PIL_FORMATS or detected_format != sniffed_fmt:
                supported = ", ".join(sorted(ALLOWED_PIL_FORMATS))
                raise UnsupportedFormatError(
                    f"Unsupported or mismatched image format '{detected_format}'. Supported formats: {supported}."
                )

            # Step 2: Enforce pixel ceiling BEFORE raster loading / decompression
            width, height = img.size
            num_pixels = width * height
            if num_pixels > MAX_IMAGE_PIXELS:
                raise InvalidImageDimensionsError(
                    f"Image resolution ({width}x{height} = {num_pixels:,} pixels) "
                    f"exceeds maximum permitted limit of {MAX_IMAGE_PIXELS:,} pixels."
                )

            # Step 3: Structural container verification
            img.verify()

            # Step 4: Reopen for decoding (verify invalidates internal file pointers)
            buf.seek(0)
            img = Image.open(buf)

            # Step 5: EXIF orientation handling (ensures correct spatial dimensions)
            img = ImageOps.exif_transpose(img)

            # Re-verify dimensions on transposed instance before raster conversion
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageDimensionsError(
                    f"Image resolution ({width}x{height} = {width * height:,} pixels) "
                    f"exceeds maximum permitted limit of {MAX_IMAGE_PIXELS:,} pixels."
                )

            # Step 6: Enforce minimum dimension floor if Level A is enabled
            if ENABLE_LEVEL_A_PREVALIDATION:
                if min(width, height) < MIN_IMAGE_DIMENSION:
                    raise InvalidImageDimensionsError(
                        f"Image resolution ({width}x{height}) is too small. "
                        f"Minimum required resolution is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
                    )

            # Step 7: Convert to 3-channel RGB (safely handles RGBA, LA, Palette, Grayscale, CMYK)
            # For transparent inputs, convert("RGB") fills alpha background with black (matching PlantVillage)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Step 8: Explicit raster load to catch corrupted scanlines and truncated files
            img.load()

            # Step 9: Bilinear resize to 224x224 (matches tf.image.resize training pipeline)
            img_resized = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)

            # Step 10: Convert to float32 NumPy array in [0.0, 255.0] range
            arr_rgb = np.asarray(img_resized, dtype=np.float32)

            # Step 11: Uniform / blank canvas detection on 224x224 luminance
            if ENABLE_LEVEL_A_PREVALIDATION:
                # ITU-R BT.601 luminance formula
                luminance = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
                sigma = float(np.std(luminance))
                if sigma <= UNIFORM_CANVAS_STD_THRESHOLD:
                    raise UniformCanvasError(
                        "Uploaded image appears to be a blank or solid-color canvas. "
                        "Please provide a clear photograph of a crop leaf."
                    )

            # Shape: (1, 224, 224, 3)
            return np.expand_dims(arr_rgb, axis=0)

        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageDimensionsError(
                f"Image exceeds safe decompression limits: {str(exc)}"
            ) from exc
        except ImageValidationError:
            raise
        except (OSError, Image.UnidentifiedImageError) as exc:
            raise CorruptImageError(
                f"Corrupt or unreadable image file: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise CorruptImageError(
                f"Corrupt or unreadable image file: {str(exc)}"
            ) from exc


class ModelService:
    """
    Thread-safe model wrapper managing model lifecycle and inference.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path: Path = model_path or MODEL_PATH
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model_path(self) -> Path:
        """Returns the configured model filesystem path (internal server use only)."""
        return self._model_path

    @property
    def safe_model_identifier(self) -> str:
        """
        Returns a sanitized model identifier (relative project path or filename)
        suitable for public API responses, never leaking host filesystem paths.
        """
        if self._model_path is None:
            return DEFAULT_MODEL_REL_PATH
        try:
            return self._model_path.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            return self._model_path.name

    @property
    def load_error(self) -> Optional[str]:
        """Returns the last loading error message for internal server logging."""
        return self._load_error

    def load_model(self, path: Optional[Path] = None) -> None:
        """
        Loads the Keras model from disk once during application startup.

        Args:
            path: Optional override path to the .keras model file.
        """
        target_path = path or self._model_path
        logger.info(f"Loading inference model from {target_path}...")

        if not target_path.is_file():
            self._load_error = f"Model file not found at: {target_path}"
            logger.error(self._load_error)
            raise FileNotFoundError(self._load_error)

        with self._lock:
            try:
                import keras
                # Load with compile=False for faster loading (inference only)
                self._model = keras.models.load_model(str(target_path), compile=False)
                self._model_path = target_path
                self._load_error = None
                logger.info("Inference model loaded successfully.")
            except Exception as exc:
                self._model = None
                self._load_error = f"Failed to load model: {str(exc)}"
                logger.error(self._load_error, exc_info=True)
                raise RuntimeError(self._load_error) from exc

    def unload_model(self) -> None:
        """Unloads the model and releases memory during shutdown."""
        with self._lock:
            self._model = None
            self._load_error = None
            logger.info("Model unloaded.")

    def set_model(self, model: Any) -> None:
        """Injects a model (used for mocking in lightweight tests)."""
        with self._lock:
            self._model = model
            self._load_error = None

    def predict(self, input_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Runs model inference on the preprocessed image tensor.

        Thread-safe: synchronized via self._lock to prevent Keras/TensorFlow
        graph retracing and state collisions under concurrent execution.

        Args:
            input_tensor: NumPy array of shape (1, 224, 224, 3) in [0.0, 255.0].

        Returns:
            Dict[str, Any]: Dictionary matching PredictionResponse schema.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        if input_tensor.shape != (1, 224, 224, 3):
            raise ValueError(f"Expected input shape (1, 224, 224, 3), got {input_tensor.shape}")

        with self._lock:
            # Model inference (returns shape (1, 15))
            raw_preds = self._model.predict(input_tensor, verbose=0)
            scores = raw_preds[0]

        if len(scores) != len(TARGET_CLASS_MAPPING):
            raise ValueError(
                f"Model output dimension mismatch: expected {len(TARGET_CLASS_MAPPING)}, got {len(scores)}"
            )

        # Sort descending by softmax score
        top_indices = np.argsort(scores)[::-1]
        predicted_class_id = int(top_indices[0])
        pred_meta = CLASS_ID_TO_METADATA[predicted_class_id]

        # Top 3 class predictions
        top_predictions: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices[:3], start=1):
            item_meta = CLASS_ID_TO_METADATA[int(idx)]
            top_predictions.append({
                "rank": rank,
                "class_id": int(idx),
                "raw_class": item_meta["raw_class"],
                "crop": item_meta["crop"],
                "condition": item_meta["condition"],
                "health": item_meta["health"],
                "score": float(scores[idx]),
            })

        # Complete score distribution across all 15 classes
        all_scores: Dict[str, float] = {
            CLASS_ID_TO_METADATA[i]["raw_class"]: float(scores[i])
            for i in range(len(scores))
        }

        return {
            "predicted_class_id": predicted_class_id,
            "raw_class": pred_meta["raw_class"],
            "crop": pred_meta["crop"],
            "condition": pred_meta["condition"],
            "health": pred_meta["health"],
            "score": float(scores[predicted_class_id]),
            "top_predictions": top_predictions,
            "all_scores": all_scores,
            "disclaimer": DISCLAIMER_TEXT,
        }


# Global singleton instance for the FastAPI application lifespan
model_service = ModelService()
