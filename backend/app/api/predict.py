"""
Prediction endpoint accepting leaf image uploads.
"""
import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import (
    ALLOWED_MIME_TYPES,
    CHUNK_SIZE_BYTES,
    MAX_DECODE_CONCURRENCY,
    MAX_DECODE_QUEUE_SIZE,
    MAX_UPLOAD_SIZE_BYTES,
)
from ..schemas.prediction import PredictionResponse
from ..services.inference import (
    ImageValidationError,
    model_service,
    preprocess_image,
    sniff_image_format,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

# Concurrency & admission primitives (lazily bound to the active event loop)
_decode_semaphore: Optional[asyncio.Semaphore] = None
_decode_semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

_inference_semaphore: Optional[asyncio.Semaphore] = None
_inference_semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

_decode_waiters: int = 0
_decode_waiters_lock = threading.Lock()


def get_decode_semaphore() -> asyncio.Semaphore:
    """Returns decode semaphore tied to the running event loop."""
    global _decode_semaphore, _decode_semaphore_loop
    current_loop = asyncio.get_running_loop()
    if _decode_semaphore is None or _decode_semaphore_loop != current_loop:
        _decode_semaphore = asyncio.Semaphore(MAX_DECODE_CONCURRENCY)
        _decode_semaphore_loop = current_loop
    return _decode_semaphore


def get_inference_semaphore() -> asyncio.Semaphore:
    """Returns inference semaphore tied to the running event loop."""
    global _inference_semaphore, _inference_semaphore_loop
    current_loop = asyncio.get_running_loop()
    if _inference_semaphore is None or _inference_semaphore_loop != current_loop:
        _inference_semaphore = asyncio.Semaphore(1)
        _inference_semaphore_loop = current_loop
    return _inference_semaphore


@router.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Prediction computed successfully."},
        400: {"description": "Invalid input, empty upload, or corrupt/unreadable image."},
        413: {"description": "Payload too large (exceeds 10 MB limit)."},
        415: {"description": "Unsupported media type or declared MIME format mismatch."},
        422: {"description": "Image dimensions sub-minimum (<64px), pixel limit exceeded (>20MP), or uniform blank canvas."},
        429: {"description": "Server is handling peak request volume (overload rejection)."},
        503: {"description": "Model is not loaded or inference engine unavailable."},
    },
    summary="Predict Crop Disease from Leaf Image",
    description=(
        "Accepts a single Solanaceae leaf image upload, validates its structure, "
        "preprocesses it to 224x224 RGB, and returns top-3 candidate predictions "
        "along with softmax scores across all 15 trained classes."
    ),
)
async def predict(
    file: Optional[UploadFile] = File(None, description="Leaf image file (JPEG, PNG, WEBP)"),
    image: Optional[UploadFile] = File(None, description="Alternative field name for leaf image file"),
) -> PredictionResponse:
    # 1. Validate that an upload field was provided
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image uploaded. Provide a file in multipart form field 'file' or 'image'.",
        )

    # 2. Check model readiness before consuming upload stream
    if not model_service.is_loaded:
        if model_service.load_error:
            logger.error("Inference requested while model is unready: %s", model_service.load_error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service is currently unavailable.",
        )

    # 3. Preliminary MIME type check on client-declared header
    if upload.content_type:
        content_type_clean = upload.content_type.split(";")[0].strip().lower()
        if content_type_clean not in ALLOWED_MIME_TYPES:
            supported = ", ".join(sorted(ALLOWED_MIME_TYPES))
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media type '{upload.content_type}'. Supported MIME types: {supported}.",
            )

    # 4. Stream upload in bounded chunks to strictly enforce 10 MB limit without unbounded memory consumption
    chunks: List[bytes] = []
    total_bytes = 0

    try:
        while True:
            chunk = await upload.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Uploaded file exceeds maximum permitted size of {max_mb} MB.",
                )
            chunks.append(chunk)
    finally:
        await upload.close()

    # 5. Enforce non-empty upload
    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    image_bytes = b"".join(chunks)

    # 6. Check declared MIME against sniffed binary magic bytes if header was specified
    if upload.content_type:
        content_type_clean = upload.content_type.split(";")[0].strip().lower()
        if content_type_clean in MIME_TO_FORMAT:
            expected_fmt = MIME_TO_FORMAT[content_type_clean]
            sniffed_fmt = sniff_image_format(image_bytes)
            if sniffed_fmt != expected_fmt:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"Declared MIME type '{upload.content_type}' does not match "
                        f"detected file signature '{sniffed_fmt or 'unknown'}'."
                    ),
                )

    # 7. Bounded admission & CPU worker thread offloading for image preprocessing
    global _decode_waiters
    with _decode_waiters_lock:
        if _decode_waiters >= MAX_DECODE_QUEUE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is currently handling peak request volume. Please retry your request shortly.",
            )
        _decode_waiters += 1

    try:
        async with get_decode_semaphore():
            input_tensor = await asyncio.to_thread(preprocess_image, image_bytes)
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        with _decode_waiters_lock:
            _decode_waiters -= 1

    # 8. Execute inference and construct structured response via worker thread offloading
    try:
        async with get_inference_semaphore():
            result = await asyncio.to_thread(model_service.predict, input_tensor)
        return PredictionResponse(**result)
    except RuntimeError as exc:
        logger.error("Model runtime failure during inference: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service is currently unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected failure during prediction pipeline: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during prediction processing.",
        ) from exc
