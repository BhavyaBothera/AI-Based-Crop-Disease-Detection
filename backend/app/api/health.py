"""
Health and readiness probe endpoints.
"""
import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..config import API_TITLE, API_VERSION
from ..schemas.prediction import HealthResponse, ReadyResponse
from ..services.inference import model_service
from ..services.knowledge import knowledge_service
from ..services.leaf_gate import leaf_gate_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process Health Check",
    description="Returns basic operational health status of the running FastAPI process.",
)
async def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=API_TITLE,
        version=API_VERSION,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        200: {"description": "Model and knowledge base are loaded and ready for requests."},
        503: {"description": "Model or knowledge base is not loaded or initialization failed."},
    },
    summary="System Readiness Probe",
    description="Reports whether both the trained inference model and knowledge base are loaded and ready for requests.",
)
async def get_ready():
    if model_service.is_loaded and knowledge_service.is_loaded:
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            knowledge_loaded=True,
            gate_loaded=leaf_gate_service.is_loaded,
            model_path=model_service.safe_model_identifier,
        )

    # Server-side diagnostic log without leaking host paths to client
    if not model_service.is_loaded and model_service.load_error:
        logger.error("Model readiness check failed: %s", model_service.load_error)
    if not knowledge_service.is_loaded and knowledge_service.load_error:
        logger.error("Knowledge base readiness check failed: %s", knowledge_service.load_error)

    if not model_service.is_loaded and not knowledge_service.is_loaded:
        detail = "Model inference and knowledge base services are not ready."
    elif not model_service.is_loaded:
        detail = "Model inference service is not ready."
    else:
        detail = "Knowledge base service is not ready."

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "model_loaded": model_service.is_loaded,
            "knowledge_loaded": knowledge_service.is_loaded,
            "gate_loaded": leaf_gate_service.is_loaded,
            "model_path": model_service.safe_model_identifier,
            "detail": detail,
        },
    )
