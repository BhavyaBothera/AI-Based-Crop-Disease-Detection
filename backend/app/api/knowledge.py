"""
Knowledge base API router providing agronomic and disease information.
"""
import logging

from fastapi import APIRouter, HTTPException, Path, status

from ..schemas.knowledge import DiseaseKnowledgeRecord
from ..services.knowledge import knowledge_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Knowledge Base"])


@router.get(
    "/api/knowledge/{class_id}",
    response_model=DiseaseKnowledgeRecord,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Disease knowledge record retrieved successfully."},
        404: {"description": "Unknown class ID (valid range: 0 through 14)."},
        422: {"description": "Validation error (non-integer class ID format)."},
        503: {"description": "Knowledge base service is uninitialized or unavailable."},
    },
    summary="Get Disease Knowledge by Class ID",
    description=(
        "Retrieves validated agronomic information, diagnostic symptoms, prevention, "
        "IPM treatment principles, extension escalation guidelines, and source references "
        "for a specific model class ID (0-14)."
    ),
)
async def get_disease_knowledge(
    class_id: int = Path(..., description="Authoritative class index (0 through 14)"),
) -> DiseaseKnowledgeRecord:
    # 1. Check knowledge service availability
    if not knowledge_service.is_loaded:
        if knowledge_service.load_error:
            logger.error("Knowledge base requested while unready: %s", knowledge_service.load_error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base service is currently unavailable.",
        )

    # 2. Check class_id bounds and existence (return 404 for unknown classes)
    if class_id < 0 or class_id > 14 or not knowledge_service.has_class(class_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease knowledge record not found for class_id {class_id}.",
        )

    # 3. Retrieve record defensively
    try:
        record = knowledge_service.get_disease_record(class_id)
    except Exception as exc:
        logger.error(
            "Unexpected error retrieving knowledge record for class_id %s: %s",
            class_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while retrieving disease knowledge.",
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease knowledge record not found for class_id {class_id}.",
        )

    return DiseaseKnowledgeRecord(**record)
