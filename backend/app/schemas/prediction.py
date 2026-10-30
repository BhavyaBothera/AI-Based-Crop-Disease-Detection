"""
Pydantic schemas for prediction outputs and health checks.
"""
from typing import Dict, List
from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    """Prediction detail for a single candidate class."""
    rank: int = Field(..., description="Rank in prediction order (1 = highest score)")
    class_id: int = Field(..., ge=0, le=14, description="Authoritative class index (0-14)")
    raw_class: str = Field(..., description="Exact dataset class name (e.g., Potato___Early_blight)")
    crop: str = Field(..., description="Human-readable common crop name (e.g., Potato)")
    condition: str = Field(..., description="Human-readable disease condition or Healthy status")
    health: str = Field(..., description="Health classification: 'Healthy' or 'Diseased'")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw model softmax score. Note: This represents model output distribution across the 15 classes, not calibrated probability or diagnostic certainty."
    )


class PredictionResponse(BaseModel):
    """Complete prediction response returned by POST /api/predict."""
    predicted_class_id: int = Field(..., ge=0, le=14, description="Top predicted class ID (0-14)")
    raw_class: str = Field(..., description="Raw dataset identifier for the top predicted class")
    crop: str = Field(..., description="Human-readable crop name for top prediction")
    condition: str = Field(..., description="Human-readable condition/disease for top prediction")
    health: str = Field(..., description="Health status ('Healthy' or 'Diseased')")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Top prediction softmax score output by the model."
    )
    top_predictions: List[ClassPrediction] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Top three predicted classes ranked by model softmax score."
    )
    all_scores: Dict[str, float] = Field(
        ...,
        description="Model softmax scores across all 15 supported Solanaceae classes."
    )
    disclaimer: str = Field(
        ...,
        description="Informational and advisory disclaimer regarding automated ML predictions."
    )


class HealthResponse(BaseModel):
    """Process health status returned by GET /health."""
    status: str = Field(..., description="Operational status of the backend process")
    app: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")


class ReadyResponse(BaseModel):
    """System readiness status returned by GET /ready."""
    status: str = Field(..., description="Readiness status ('ready' or 'not_ready')")
    model_loaded: bool = Field(..., description="Whether the inference model is loaded in memory")
    knowledge_loaded: bool = Field(..., description="Whether the agronomic knowledge base is loaded")
    gate_loaded: bool = Field(
        False,
        description="Whether the trained leaf-acceptance gate is loaded (false means the deterministic input screen is in charge)",
    )
    model_path: str = Field(..., description="Safe relative identifier of the loaded model")
