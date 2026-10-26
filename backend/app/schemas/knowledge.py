"""
Pydantic schemas for the crop disease knowledge base.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Authoritative scientific or extension source citation."""
    organization: str = Field(..., description="Publishing university, extension service, or agricultural authority")
    title: str = Field(..., description="Title of the cited reference, guide, or bulletin")
    url: str = Field(..., description="Direct link or landing page URL for the source")
    claims_supported: str = Field(..., description="Specific agronomic or pathological claims substantiated by this source")
    reference_url_or_publication: str = Field(..., description="Permanent URL, bulletin code, or publication citation")


class SymptomsInfo(BaseModel):
    """Visual symptom descriptions, foliar signs, and look-alike conditions."""
    summary: str = Field(..., description="Concise summary of characteristic disease symptoms")
    foliar_symptoms: List[str] = Field(..., description="Specific leaf manifestations and lesion patterns")
    stem_and_fruit_symptoms: List[str] = Field(..., description="Stem, petiole, and fruit symptoms if present")
    look_alike_conditions: List[str] = Field(..., description="Visually similar conditions or differential diagnoses")


class FavorableConditionsInfo(BaseModel):
    """Environmental parameters and disease transmission biology."""
    environmental_factors: str = Field(..., description="Temperature, humidity, and leaf wetness favoring disease onset")
    transmission_routes: List[str] = Field(..., description="Vectors, seed transmission, water splash, and survival mechanisms")


class PreventionAndCulturalInfo(BaseModel):
    """Non-chemical cultural management, sanitation, and prevention practices."""
    sanitation: List[str] = Field(..., description="Field hygiene, debris removal, weed management, and equipment disinfection")
    crop_rotation: str = Field(..., description="Crop rotation duration and non-host crop recommendations")
    irrigation_and_airflow: List[str] = Field(..., description="Canopy airflow, plant spacing, and irrigation management")
    resistant_cultivars: List[str] = Field(..., description="Genetic resistance and cultivar selection advice")


class TreatmentInfo(BaseModel):
    """Integrated pest and disease management treatment principles."""
    urgency: str = Field(..., description="Action threshold and urgency level for intervention")
    non_chemical_controls: List[str] = Field(..., description="Biological, mechanical, and cultural control options")
    ipm_chemical_principles: List[str] = Field(..., description="General chemical management principles and resistance mitigation")
    safety_and_regulatory_constraints: List[str] = Field(..., description="Regulatory adherence, label compliance, and diagnostic confirmation requirements")


class ExtensionConsultationInfo(BaseModel):
    """Guidelines on when and who to contact for field confirmation."""
    escalation_triggers: List[str] = Field(..., description="Conditions requiring physical expert inspection or laboratory testing")
    local_specialist: str = Field(..., description="Relevant local agricultural extension contact or pathology service")


class DiseaseKnowledgeRecord(BaseModel):
    """Complete agronomic and pathological knowledge record for a crop condition."""
    class_id: int = Field(..., ge=0, le=14, description="Authoritative class index (0-14)")
    raw_class: str = Field(..., description="Exact dataset class name (e.g., Potato___Early_blight)")
    crop: str = Field(..., description="Human-readable common crop name (e.g., Potato)")
    condition: str = Field(..., description="Human-readable condition or disease name")
    health_status: str = Field(..., description="Health category: 'Healthy' or 'Diseased'")
    causal_agent: Optional[str] = Field(None, description="Scientific binomial name of the pathogen, or null for healthy crops")
    symptoms: SymptomsInfo = Field(..., description="Diagnostic visual symptoms and look-alikes")
    favorable_conditions: FavorableConditionsInfo = Field(..., description="Environmental drivers and transmission vectors")
    prevention_and_cultural: PreventionAndCulturalInfo = Field(..., description="Cultural management, rotation, and sanitation")
    treatment: TreatmentInfo = Field(..., description="IPM treatment principles and safety constraints")
    extension_consultation: ExtensionConsultationInfo = Field(..., description="Escalation triggers and specialist consultation guidance")
    sources: List[SourceReference] = Field(..., min_length=1, description="Citations and evidence sources supporting this entry")
    safety_and_uncertainty_notes: str = Field(..., description="Diagnostic uncertainty disclaimer and field verification guidelines")
