"""
Content-level input suitability assessment (Level B gate).

PURPOSE
-------
The 15-class disease model is closed-world: it returns a softmax distribution no matter
what it is shown, so its argmax alone must never be treated as evidence that the input
is a crop leaf. This module screens the decoded image before inference and produces one
of three outcomes:

    accepted  - a supported crop leaf was verified in a usable photograph
    abstained - the subject cannot be verified reliably (ambiguous or poor quality)
    rejected  - the image is not a supported crop leaf

TWO SCREENS, ONE DECISION
-------------------------
1. The *trained* leaf-acceptance gate (``backend.app.services.leaf_gate``) supplies class
   probabilities for `supported_leaf` / `other_plant` / `non_leaf`. When it is loaded it is
   the content authority, and the decision comes from its top class plus a confidence floor.
2. The deterministic plant-pigmentation screen below is kept as the FALLBACK used when the
   gate artefact is missing or disabled. It measures plant pigmentation, exposure and
   spatial concentration with fixed thresholds calibrated on the non-test validation split.

In both cases the disease model's own softmax is never consulted: a confidently wrong
softmax on an out-of-distribution image is exactly the failure this gate exists to prevent.

HONEST LIMITS
-------------
The gate separates *plant tissue in a frame* from *no plant tissue*, and separates the 15
trained species from leaves of other species it saw during training. It is not a
species-level botanical classifier: an unfamiliar leaf may be abstained on or classified as
`other_plant`, and the measured residual error rates are published in
``docs/audit_reports/non_leaf_input_safety_report.md`` rather than claimed to be zero.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    ENABLE_INPUT_GATE,
    INPUT_GATE_ACCEPT_AT_PLANT_EVIDENCE,
    INPUT_GATE_MIN_DETAIL_VARIANCE,
    INPUT_GATE_MIN_DOMINANT_REGION,
    INPUT_GATE_MIN_MEAN_LUMINANCE,
    INPUT_GATE_REGION_AREA_FLOOR,
    INPUT_GATE_REJECT_BELOW_PLANT_EVIDENCE,
    INPUT_GATE_SCATTERED_REGION_COUNT,
    LEAF_GATE_MIN_CONFIDENCE,
)

# Class names emitted by the trained gate (see services/leaf_gate.py).
GATE_SUPPORTED_LEAF = "supported_leaf"
GATE_OTHER_PLANT = "other_plant"
GATE_NON_LEAF = "non_leaf"
GATE_CLASSES = (GATE_SUPPORTED_LEAF, GATE_OTHER_PLANT, GATE_NON_LEAF)

# Decision constants (also used as the machine-readable `status` field).
ACCEPTED = "accepted"
ABSTAINED = "abstained"
REJECTED = "rejected"

# Broad plant-pigmentation definition. Leaves, including chlorotic and yellowing tissue,
# fall inside this band; skin, wood, paper, most textiles, concrete and sky do not.
PLANT_HUE_MIN_DEG = 40.0
PLANT_HUE_MAX_DEG = 170.0
PLANT_SATURATION_MIN = 0.12
PLANT_VALUE_MIN = 0.08

# Rule codes returned to clients.
CODE_ACCEPTED = "SUPPORTED_LEAF_INPUT"
CODE_NON_LEAF = "NON_LEAF_IMAGE"
CODE_UNSUPPORTED_SPECIES = "UNSUPPORTED_SPECIES"
CODE_AMBIGUOUS = "AMBIGUOUS_IMAGE"
CODE_SPARSE = "SPARSE_PLANT_EVIDENCE"
CODE_SCATTERED = "SCATTERED_PLANT_EVIDENCE"
CODE_UNDEREXPOSED = "UNDEREXPOSED_IMAGE"
CODE_LOW_DETAIL = "LOW_DETAIL_IMAGE"

HEADLINE = "We couldn't verify that this image contains a supported crop leaf."

_DETAIL_MESSAGES = {
    CODE_NON_LEAF: (
        "No plant pigmentation was detected. This usually means the photograph shows a "
        "person, animal, document, screen, building or another non-plant subject. "
        "PHYTOSIGNAL screens leaves of tomato, potato and bell pepper only."
    ),
    CODE_SPARSE: (
        "Only a small amount of plant material is visible. Move closer and frame a single "
        "leaf so that it fills most of the photograph."
    ),
    CODE_SCATTERED: (
        "Plant material is present, but it is scattered across a wide scene instead of "
        "being framed as one specimen. Move closer so a single leaf dominates the frame."
    ),
    CODE_UNDEREXPOSED: (
        "The photograph is too dark to assess plant tissue reliably. Add light, avoid deep "
        "shadows, and try again."
    ),
    CODE_LOW_DETAIL: (
        "The photograph contains too little visible detail (severe blur or a flat, "
        "featureless surface) to assess plant tissue."
    ),
    CODE_UNSUPPORTED_SPECIES: (
        "This looks like a plant leaf, but not one of the crops PHYTOSIGNAL supports. "
        "Diagnoses are available for tomato, potato and bell pepper only, so no result is "
        "given rather than a guess."
    ),
    CODE_AMBIGUOUS: (
        "The subject of this photograph could not be verified confidently. Frame a single "
        "leaf filling most of the image, in even light, and try again."
    ),
}


@dataclass(frozen=True)
class InputAssessment:
    """Outcome of the deterministic input suitability gate."""

    decision: str
    code: str
    category: str
    headline: str
    detail: str
    signals: Dict[str, Any]

    @property
    def is_accepted(self) -> bool:
        return self.decision == ACCEPTED

    def to_error_detail(self) -> Dict[str, Any]:
        """
        Serializes a non-accepted assessment into a machine-readable API payload.

        The shape is stable and intentionally not a diagnosis: it carries no class,
        crop, condition, health status or scores.
        """
        return {
            "status": self.decision,
            "code": self.code,
            "category": self.category,
            "message": self.headline,
            "detail": self.detail,
            "retryable": True,
            "signals": self.signals,
        }


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    """Validates a HxWx3 RGB array and returns float32 channel values in [0, 1]."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Expected a non-empty RGB array.")
    if float(arr.max()) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _hsv_channels(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HSV conversion (H in degrees, S and V in [0, 1])."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = rgb.max(axis=-1)
    min_c = rgb.min(axis=-1)
    delta = max_c - min_c

    saturation = np.where(max_c > 1e-6, delta / np.maximum(max_c, 1e-6), 0.0)

    hue = np.zeros_like(max_c)
    chromatic = delta > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        is_r = (max_c == r) & chromatic
        is_g = (max_c == g) & chromatic & ~is_r
        is_b = chromatic & ~is_r & ~is_g
        hue[is_r] = ((g - b)[is_r] / delta[is_r]) % 6.0
        hue[is_g] = ((b - r)[is_g] / delta[is_g]) + 2.0
        hue[is_b] = ((r - g)[is_b] / delta[is_b]) + 4.0
    hue = np.nan_to_num(hue * 60.0, nan=0.0, posinf=0.0, neginf=0.0)
    return hue, saturation, max_c


def _plant_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels consistent with plant pigmentation."""
    hue, saturation, value = _hsv_channels(rgb)
    return (
        (hue >= PLANT_HUE_MIN_DEG)
        & (hue <= PLANT_HUE_MAX_DEG)
        & (saturation >= PLANT_SATURATION_MIN)
        & (value >= PLANT_VALUE_MIN)
    )


def _connected_region_stats(mask: np.ndarray) -> Tuple[float, int]:
    """
    Returns (largest connected component area fraction, count of qualifying components)
    for an 8-connected boolean mask, using run-length union-find (no SciPy dependency).
    """
    height = mask.shape[0]
    total = mask.size
    if total == 0 or not mask.any():
        return 0.0, 0

    parent: List[int] = []
    areas: List[int] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        if areas[root_a] < areas[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        areas[root_a] += areas[root_b]

    previous_runs: List[Tuple[int, int, int]] = []

    for y in range(height):
        row = mask[y].astype(np.int8)
        transitions = np.flatnonzero(np.diff(np.concatenate(([0], row, [0]))))
        current_runs: List[Tuple[int, int, int]] = []
        for start, end in zip(transitions[0::2], transitions[1::2]):
            label = len(parent)
            parent.append(label)
            areas.append(int(end - start))
            current_runs.append((int(start), int(end), label))

        for start, end, label in current_runs:
            for prev_start, prev_end, prev_label in previous_runs:
                # 8-connectivity: runs overlap when expanded by one pixel.
                if prev_end >= start and prev_start <= end:
                    union(label, prev_label)
        previous_runs = current_runs

    largest = 0
    qualifying = 0
    area_floor = INPUT_GATE_REGION_AREA_FLOOR * total
    for node in range(len(parent)):
        if find(node) == node:
            largest = max(largest, areas[node])
            if areas[node] >= area_floor and areas[node] > 0:
                qualifying += 1

    return float(largest) / float(total), int(qualifying)


def _detail_variance(luminance: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of luminance; a low value means no detail."""
    lap = (
        -4.0 * luminance
        + np.roll(luminance, 1, axis=0)
        + np.roll(luminance, -1, axis=0)
        + np.roll(luminance, 1, axis=1)
        + np.roll(luminance, -1, axis=1)
    )
    # Ignore the wrapped border contribution.
    lap = lap[1:-1, 1:-1] if lap.shape[0] > 2 and lap.shape[1] > 2 else lap
    return float(np.var(lap))


def _assessment(decision: str, code: str, category: str, signals: Dict[str, Any]) -> InputAssessment:
    return InputAssessment(
        decision=decision,
        code=code,
        category=category,
        headline=HEADLINE,
        detail=_DETAIL_MESSAGES[code],
        signals=signals,
    )


def _trained_gate_verdict(gate_probabilities: Optional[Dict[str, Any]]) -> Optional[tuple]:
    """
    Extracts (label, confidence) from gate probabilities, or None when unavailable.

    Malformed payloads are treated as "gate unavailable" so the deterministic fallback
    still runs; silently skipping content screening is never an option.
    """
    if not gate_probabilities:
        return None
    try:
        scores = {str(key): float(value) for key, value in gate_probabilities.items()}
    except (TypeError, ValueError):
        return None
    if set(scores) != set(GATE_CLASSES):
        return None
    if any(value != value for value in scores.values()):  # NaN
        return None
    label = max(scores, key=scores.get)
    return label, scores[label], scores


def assess_image(
    image: np.ndarray,
    gate_probabilities: Optional[Dict[str, Any]] = None,
) -> InputAssessment:
    """
    Assesses a decoded RGB image for input suitability.

    Args:
        image: (H, W, 3) array in [0, 255] (as produced by ``preprocess_image``) or [0, 1].
        gate_probabilities: Optional class probabilities from the trained leaf-acceptance
            gate. When present they are authoritative for the content decision; when absent
            (model missing or disabled) the deterministic pigment screen decides instead.

    Returns:
        InputAssessment. Never raises for ordinary image content; malformed arrays raise
        ValueError (a programming error, not user input).
    """
    rgb = _normalize_rgb(image)
    mask = _plant_mask(rgb)

    plant_evidence = float(mask.mean())
    dominant_region, region_count = _connected_region_stats(mask)

    luminance = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    mean_luminance = float(luminance.mean())
    detail_variance = _detail_variance(luminance)

    signals: Dict[str, Any] = {
        "plant_evidence": round(plant_evidence, 4),
        "dominant_region": round(dominant_region, 4),
        "region_count": region_count,
        "mean_luminance": round(mean_luminance, 4),
        "detail_variance": round(detail_variance, 6),
    }

    verdict = _trained_gate_verdict(gate_probabilities)
    if verdict is not None:
        label, confidence, scores = verdict
        signals["gate_probabilities"] = {name: round(value, 4) for name, value in scores.items()}
        signals["gate_label"] = label
        signals["gate_confidence"] = round(confidence, 4)
        signals["gate_threshold"] = LEAF_GATE_MIN_CONFIDENCE

    if not ENABLE_INPUT_GATE:
        return InputAssessment(
            decision=ACCEPTED,
            code=CODE_ACCEPTED,
            category="supported_leaf",
            headline="Input suitability gate disabled by configuration.",
            detail="All decodable images proceed to classification.",
            signals=signals,
        )

    if mean_luminance < INPUT_GATE_MIN_MEAN_LUMINANCE:
        return _assessment(ABSTAINED, CODE_UNDEREXPOSED, "low_quality_image", signals)

    if verdict is not None:
        label, confidence, _ = verdict
        if confidence < LEAF_GATE_MIN_CONFIDENCE:
            return _assessment(ABSTAINED, CODE_AMBIGUOUS, "ambiguous_image", signals)
        if label == GATE_SUPPORTED_LEAF:
            return InputAssessment(
                decision=ACCEPTED,
                code=CODE_ACCEPTED,
                category="supported_leaf",
                headline="Input accepted for classification.",
                detail="A supported crop leaf was verified in a usable photograph.",
                signals=signals,
            )
        if label == GATE_OTHER_PLANT:
            return _assessment(REJECTED, CODE_UNSUPPORTED_SPECIES, "unsupported_species", signals)
        return _assessment(REJECTED, CODE_NON_LEAF, "non_leaf_image", signals)

    # Deterministic fallback: no trained gate artefact is available.
    if plant_evidence < INPUT_GATE_REJECT_BELOW_PLANT_EVIDENCE:
        return _assessment(REJECTED, CODE_NON_LEAF, "non_leaf_image", signals)

    if detail_variance < INPUT_GATE_MIN_DETAIL_VARIANCE:
        return _assessment(ABSTAINED, CODE_LOW_DETAIL, "low_quality_image", signals)

    if plant_evidence < INPUT_GATE_ACCEPT_AT_PLANT_EVIDENCE:
        return _assessment(ABSTAINED, CODE_SPARSE, "ambiguous_image", signals)

    if (
        region_count >= INPUT_GATE_SCATTERED_REGION_COUNT
        and dominant_region < INPUT_GATE_MIN_DOMINANT_REGION
    ):
        return _assessment(ABSTAINED, CODE_SCATTERED, "ambiguous_image", signals)

    return InputAssessment(
        decision=ACCEPTED,
        code=CODE_ACCEPTED,
        category="supported_leaf",
        headline="Input accepted for classification.",
        detail="Plant-tissue evidence is present in a framed, usable photograph.",
        signals=signals,
    )
