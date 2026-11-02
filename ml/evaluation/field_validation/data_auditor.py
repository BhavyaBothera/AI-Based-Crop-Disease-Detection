"""
Field Dataset Auditor & Leakage Detector
========================================
Part of PHYTOSIGNAL 2.0 - Phase 1: Field Validation & Generalization Audit.

Comprehensive quality and leakage audit tool for field validation datasets.
Performs:
1. Image readability and format validation (JPEG, PNG, WebP).
2. Physical file corruption detection and dimension limit enforcement.
3. Exact duplicate detection via SHA-256 byte hashing.
4. Perceptual near-duplicate detection via 64-bit dHash and Hamming distance.
5. Group-aware leakage detection across field_dev and field_test splits.
6. Controlled dataset leak verification (detects overlap with PlantVillage baseline).
7. Stratification and class distribution profiling.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from .manifest_schema import (
    CROP_TO_CLASSES,
    FieldSampleRecord,
    ManifestValidationError,
    load_and_validate_manifest,
)

logger = logging.getLogger(__name__)

# Operational limits
MIN_DIMENSION = 64
MAX_IMAGE_PIXELS = 20 * 1024 * 1024  # 20 MP ceiling
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
DEFAULT_NEAR_DUPLICATE_HAMMING_THRESHOLD = 6


def compute_sha256(image_bytes: bytes) -> str:
    """Computes SHA-256 hexadecimal digest for raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) for perceptual similarity analysis.
    Resizes to (hash_size + 1, hash_size) grayscale, compares adjacent horizontal pixels.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return f"{decimal_val:016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Computes bitwise Hamming distance between two 16-character hex strings."""
    if not hex1 or not hex2 or len(hex1) != 16 or len(hex2) != 16:
        return 64
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    return bin(val1 ^ val2).count("1")


@dataclass
class ImageQualityReport:
    """Diagnostic outcome for a single image file."""
    is_valid: bool
    format: Optional[str] = None
    width: int = 0
    height: int = 0
    sha256: str = ""
    dhash: str = ""
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def audit_single_image(file_path: Path) -> ImageQualityReport:
    """
    Validates physical container, readable scanlines, dimensions, and hashes.
    """
    if not file_path.is_file():
        return ImageQualityReport(
            is_valid=False,
            error_type="FileNotFoundError",
            error_message=f"File not found on disk: {file_path}",
        )

    try:
        raw_bytes = file_path.read_bytes()
        if len(raw_bytes) == 0:
            return ImageQualityReport(
                is_valid=False,
                error_type="EmptyFileError",
                error_message="File is empty (0 bytes).",
            )

        sha = compute_sha256(raw_bytes)

        with Image.open(file_path) as img:
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                return ImageQualityReport(
                    is_valid=False,
                    format=fmt,
                    sha256=sha,
                    error_type="UnsupportedFormatError",
                    error_message=f"Unsupported format '{fmt}'. Allowed: {ALLOWED_FORMATS}",
                )

            w, h = img.size
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                return ImageQualityReport(
                    is_valid=False,
                    format=fmt,
                    width=w,
                    height=h,
                    sha256=sha,
                    error_type="InvalidDimensionsError",
                    error_message=f"Dimensions ({w}x{h}) below minimum required {MIN_DIMENSION}x{MIN_DIMENSION}.",
                )

            if w * h > MAX_IMAGE_PIXELS:
                return ImageQualityReport(
                    is_valid=False,
                    format=fmt,
                    width=w,
                    height=h,
                    sha256=sha,
                    error_type="InvalidDimensionsError",
                    error_message=f"Pixel count ({w * h:,}) exceeds maximum limit ({MAX_IMAGE_PIXELS:,}).",
                )

            # Check raster decode integrity
            img.load()
            dh = compute_dhash(img)

        return ImageQualityReport(
            is_valid=True,
            format=fmt,
            width=w,
            height=h,
            sha256=sha,
            dhash=dh,
        )

    except Exception as exc:
        return ImageQualityReport(
            is_valid=False,
            error_type="CorruptImageError",
            error_message=f"Corrupt or unreadable image: {str(exc)}",
        )


@dataclass
class AuditReport:
    """Aggregated results of a complete dataset quality and leakage audit."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_files: List[Dict[str, str]] = field(default_factory=list)
    corrupt_images: List[Dict[str, str]] = field(default_factory=list)
    unsupported_format_images: List[Dict[str, str]] = field(default_factory=list)
    exact_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    near_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    group_leaks: List[Dict[str, Any]] = field(default_factory=list)
    controlled_dataset_leaks: List[Dict[str, Any]] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    crop_distribution: Dict[str, int] = field(default_factory=dict)
    split_distribution: Dict[str, int] = field(default_factory=dict)
    verification_distribution: Dict[str, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    lighting_distribution: Dict[str, int] = field(default_factory=dict)
    background_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Returns True if no fatal data integrity or leakage issues were detected."""
        return (
            len(self.missing_files) == 0
            and len(self.corrupt_images) == 0
            and len(self.unsupported_format_images) == 0
            and len(self.group_leaks) == 0
            and len(self.controlled_dataset_leaks) == 0
        )

    @property
    def corrupted_files(self) -> List[Dict[str, str]]:
        """Backward-compatibility alias."""
        return self.corrupt_images

    @property
    def group_leakages(self) -> List[Dict[str, Any]]:
        """Backward-compatibility alias."""
        return self.group_leaks

    @property
    def controlled_baseline_overlap_count(self) -> int:
        """Backward-compatibility alias."""
        return len(self.controlled_dataset_leaks)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias
AuditSummary = AuditReport


class DataQualityAuditor:
    """
    Auditor for field validation manifests and raw imagery.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[Path, str]] = None,
        dataset_root: Optional[Union[Path, str]] = None,
        controlled_manifest_path: Optional[Union[Path, str]] = None,
        near_duplicate_threshold: int = DEFAULT_NEAR_DUPLICATE_HAMMING_THRESHOLD,
    ):
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        if dataset_root:
            self.dataset_root = Path(dataset_root).resolve()
        elif self.manifest_path:
            self.dataset_root = self.manifest_path.parent
        else:
            self.dataset_root = Path.cwd()

        self.controlled_manifest_path = (
            Path(controlled_manifest_path).resolve() if controlled_manifest_path else None
        )
        self.near_duplicate_threshold = near_duplicate_threshold

        self._controlled_hashes: Set[str] = set()
        if self.controlled_manifest_path and self.controlled_manifest_path.is_file():
            self._load_controlled_hashes()

    def _load_controlled_hashes(self) -> None:
        """Loads SHA-256 hashes from the controlled PlantVillage manifest to detect leakage."""
        try:
            with open(self.controlled_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sha = row.get("sha256", "").strip()
                    if sha and sha != "NONE":
                        self._controlled_hashes.add(sha)
            logger.info("Loaded %d controlled baseline hashes for leakage verification.", len(self._controlled_hashes))
        except Exception as exc:
            logger.warning("Could not load controlled baseline hashes: %s", exc)

    def audit_all(self, manifest_path: Optional[Union[Path, str]] = None) -> AuditReport:
        """Loads and audits all records in the specified or default manifest."""
        target_path = Path(manifest_path).resolve() if manifest_path else self.manifest_path
        if not target_path or not target_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {target_path}")

        records, schema_errors = load_and_validate_manifest(target_path)
        return self.audit(records)

    def audit(
        self,
        records: List[FieldSampleRecord],
        compute_perceptual_hashes: bool = True,
    ) -> AuditReport:
        """
        Executes complete quality, integrity, and leakage audit across all records.
        """
        report = AuditReport(total_records=len(records))

        sha_to_records: Dict[str, List[FieldSampleRecord]] = {}
        dhash_to_records: Dict[str, List[FieldSampleRecord]] = {}
        group_to_records: Dict[str, List[FieldSampleRecord]] = {}

        for rec in records:
            # Profiling distributions
            report.class_distribution[rec.ground_truth_class] = report.class_distribution.get(rec.ground_truth_class, 0) + 1
            report.crop_distribution[rec.crop] = report.crop_distribution.get(rec.crop, 0) + 1
            report.split_distribution[rec.split] = report.split_distribution.get(rec.split, 0) + 1
            report.verification_distribution[rec.verification_status] = report.verification_distribution.get(rec.verification_status, 0) + 1
            report.severity_distribution[rec.disease_severity] = report.severity_distribution.get(rec.disease_severity, 0) + 1
            report.lighting_distribution[rec.lighting_condition] = report.lighting_distribution.get(rec.lighting_condition, 0) + 1
            report.background_distribution[rec.background_type] = report.background_distribution.get(rec.background_type, 0) + 1

            # Group split tracking
            grp_id = rec.duplicate_group_id.strip() if rec.duplicate_group_id else ""
            if grp_id and grp_id != "NONE":
                if grp_id not in group_to_records:
                    group_to_records[grp_id] = []
                group_to_records[grp_id].append(rec)

            # Resolve image file
            target_path = Path(rec.file_path)
            if not target_path.is_absolute():
                target_path = self.dataset_root / target_path

            img_report = audit_single_image(target_path)

            if not img_report.is_valid:
                report.invalid_records += 1
                if img_report.error_type == "FileNotFoundError":
                    report.missing_files.append({
                        "image_id": rec.image_id,
                        "file_path": rec.file_path,
                        "error": img_report.error_message or "File not found",
                    })
                elif img_report.error_type == "UnsupportedFormatError":
                    report.unsupported_format_images.append({
                        "image_id": rec.image_id,
                        "file_path": rec.file_path,
                        "format": img_report.format or "unknown",
                        "error": img_report.error_message or "Unsupported format",
                    })
                else:
                    report.corrupt_images.append({
                        "image_id": rec.image_id,
                        "file_path": rec.file_path,
                        "error_type": img_report.error_type or "UnknownError",
                        "error": img_report.error_message or "Corrupted or unreadable",
                    })
                continue

            report.valid_records += 1
            rec.sha256 = img_report.sha256
            rec.dhash = img_report.dhash

            # Check overlap with controlled baseline
            if rec.sha256 in self._controlled_hashes:
                report.controlled_dataset_leaks.append({
                    "image_id": rec.image_id,
                    "sha256": rec.sha256,
                    "file_path": rec.file_path,
                })
                logger.error("Data Leakage Alert: Image %s matches a controlled baseline image!", rec.image_id)

            # Track exact duplicates
            if rec.sha256 not in sha_to_records:
                sha_to_records[rec.sha256] = []
            sha_to_records[rec.sha256].append(rec)

            # Track perceptual dHash
            if compute_perceptual_hashes and rec.dhash:
                if rec.dhash not in dhash_to_records:
                    dhash_to_records[rec.dhash] = []
                dhash_to_records[rec.dhash].append(rec)

        # 1. Exact Duplicate Analysis (Same SHA-256)
        for sha, matched_recs in sha_to_records.items():
            if len(matched_recs) > 1:
                splits = {r.split for r in matched_recs}
                report.exact_duplicates.append({
                    "sha256": sha,
                    "count": len(matched_recs),
                    "image_ids": [r.image_id for r in matched_recs],
                    "crosses_splits": len(splits) > 1,
                    "splits": sorted(splits),
                })

        # 2. Group Leakage Analysis (Same duplicate_group_id across splits)
        for grp, grecs in group_to_records.items():
            splits = {r.split for r in grecs}
            if len(splits) > 1:
                report.group_leaks.append({
                    "group_id": grp,
                    "image_ids": [r.image_id for r in grecs],
                    "splits": sorted(splits),
                })

        # 3. Near-Duplicate Analysis (dHash Hamming distance <= threshold)
        if compute_perceptual_hashes:
            # Check within the same dhash bucket (distance 0)
            for dh, recs in dhash_to_records.items():
                if len(recs) > 1:
                    splits = {r.split for r in recs}
                    report.near_duplicates.append({
                        "hamming_distance": 0,
                        "pair_1_ids": [r.image_id for r in recs[:1]],
                        "pair_2_ids": [r.image_id for r in recs[1:]],
                        "image_ids": [r.image_id for r in recs],
                        "crosses_splits": len(splits) > 1,
                        "splits": sorted(splits),
                    })

            # Check across different dhash buckets
            dhash_list = list(dhash_to_records.keys())
            for i in range(len(dhash_list)):
                for j in range(i + 1, len(dhash_list)):
                    dh1 = dhash_list[i]
                    dh2 = dhash_list[j]
                    dist = hamming_distance(dh1, dh2)
                    if dist <= self.near_duplicate_threshold:
                        recs1 = dhash_to_records[dh1]
                        recs2 = dhash_to_records[dh2]
                        all_recs = recs1 + recs2
                        splits = {r.split for r in all_recs}
                        report.near_duplicates.append({
                            "hamming_distance": dist,
                            "pair_1_ids": [r.image_id for r in recs1],
                            "pair_2_ids": [r.image_id for r in recs2],
                            "image_ids": [r.image_id for r in all_recs],
                            "crosses_splits": len(splits) > 1,
                            "splits": sorted(splits),
                        })

        return report

    @staticmethod
    def generate_markdown_report(report: AuditReport) -> str:
        """Generates a comprehensive Markdown audit summary."""
        md = []
        md.append("# Field Validation Dataset Audit & Leakage Report")
        md.append(f"**Total Manifest Records:** {report.total_records}")
        md.append(f"**Valid & Accessible Records:** {report.valid_records}")
        md.append(f"**Invalid / Inaccessible Records:** {report.invalid_records}")
        status = "PASSED" if report.is_clean else "FAILED"
        md.append(f"**Audit Integrity Status:** **{status}**")
        md.append("")

        md.append("## Integrity Issues Ledger")
        md.append(f"- **Missing Image Files:** {len(report.missing_files)}")
        md.append(f"- **Corrupted / Unreadable Images:** {len(report.corrupt_images)}")
        md.append(f"- **Unsupported Format Images:** {len(report.unsupported_format_images)}")
        md.append(f"- **Exact Duplicate Groups:** {len(report.exact_duplicates)}")
        md.append(f"- **Perceptual Near-Duplicates:** {len(report.near_duplicates)}")
        md.append(f"- **Cross-Split Group Leakages:** {len(report.group_leaks)}")
        md.append(f"- **Controlled Baseline Leaks:** {len(report.controlled_dataset_leaks)}")
        md.append("")

        if report.group_leaks:
            md.append("### [ALERT] Cross-Split Leakage Details")
            md.append("| Group ID | Partitions Crossed | Sample Count |")
            md.append("| :--- | :---: | :---: |")
            for gl in report.group_leaks:
                md.append(f"| `{gl['group_id']}` | {', '.join(gl['splits'])} | {len(gl['image_ids'])} |")
            md.append("")

        if report.controlled_dataset_leaks:
            md.append("### [CRITICAL ALERT] Controlled Baseline Leakage Details")
            md.append("The following field samples duplicate images from the controlled PlantVillage baseline:")
            for cl in report.controlled_dataset_leaks:
                md.append(f"- Image ID `{cl['image_id']}` (SHA-256: `{cl['sha256'][:16]}...`)")
            md.append("")

        md.append("## Split & Class Profiling")
        md.append("### Splits")
        for sname, scount in report.split_distribution.items():
            md.append(f"- **{sname}:** {scount}")
        md.append("")

        md.append("### Crop Distribution")
        for cname, ccount in report.crop_distribution.items():
            md.append(f"- **{cname}:** {ccount}")
        md.append("")

        md.append("### Verification Status Breakdown")
        for vname, vcount in report.verification_distribution.items():
            md.append(f"- **{vname}:** {vcount}")
        md.append("")

        return "\n".join(md)


# Alias
FieldDataAuditor = DataQualityAuditor
