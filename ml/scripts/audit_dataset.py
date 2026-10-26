#!/usr/bin/env python3
"""
Crop Disease Dataset Audit Tool
===============================
Part of the "AI-Based Crop Disease Detection and Treatment Recommendation System"

This script inspects a downloaded dataset directory without modifying or moving files.
It analyzes:
  - File formats, corrupted images, and non-image files
  - Solanaceae crop presence (Tomato, Potato, Bell Pepper/Capsicum)
  - Healthy vs. Diseased category identification
  - Class distribution, imbalance ratio, and majority-class baseline accuracy
  - Exact duplicates (SHA-256 byte hashing)
  - Potential near-duplicates (64-bit difference hash, dHash)
  - Cross-split data leakage (if train/val/test partitions exist)

Outputs:
  - dataset_audit_report.md  (human-readable markdown for project documentation)
  - dataset_audit_report.json (machine-readable structured metrics)

Usage:
  py -3.11 ml/scripts/audit_dataset.py --data-dir "ml/data/raw_dataset" --output-dir "docs/audit_reports"
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Supported image file extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Known partition directory names for split-leakage checking
KNOWN_PARTITIONS = {"train", "val", "valid", "validation", "test"}

# Transparent matching rules for target Solanaceae crops
# Patterns are evaluated against lowercase folder / class names.
CROP_RULES = {
    "Tomato": [r"\btomato\b", r"^tomato", r"solanum[_\s]lycopersicum"],
    "Potato": [r"\bpotato\b", r"^potato", r"solanum[_\s]tuberosum"],
    "Bell Pepper": [
        r"\bpepper\b",
        r"\bcapsicum\b",
        r"\bbell[_\s]pepper\b",
        r"\bpepper[_\s]bell\b",
        r"^pepper",
    ],
}

# Heuristics to detect healthy classes
HEALTHY_PATTERNS = [r"\bhealthy\b", r"\bnormal\b", r"\buninfected\b"]


def compute_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file reading in chunks to prevent memory spikes."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Computes a 64-bit Difference Hash (dHash) for an image using Pillow.

    Algorithm:
      1. Convert image to 8-bit grayscale ('L').
      2. Resize to (hash_size + 1, hash_size) -> (9, 8) using bilinear interpolation.
         This emphasizes structural gradient information while discarding high-frequency noise.
      3. Compare adjacent pixels horizontally across each of the 8 rows.
      4. Assign bit 1 if left_pixel > right_pixel, else 0.
      5. Pack 64 bits into a 64-bit unsigned integer.

    Parameters:
      image: PIL.Image object
      hash_size: Grid height and width (default 8 -> 64-bit hash)

    Returns:
      64-bit integer representing the structural hash.
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(resized.getdata())
    width = hash_size + 1
    dhash_int = 0
    for row in range(hash_size):
        row_offset = row * width
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            dhash_int = (dhash_int << 1) | (1 if left > right else 0)
    return dhash_int


def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes the Hamming distance (number of differing bits) between two 64-bit integer hashes."""
    return (hash1 ^ hash2).bit_count()


def detect_crop_and_health(class_name: str) -> Tuple[str, str, str]:
    """
    Applies documented regex rules to extract:
      - Crop name (Tomato, Potato, Bell Pepper, or 'Other / Unmatched')
      - Health status (Healthy, Diseased, or Unknown)
      - Condition (clean condition string)
    """
    # Normalize underscores, hyphens, and spaces to single spaces for robust regex matching
    clean_text = re.sub(r"[\W_]+", " ", class_name.lower()).strip()

    # Match Crop
    matched_crop = "Other / Unmatched"
    for crop, patterns in CROP_RULES.items():
        if any(re.search(pat, clean_text) for pat in patterns):
            matched_crop = crop
            break

    # Match Health
    is_healthy = any(re.search(pat, clean_text) for pat in HEALTHY_PATTERNS)
    health_status = "Healthy" if is_healthy else "Diseased"

    # Human-readable condition label
    cleaned_condition = class_name.replace("___", " - ").replace("__", " - ").replace("_", " ")

    return matched_crop, health_status, cleaned_condition


def find_near_duplicate_candidates(
    dhash_records: List[Tuple[str, int, str, str]],
    threshold: int = 2,
    max_candidates: int = 100,
) -> List[Dict]:
    """
    Finds potential near-duplicate image pairs using multi-index hashing on 64-bit dHash.
    
    Pigeonhole Principle Indexing:
      A 64-bit hash is partitioned into four 16-bit segments (B0, B1, B2, B3).
      If two hashes differ by <= threshold (default 2 bits), at least two 16-bit blocks
      must have at most 1 bit difference, and candidates must share or closely match blocks.
      This provides fast candidate pruning without O(N^2) comparison.
    """
    if len(dhash_records) < 2:
        return []

    # Map each 16-bit block to record indices
    block_index = defaultdict(list)
    for idx, (_, dhash_val, _, _) in enumerate(dhash_records):
        for block_id in range(4):
            # Extract 16 bits
            block = (dhash_val >> (block_id * 16)) & 0xFFFF
            block_index[(block_id, block)].append(idx)

    checked_pairs: Set[Tuple[int, int]] = set()
    candidates = []

    for idx1, (path1, hash1, cls1, part1) in enumerate(dhash_records):
        # Retrieve candidate neighbors that share any 16-bit block
        candidate_indices = set()
        for block_id in range(4):
            block = (hash1 >> (block_id * 16)) & 0xFFFF
            candidate_indices.update(block_index[(block_id, block)])

        for idx2 in candidate_indices:
            if idx2 <= idx1:
                continue
            pair = (idx1, idx2)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            path2, hash2, cls2, part2 = dhash_records[idx2]
            dist = hamming_distance(hash1, hash2)
            if dist <= threshold:
                candidates.append({
                    "image_1": path1,
                    "image_2": path2,
                    "class_1": cls1,
                    "class_2": cls2,
                    "partition_1": part1,
                    "partition_2": part2,
                    "hamming_distance": dist,
                    "cross_partition_leak": (part1 != part2) if (part1 and part2) else False,
                    "cross_class_match": (cls1 != cls2),
                })
                if len(candidates) >= max_candidates:
                    return candidates

    return candidates


def audit_dataset(
    data_dir: Path,
    dhash_threshold: int = 2,
    min_dimension: int = 128,
    max_near_dups: int = 100,
) -> Dict:
    """Executes comprehensive audit over the specified directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Provided dataset directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {data_dir}")

    start_time = time.time()

    # Discover structure
    all_files: List[Path] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            all_files.append(Path(root) / f)

    total_files = len(all_files)
    image_paths: List[Path] = []
    non_image_files: List[str] = []

    for f in all_files:
        if f.suffix.lower() in VALID_IMAGE_EXTS:
            image_paths.append(f)
        else:
            rel = str(f.relative_to(data_dir))
            non_image_files.append(rel)

    # Detect top-level partitions if present
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    has_partitions = any(p.name.lower() in KNOWN_PARTITIONS for p in subdirs)

    # Class parsing
    class_images: Dict[str, List[Path]] = defaultdict(list)
    image_partition_map: Dict[str, str] = {}

    for img_path in image_paths:
        rel_parts = img_path.relative_to(data_dir).parts
        if len(rel_parts) > 1:
            if has_partitions and rel_parts[0].lower() in KNOWN_PARTITIONS:
                part = rel_parts[0].lower()
                cls = rel_parts[1] if len(rel_parts) > 2 else "unclassified"
            else:
                part = "unpartitioned"
                cls = rel_parts[0]
        else:
            part = "unpartitioned"
            cls = "root_unclassified"

        class_images[cls].append(img_path)
        image_partition_map[str(img_path)] = part

    # Image metadata analysis
    format_counts: Counter = Counter()
    unreadable_files: List[Dict] = []
    dimensions: List[Tuple[int, int]] = []
    small_images: List[Dict] = []
    sha256_map: Dict[str, List[str]] = defaultdict(list)
    dhash_records: List[Tuple[str, int, str, str]] = []

    for img_path in image_paths:
        rel_path = str(img_path.relative_to(data_dir))
        cls = [k for k, v in class_images.items() if img_path in v][0]
        part = image_partition_map.get(str(img_path), "")

        # Exact SHA-256 hash
        try:
            sha = compute_sha256(img_path)
            sha256_map[sha].append(rel_path)
        except Exception as e:
            unreadable_files.append({"file": rel_path, "reason": f"Hash error: {str(e)}"})
            continue

        # Pillow image inspection
        if PIL_AVAILABLE:
            try:
                with Image.open(img_path) as im:
                    fmt = im.format or img_path.suffix.upper().lstrip(".")
                    format_counts[fmt] += 1
                    w, h = im.size
                    dimensions.append((w, h))

                    if w < min_dimension or h < min_dimension:
                        small_images.append({"file": rel_path, "width": w, "height": h})

                    im.verify()

                # Re-open for difference hash computation (verify invalidates stream)
                with Image.open(img_path) as im:
                    dh = compute_dhash(im)
                    dhash_records.append((rel_path, dh, cls, part))

            except (UnidentifiedImageError, OSError, Exception) as e:
                unreadable_files.append({"file": rel_path, "reason": f"Corrupt image: {str(e)}"})
        else:
            format_counts[img_path.suffix.upper().lstrip(".")] += 1

    # Exact duplicates analysis
    exact_duplicate_groups = {h: paths for h, paths in sha256_map.items() if len(paths) > 1}
    total_exact_duplicate_images = sum(len(paths) - 1 for paths in exact_duplicate_groups.values())

    # Near-duplicates analysis (dHash)
    near_duplicate_candidates = []
    if PIL_AVAILABLE and len(dhash_records) > 0:
        near_duplicate_candidates = find_near_duplicate_candidates(
            dhash_records, threshold=dhash_threshold, max_candidates=max_near_dups
        )

    # Check for cross-partition leakage
    cross_partition_leaks = []
    for sha, paths in exact_duplicate_groups.items():
        parts = {image_partition_map.get(str(data_dir / p)) for p in paths}
        if len(parts) > 1 and "unpartitioned" not in parts:
            cross_partition_leaks.append({
                "hash": sha,
                "partitions_involved": list(parts),
                "files": paths,
            })

    # Dimension statistics
    dim_stats = {}
    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        dim_stats = {
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_width": round(sum(widths) / len(widths), 1),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_height": round(sum(heights) / len(heights), 1),
            "square_aspect_count": sum(1 for w, h in dimensions if w == h),
            "total_measured": len(dimensions),
        }

    # Class metrics
    total_valid_images = len(image_paths) - len(unreadable_files)
    class_breakdown = []
    crop_coverage = {"Tomato": [], "Potato": [], "Bell Pepper": [], "Other": []}

    class_counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    for cls, count in sorted_classes:
        crop, health, condition = detect_crop_and_health(cls)
        pct = round((count / total_valid_images * 100), 2) if total_valid_images > 0 else 0.0

        item = {
            "class_name": cls,
            "crop": crop,
            "health_status": health,
            "condition": condition,
            "count": count,
            "percentage": pct,
        }
        class_breakdown.append(item)

        if crop in crop_coverage:
            crop_coverage[crop].append(item)
        else:
            crop_coverage["Other"].append(item)

    # Imbalance and baseline calculations
    counts_list = list(class_counts.values()) if class_counts else [0]
    max_count = max(counts_list) if counts_list else 0
    min_count = min(counts_list) if counts_list else 0
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else None
    majority_class_baseline = round((max_count / total_valid_images * 100), 2) if total_valid_images > 0 else 0.0

    audit_duration = round(time.time() - start_time, 2)

    return {
        "dataset_path": str(data_dir.resolve()),
        "audit_timestamp": datetime.now().isoformat(),
        "audit_duration_seconds": audit_duration,
        "pil_available": PIL_AVAILABLE,
        "summary": {
            "total_files": total_files,
            "total_images": len(image_paths),
            "valid_images": total_valid_images,
            "unreadable_images": len(unreadable_files),
            "non_image_files": len(non_image_files),
            "total_classes": len(class_images),
            "has_partitions": has_partitions,
        },
        "solanaceae_coverage": {
            "tomato_classes": len(crop_coverage["Tomato"]),
            "tomato_images": sum(c["count"] for c in crop_coverage["Tomato"]),
            "potato_classes": len(crop_coverage["Potato"]),
            "potato_images": sum(c["count"] for c in crop_coverage["Potato"]),
            "bell_pepper_classes": len(crop_coverage["Bell Pepper"]),
            "bell_pepper_images": sum(c["count"] for c in crop_coverage["Bell Pepper"]),
            "other_classes": len(crop_coverage["Other"]),
            "other_images": sum(c["count"] for c in crop_coverage["Other"]),
        },
        "class_imbalance": {
            "max_class_count": max_count,
            "min_class_count": min_count,
            "imbalance_ratio": imbalance_ratio,
            "majority_class_baseline_accuracy": majority_class_baseline,
        },
        "formats": dict(format_counts),
        "dimensions": dim_stats,
        "small_images_count": len(small_images),
        "exact_duplicates": {
            "duplicate_groups_count": len(exact_duplicate_groups),
            "redundant_images_count": total_exact_duplicate_images,
        },
        "near_duplicates": {
            "dhash_threshold": dhash_threshold,
            "candidate_pairs_found": len(near_duplicate_candidates),
        },
        "cross_partition_leakage": {
            "exact_leak_clusters_count": len(cross_partition_leaks),
        },
        "details": {
            "class_breakdown": class_breakdown,
            "unreadable_files": unreadable_files[:50],
            "non_image_files": non_image_files[:50],
            "exact_duplicate_samples": [
                {"hash": h, "sample_files": paths[:5]}
                for h, paths in list(exact_duplicate_groups.items())[:20]
            ],
            "near_duplicate_samples": near_duplicate_candidates[:max_near_dups],
            "cross_partition_leak_samples": cross_partition_leaks[:20],
        },
    }


def generate_markdown_report(report: Dict) -> str:
    """Renders the audit dictionary as a clean, academic Markdown document."""
    s = report["summary"]
    cov = report["solanaceae_coverage"]
    imb = report["class_imbalance"]
    dim = report["dimensions"]
    ex = report["exact_duplicates"]
    nd = report["near_duplicates"]
    leak = report["cross_partition_leakage"]

    md = []
    md.append("# Dataset Audit Report: Crop Disease Collection")
    md.append(f"**Audit Generated:** {report['audit_timestamp']}  ")
    md.append(f"**Dataset Directory:** `{report['dataset_path']}`  ")
    md.append(f"**Audit Execution Time:** {report['audit_duration_seconds']} seconds  ")
    md.append(f"**Image Analysis Engine:** {'Pillow (PIL) Active' if report['pil_available'] else 'Header-Only Mode'}\n")

    # Important Academic Notice
    md.append("> [!IMPORTANT]")
    md.append("> **Data Leakage Notice for Academic Evaluation:**")
    md.append("> This audit is the first stage of leakage prevention. Any exact duplicate groups or confirmed near-duplicate")
    md.append("> pairs identified below must **not** be split across training, validation, and test sets.")
    md.append("> Image augmentations must be performed dynamically on the training split only.\n")

    # Executive Summary Table
    md.append("## 1. Executive Summary")
    md.append("| Metric | Value | Status / Description |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Total Files Audited** | {s['total_files']} | All files in directory hierarchy |")
    md.append(f"| **Valid Images** | {s['valid_images']} | Verified readable image binaries |")
    md.append(f"| **Corrupted / Unreadable Images** | {s['unreadable_images']} | {'⚠️ Flagged' if s['unreadable_images'] > 0 else '✅ None'} |")
    md.append(f"| **Non-Image Files Found** | {s['non_image_files']} | {'⚠️ Needs cleanup' if s['non_image_files'] > 0 else '✅ None'} |")
    md.append(f"| **Total Classes Detected** | {s['total_classes']} | Extracted from folder hierarchy |")
    md.append(f"| **Pre-existing Partitions** | {'Yes (Train/Val/Test)' if s['has_partitions'] else 'No (Single Directory)'} | Layout detection |")
    md.append(f"| **Majority-Class Baseline** | **{imb['majority_class_baseline_accuracy']}%** | Minimum performance floor to beat |")
    md.append(f"| **Class Imbalance Ratio** | **{imb['imbalance_ratio']} : 1** | Ratio of max class to min class |")
    md.append("")

    # Solanaceae Coverage Table
    md.append("## 2. Solanaceae Crop Coverage Verification")
    md.append("Transparent rule matching against confirmed target crops (Tomato, Potato, Bell Pepper):")
    md.append("")
    md.append("| Target Crop | Classes Found | Total Images | Status |")
    md.append("| :--- | :--- | :--- | :--- |")
    
    t_status = "✅ Supported" if cov['tomato_classes'] > 0 else "❌ Missing"
    p_status = "✅ Supported" if cov['potato_classes'] > 0 else "❌ Missing"
    b_status = "✅ Supported" if cov['bell_pepper_classes'] > 0 else "❌ Missing"

    t_lbl = "class" if cov['tomato_classes'] == 1 else "classes"
    p_lbl = "class" if cov['potato_classes'] == 1 else "classes"
    b_lbl = "class" if cov['bell_pepper_classes'] == 1 else "classes"
    o_lbl = "class" if cov['other_classes'] == 1 else "classes"

    md.append(f"| **Tomato** | {cov['tomato_classes']} {t_lbl} | {cov['tomato_images']} images | {t_status} |")
    md.append(f"| **Potato** | {cov['potato_classes']} {p_lbl} | {cov['potato_images']} images | {p_status} |")
    md.append(f"| **Bell Pepper** | {cov['bell_pepper_classes']} {b_lbl} | {cov['bell_pepper_images']} images | {b_status} |")
    md.append(f"| *Other / Non-Solanaceae* | {cov['other_classes']} {o_lbl} | {cov['other_images']} images | Filter candidate |")
    md.append("")

    # Class Breakdown Table
    md.append("## 3. Detected Class Breakdown")
    md.append("| Class Directory Name | Crop | Health Status | Images | % Representation |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for c in report["details"]["class_breakdown"]:
        md.append(f"| `{c['class_name']}` | {c['crop']} | {c['health_status']} | {c['count']} | {c['percentage']}% |")
    md.append("")

    # Quality and Dimensions
    md.append("## 4. Image Quality & Dimension Profile")
    if dim:
        md.append(f"- **Resolution Range:** Min: `{dim['min_width']}x{dim['min_height']}` px | Max: `{dim['max_width']}x{dim['max_height']}` px")
        md.append(f"- **Mean Resolution:** `{dim['mean_width']}x{dim['mean_height']}` px")
        md.append(f"- **Square Aspect Ratio (1:1):** {dim['square_aspect_count']} / {dim['total_measured']} images")
    md.append(f"- **Images Below Threshold (<128px):** {report['small_images_count']}")
    md.append(f"- **Image Formats Encountered:** {', '.join(f'{k} ({v})' for k, v in report['formats'].items())}")
    md.append("")

    # Duplicate & Leakage Findings
    md.append("## 5. Duplicate Detection & Data Leakage Review")
    md.append(f"- **Exact Byte Duplicates (SHA-256):** {ex['duplicate_groups_count']} groups ({ex['redundant_images_count']} redundant duplicate files).")
    md.append(f"- **Perceptual Near-Duplicate Candidates (dHash Hamming distance <= {nd['dhash_threshold']}):** {nd['candidate_pairs_found']} candidate pairs flagged.")
    md.append(f"- **Cross-Partition Leakage Clusters:** {leak['exact_leak_clusters_count']} clusters cross train/val/test boundaries.")
    md.append("")

    if report["details"]["exact_duplicate_samples"]:
        md.append("### Sample Exact Duplicate Groups")
        for idx, grp in enumerate(report["details"]["exact_duplicate_samples"][:5], 1):
            md.append(f"**Group {idx} (SHA-256: `{grp['hash'][:12]}...`):**")
            for fpath in grp["sample_files"]:
                md.append(f"  - `{fpath}`")
        md.append("")

    if report["details"]["near_duplicate_samples"]:
        md.append("### Sample Near-Duplicate Candidates (Requires Review)")
        md.append("| Image A | Image B | Class A | Class B | Distance | Cross-Split Leak? |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for pair in report["details"]["near_duplicate_samples"][:10]:
            md.append(f"| `{pair['image_1']}` | `{pair['image_2']}` | {pair['class_1']} | {pair['class_2']} | {pair['hamming_distance']} | {'⚠️ LEAK' if pair['cross_partition_leak'] else 'No'} |")
        md.append("")

    # Actionable Recommendations
    md.append("## 6. Recommendations for Phase 2 (ML Pipeline)")
    if cov["tomato_classes"] > 0 and cov["potato_classes"] > 0 and cov["bell_pepper_classes"] > 0:
        md.append("1. **Crops Approved:** Tomato, Potato, and Bell Pepper are all present with viable representation.")
    else:
        md.append("1. **Crop Selection Warning:** One or more target Solanaceae crops are missing. Review dataset selection.")
    md.append(f"2. **Baseline Target:** Model evaluation must significantly exceed the **{imb['majority_class_baseline_accuracy']}%** majority-class baseline.")
    md.append("3. **Macro-F1 Optimization:** Due to class imbalance, monitor Macro-F1 rather than raw accuracy.")
    md.append("4. **Split Sanitization:** Deduplicate identical copies before creating stratified splits.")
    md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Crop Disease Dataset Audit Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example Usage:
  py -3.11 ml/scripts/audit_dataset.py --data-dir "ml/data/candidate_dataset" --output-dir "docs/audit_reports"
        """,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Path to the directory containing dataset images.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="docs/audit_reports",
        help="Directory where dataset_audit_report.md and dataset_audit_report.json will be saved.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=2,
        help="Max Hamming distance (0-64) to flag images as perceptual near-duplicates (default: 2).",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=128,
        help="Minimum width/height in pixels; images smaller than this are flagged (default: 128).",
    )
    parser.add_argument(
        "--max-near-dups",
        type=int,
        default=100,
        help="Max number of near-duplicate pairs to include in the output report (default: 100).",
    )

    args = parser.parse_args()

    input_path = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    print(f"\n[1/4] Initializing Dataset Audit...")
    print(f"      Target Directory : {input_path}")
    print(f"      Report Output    : {out_dir}")
    print(f"      Pillow Status    : {'Available (Full Image Inspection)' if PIL_AVAILABLE else 'Unavailable'}")

    try:
        audit_results = audit_dataset(
            data_dir=input_path,
            dhash_threshold=args.dhash_threshold,
            min_dimension=args.min_dimension,
            max_near_dups=args.max_near_dups,
        )
    except Exception as e:
        print(f"\n[ERROR] Audit aborted: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"[2/4] Audit Completed in {audit_results['audit_duration_seconds']}s")
    print(f"      Total images scanned: {audit_results['summary']['valid_images']}")
    print(f"      Total classes found : {audit_results['summary']['total_classes']}")

    # Save Reports
    print(f"[3/4] Generating Reports...")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"      Saved JSON report: {json_path}")

    md_content = generate_markdown_report(audit_results)
    md_path = out_dir / "dataset_audit_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"      Saved Markdown report: {md_path}")

    print(f"[4/4] Done. Review the generated report before proceeding to Phase 2.\n")


if __name__ == "__main__":
    main()
