"""
Builds a small, licence-clean corpus of REAL non-plant photographs used as held-out
out-of-distribution (OOD) evidence for the PHYTOSIGNAL input gate.

WHY THIS SCRIPT EXISTS
----------------------
PlantVillage contains no photographs of people, animals, buildings, documents or objects,
so the repository had no way to measure whether a leaf-acceptance model rejects them. This
script assembles such a set from Wikimedia Commons, keeping only files whose licence is
public domain / CC0 / CC BY / CC BY-SA, and records the source page, licence and author for
every downloaded file.

WHAT IT PRODUCES
----------------
    ml/data/ood_corpus/<category>/<slug>.jpg     (640 px wide, JPEG)
    ml/data/ood_corpus/manifest.json            (per-file provenance + SHA-256)

`ml/data/` is git-ignored: the corpus is research data, not source. The manifest written
here is what makes the evaluation reproducible and attributable.

USAGE
    .venv/Scripts/python.exe ml/scripts/build_ood_corpus.py
    .venv/Scripts/python.exe ml/scripts/build_ood_corpus.py --per-category 25 --refresh

The script is polite to the API (single-threaded, bounded requests, descriptive
User-Agent) and idempotent (existing files are skipped unless --refresh is passed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = PROJECT_ROOT / "ml" / "data" / "ood_corpus"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "PhytoSignal-OOD-Corpus/1.0 (academic research; contact: project maintainer) "
    "python-urllib"
)

# One entry per category the safety requirements name explicitly.
SEARCH_PLANS: dict[str, list[str]] = {
    "person": [
        "portrait photograph woman face",
        "portrait photograph man face",
        "smiling person photograph indoor",
        "selfie photograph person",
        "man wearing glasses photograph",
    ],
    "animal": [
        "dog photograph pet",
        "cat photograph pet",
        "horse standing photograph",
        "cow in field photograph",
        "bird perched photograph",
    ],
    "building": [
        "office building facade windows",
        "apartment building exterior photograph",
        "house exterior photograph",
        "skyscraper photograph city",
        "church building exterior photograph",
    ],
    "document": [
        "scanned document page text",
        "typewritten letter page",
        "printed newspaper page scan",
        "invoice document scan",
        "book page printed text",
    ],
    "object": [
        "coffee mug on table photograph",
        "wooden chair photograph",
        "bicycle photograph street",
        "laptop computer photograph desk",
        "pair of shoes photograph",
    ],
    "graphic": [
        "bar chart statistics",
        "pie chart diagram",
        "flowchart diagram",
        "technical drawing blueprint",
        "movie poster graphic design",
    ],
}

# Titles containing these words are excluded: the corpus must not contain plant tissue,
# because every file is used as a *negative* (leaf-free) example.
PLANT_TITLE_PATTERN = re.compile(
    r"leaf|leaves|foliage|plant|flower|tree|grass|garden|botanic|herbarium|fern|moss|"
    r"forest|seedling|fruit|vegetable|orchard|vineyard|crop|maize|corn|tomato|potato|"
    r"pepper|cucumber|orchid|succulent|cactus",
    re.IGNORECASE,
)

LICENCE_PATTERN = re.compile(
    r"^(cc0|public domain|pd[- ]|pd$|cc[- ]by(-sa)?[- ]?\d?(\.\d)?$|cc[- ]by(-sa)?\b)",
    re.IGNORECASE,
)

ACCEPTED_MIME = {"image/jpeg", "image/png"}


def api_get(params: dict, attempts: int = 4) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    delay = 2.0
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def search_files(query: str, limit: int = 40) -> list[dict]:
    """Returns candidate files for one Commons search query."""
    payload = api_get(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "640",
        }
    )
    pages = payload.get("query", {}).get("pages", [])
    results = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata", {})
        licence = strip_html(metadata.get("LicenseShortName", {}).get("value", ""))
        results.append(
            {
                "title": page.get("title", ""),
                "mime": info.get("mime", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "thumb": info.get("thumburl", ""),
                "description_url": info.get("descriptionurl", ""),
                "licence": licence,
                "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
            }
        )
    return results


def slugify(title: str) -> str:
    stem = title.split(":", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:60] or "file"


def safe_print(message: str) -> None:
    """Prints titles containing non-cp1252 characters without crashing the run."""
    encoding = sys.stdout.encoding or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding))


def download(url: str, destination: Path, attempts: int = 5) -> str:
    """Downloads one file, honouring the API's rate limit with exponential backoff."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts:
                raise
            print(f"    ... rate limited, retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=24)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": []}
    if MANIFEST_PATH.exists() and not args.refresh:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def save_manifest() -> None:
        """Persists provenance incrementally so an interrupted run loses nothing."""
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["licence_policy"] = (
            "Only Public domain, CC0, CC BY and CC BY-SA files were accepted. The per-file "
            "source, licence and author are recorded above; attribution requirements are "
            "satisfied by retaining this manifest alongside the images."
        )
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    known_titles = {entry["title"] for entry in manifest["files"]}
    seen_hashes = {entry["sha256"] for entry in manifest["files"]}

    for category, queries in SEARCH_PLANS.items():
        target_dir = CORPUS_ROOT / category
        target_dir.mkdir(parents=True, exist_ok=True)
        collected = sum(1 for entry in manifest["files"] if entry["category"] == category)
        print(f"[{category}] starting with {collected}/{args.per_category}")

        for query in queries:
            if collected >= args.per_category:
                break
            try:
                candidates = search_files(query)
            except Exception as exc:  # network hiccup: skip this query, keep going
                print(f"  ! search failed for {query!r}: {exc}")
                continue
            for candidate in candidates:
                if collected >= args.per_category:
                    break
                title = candidate["title"]
                if title in known_titles or not candidate["thumb"]:
                    continue
                if candidate["mime"] not in ACCEPTED_MIME:
                    continue
                if candidate["width"] < 500:
                    continue
                if PLANT_TITLE_PATTERN.search(title):
                    continue
                if not LICENCE_PATTERN.match(candidate["licence"].strip()):
                    continue

                destination = target_dir / f"{slugify(title)}.jpg"
                if destination.exists() and not args.refresh:
                    continue
                try:
                    digest = download(candidate["thumb"], destination)
                except Exception as exc:
                    safe_print(f"  ! download failed for {title!r}: {exc}")
                    continue
                if digest in seen_hashes:
                    destination.unlink()
                    continue

                seen_hashes.add(digest)
                known_titles.add(title)
                manifest["files"].append(
                    {
                        "category": category,
                        "title": title,
                        "path": destination.relative_to(PROJECT_ROOT).as_posix(),
                        "source": candidate["description_url"],
                        "licence": candidate["licence"],
                        "author": candidate["artist"][:200],
                        "sha256": digest,
                        "query": query,
                    }
                )
                collected += 1
                safe_print(f"  + {destination.name}  [{candidate['licence']}]")
                save_manifest()
                time.sleep(0.7)

        if collected < args.per_category:
            print(f"  ! only {collected} files for {category}")

    save_manifest()

    counts: dict[str, int] = {}
    for entry in manifest["files"]:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    print("\nCorpus totals:", counts, "| total", sum(counts.values()))
    print("Manifest:", MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
