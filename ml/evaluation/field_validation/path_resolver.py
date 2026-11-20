"""
Authoritative Safe Image Path Resolver
=======================================
Part of PHYTOSIGNAL 2.0 - Evaluation Integrity & Reliability Hardening.

Guarantees:
1. Paths are resolved relative to a declared dataset root.
2. Directory traversal attempts ('..') are strictly rejected.
3. Absolute paths are rejected by default. If explicitly allowed, they must be contained in the dataset root.
4. Symlinks are fully resolved before containment checks; escaping symlinks are rejected.
5. Missing files and non-regular files (directories, sockets, FIFOs) are rejected.
6. Error messages are informative and actionable without leaking arbitrary host filesystem details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


class PathSecurityError(ValueError):
    """Raised when an image path violates security or containment boundaries."""
    pass


class InvalidImageFileError(ValueError):
    """Raised when a resolved path is not a valid regular file."""
    pass


def resolve_safe_image_path(
    file_path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    allow_absolute: bool = False,
) -> Path:
    """
    Safely resolves and validates an image file path against a dataset root.

    Args:
        file_path: The file path specified in manifest or sample record.
        dataset_root: The declared root directory of the evaluation dataset.
        allow_absolute: If False (default), absolute paths are rejected.
                        If True, absolute paths are accepted ONLY IF they resolve
                        within the declared dataset root.

    Returns:
        Path: Authoritative, fully resolved Path to a regular file within dataset_root.

    Raises:
        PathSecurityError: If directory traversal, escaping symlinks, or unauthorized absolute paths are detected.
        FileNotFoundError: If the target file does not exist.
        InvalidImageFileError: If the target exists but is a directory or non-regular file.
    """
    if file_path is None or str(file_path).strip() == "":
        raise PathSecurityError("Image file path cannot be empty or None.")

    raw_path_str = str(file_path).strip()
    raw_path = Path(raw_path_str)

    # Resolve dataset root
    if dataset_root is None:
        if raw_path.is_absolute():
            root_resolved = raw_path.parent.resolve()
            allow_absolute = True
        else:
            root_resolved = Path.cwd().resolve()
    else:
        root_resolved = Path(dataset_root).resolve()

    if not root_resolved.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")
    if not root_resolved.is_dir():
        raise InvalidImageFileError(f"Dataset root must be a directory: {dataset_root}")

    # Check for raw directory traversal tokens ('..')
    if ".." in raw_path.parts:
        raise PathSecurityError(
            f"Directory traversal detected in path '{raw_path_str}'. "
            f"Parent directory traversal ('..') is strictly prohibited."
        )

    # Check absolute path policy
    if raw_path.is_absolute():
        if not allow_absolute:
            raise PathSecurityError(
                f"Absolute path '{raw_path_str}' is prohibited. "
                f"Manifest image paths must be relative to dataset root."
            )
        candidate = raw_path
    else:
        candidate = root_resolved / raw_path

    # Check if candidate exists before symlink resolution
    if not candidate.exists():
        raise FileNotFoundError(f"Image file does not exist: {raw_path_str}")

    # Fully resolve symlinks and real path
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except Exception as exc:
        raise FileNotFoundError(f"Could not resolve image path '{raw_path_str}': {exc}") from exc

    # Enforce dataset root containment on fully resolved path
    try:
        resolved_candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise PathSecurityError(
            f"Path containment violation: '{raw_path_str}' resolves outside "
            f"the declared dataset root (symlink escape or directory breakout)."
        ) from exc

    # Ensure resolved path is a regular file
    if not resolved_candidate.is_file():
        if resolved_candidate.is_dir():
            raise InvalidImageFileError(
                f"Expected regular file but found directory at: '{raw_path_str}'"
            )
        raise InvalidImageFileError(
            f"Path is not a regular file (special device, socket, or FIFO): '{raw_path_str}'"
        )

    return resolved_candidate
