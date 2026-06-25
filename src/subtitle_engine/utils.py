"""CLI helpers and path utilities."""

from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}


def resolve_output_path(input_path: Path, output: Optional[Path] = None) -> Path:
    """Resolve the SRT output path.

    If ``output`` is provided, use it. Otherwise create ``<input>.srt``
    next to the input file.
    """
    if output:
        return output
    return input_path.with_suffix(".srt")


def validate_media_file(path: Path) -> None:
    """Raise a ValueError if the path does not look like a media file."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        joined = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {joined}"
        )
