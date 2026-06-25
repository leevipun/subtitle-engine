"""Convert transcription segments to SRT format."""

import re
from pathlib import Path
from typing import Iterable


def _format_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm."""
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3_600_000
    minutes = (total_millis % 3_600_000) // 60_000
    secs = (total_millis % 60_000) // 1_000
    millis = total_millis % 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_segment(index: int, start: float, end: float, text: str) -> str:
    """Format a single segment as an SRT block."""
    cleaned_text = text.strip()
    if not cleaned_text:
        cleaned_text = "..."
    return f"{index}\n{_format_time(start)} --> {_format_time(end)}\n{cleaned_text}\n"


def segments_to_srt(segments: Iterable[dict]) -> str:
    """Build an SRT string from WhisperX-style segments.

    Each segment is expected to be a dict with keys:
    ``start`` (float), ``end`` (float), and ``text`` (str).
    """
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment["text"])
        blocks.append(_format_segment(index, start, end, text))
    return "\n".join(blocks)


def extract_text_from_srt(path: Path) -> str:
    """Read an SRT file and return the spoken text as a single string.

    Parameters
    ----------
    path:
        Path to the SRT file to read.

    Returns
    -------
    The transcript text with subtitle lines joined by spaces.

    Raises
    ------
    FileNotFoundError:
        If the SRT file does not exist.
    ValueError:
        If no subtitle text can be extracted from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())

    texts: list[str] = []
    for block in blocks:
        lines = block.strip().splitlines()
        # A valid block has at least an index, a timecode line, and one text line.
        if len(lines) < 3:
            continue
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        if text:
            texts.append(text)

    if not texts:
        raise ValueError(f"No subtitle text found in {path}")

    return " ".join(texts)


def write_srt(segments: Iterable[dict], output_path: Path) -> None:
    """Write segments to an SRT file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(segments_to_srt(segments), encoding="utf-8")
