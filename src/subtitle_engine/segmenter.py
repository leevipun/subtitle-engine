"""Split WhisperX segments into shorter or longer subtitle chunks."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


PRESET_SHORTFORM = "shortform"
PRESET_LONGFORM = "longform"
VALID_PRESETS = {PRESET_SHORTFORM, PRESET_LONGFORM}

# Word-count targets per subtitle block.
PRESET_TARGETS = {
    PRESET_SHORTFORM: (2, 5),   # min, max
    PRESET_LONGFORM: (10, 14),  # min, max
}


def _sanitize_text(text: str) -> str:
    """Return a cleaned version of the text for display."""
    return " ".join(text.split())


def _words_from_segment(segment: dict) -> list[dict]:
    """Extract a clean list of word dicts from a WhisperX segment.

    Each word dict should have ``word`` and optionally ``start``/``end``.
    """
    raw_words = segment.get("words", [])
    words = []
    for word_entry in raw_words:
        if isinstance(word_entry, dict):
            word_text = word_entry.get("word", "").strip()
        else:
            word_text = str(word_entry).strip()
        if word_text:
            words.append({"word": word_text, **word_entry} if isinstance(word_entry, dict) else {"word": word_text})
    return words


def _split_text_evenly(text: str, chunk_count: int) -> list[str]:
    """Split text into ``chunk_count`` roughly equal word groups."""
    tokens = text.split()
    if chunk_count <= 1 or len(tokens) <= chunk_count:
        return [text]

    base_size, remainder = divmod(len(tokens), chunk_count)
    chunks = []
    index = 0
    for i in range(chunk_count):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(" ".join(tokens[index : index + size]))
        index += size
    return chunks


def _dominant_speaker(words: list[dict]) -> str | None:
    """Return the most common speaker label among the given words, if any."""
    speakers = [w.get("speaker") for w in words if w.get("speaker")]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


def _prefix_speaker(text: str, speaker: str | None) -> str:
    """Prefix a speaker label to text when one is known."""
    if not speaker:
        return text
    return f"[{speaker}] {text}"


def _split_segment(
    segment: dict,
    min_words: int,
    max_words: int,
) -> list[dict]:
    """Split a single WhisperX segment into subtitle-sized chunks.

    Word-level timings are used when available. If not, the segment's total
    duration is divided proportionally among the chunks.
    """
    words = _words_from_segment(segment)
    segment_start = float(segment.get("start", 0.0))
    segment_end = float(segment.get("end", segment_start))

    if not words:
        cleaned = _sanitize_text(str(segment.get("text", "")))
        if cleaned:
            return [{"start": segment_start, "end": segment_end, "text": cleaned}]
        return []

    # Build chunks based on word count targets.
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        # Merge a tiny trailing chunk with the previous one if possible.
        if len(current_chunk) < min_words and chunks:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)

    # Resolve timings per chunk.
    result = []
    for chunk in chunks:
        text_words = [w["word"].strip() for w in chunk]
        text = _sanitize_text(" ".join(text_words))
        if not text:
            continue

        timed_words = [w for w in chunk if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None]
        if timed_words:
            start = float(timed_words[0]["start"])
            end = float(timed_words[-1]["end"])
        else:
            # Fallback: divide the segment duration proportionally.
            ratio = max(1, len(chunk)) / max(1, len(words))
            duration = segment_end - segment_start
            chunk_index = chunks.index(chunk)
            start = segment_start + duration * (chunk_index / len(chunks))
            end = segment_start + duration * ((chunk_index + 1) / len(chunks))

        speaker = _dominant_speaker(chunk)
        text = _prefix_speaker(text, speaker)
        result.append({"start": start, "end": end, "text": text})

    return result


def split_segments(
    segments: Iterable[dict],
    preset: str = PRESET_SHORTFORM,
) -> list[dict]:
    """Split or join WhisperX segments according to the chosen preset.

    Parameters
    ----------
    segments:
        WhisperX segments with ``start``, ``end``, ``text`` and optionally
        per-word timings.
    preset:
        ``shortform`` or ``longform``.

    Returns
    -------
    A flat list of segment dicts suitable for writing to SRT.
    """
    if preset not in VALID_PRESETS:
        valid = ", ".join(sorted(VALID_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {valid}")

    min_words, max_words = PRESET_TARGETS[preset]

    output: list[dict] = []
    for segment in segments:
        output.extend(_split_segment(segment, min_words, max_words))

    return output
