"""Split WhisperX segments into short-form or long-form subtitle chunks."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


PRESET_SHORTFORM = "shortform"
PRESET_LONGFORM = "longform"
VALID_PRESETS = {PRESET_SHORTFORM, PRESET_LONGFORM}

# (max_words, max_chars, pause_threshold_seconds)
PRESET_TARGETS = {
    PRESET_SHORTFORM: (4, 22, 0.45),
    PRESET_LONGFORM: (14, 80, 0.45),
}


def _sanitize_text(text: str) -> str:
    """Return a cleaned version of the text for display."""
    return " ".join(text.split())


def _collect_words(segments: Iterable[dict]) -> list[dict]:
    """Collect all timed words from WhisperX segments in order."""
    words: list[dict] = []
    for segment in segments:
        for word in segment.get("words", []):
            if not isinstance(word, dict):
                continue
            word_text = word.get("word", "").strip()
            if not word_text:
                continue
            if "start" not in word or "end" not in word:
                continue
            words.append({
                "word": word_text,
                "start": float(word["start"]),
                "end": float(word["end"]),
                "speaker": word.get("speaker"),
            })
    return words


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


def _group_words(
    words: list[dict],
    max_words: int,
    max_chars: int,
    pause_threshold: float,
) -> list[list[dict]]:
    """Group timed words into subtitle-sized chunks.

    A new chunk is started when any of the following is true:
    - the current chunk already has ``max_words`` words,
    - the joined text would reach ``max_chars`` characters,
    - the pause between the current and previous word exceeds
      ``pause_threshold`` seconds,
    - the speaker changes.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    for word in words:
        # Start a new group on speaker change.
        if current and current[-1].get("speaker") != word.get("speaker"):
            groups.append(current)
            current = []

        current.append(word)

        text = " ".join(w["word"].strip() for w in current)

        pause = False
        if len(current) >= 2:
            gap = current[-1]["start"] - current[-2]["end"]
            if gap > pause_threshold:
                pause = True

        if len(current) >= max_words or len(text) >= max_chars or pause:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _fallback_split_segment(segment: dict, max_words: int) -> list[dict]:
    """Split a segment without per-word timings by text alone."""
    text = _sanitize_text(str(segment.get("text", "")))
    if not text:
        return []

    tokens = text.split()
    if len(tokens) <= max_words:
        return [{
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": text,
        }]

    chunks: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if len(current) >= max_words:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    segment_start = float(segment.get("start", 0.0))
    segment_end = float(segment.get("end", segment_start))
    duration = segment_end - segment_start
    result = []
    for i, chunk in enumerate(chunks):
        start = segment_start + duration * (i / len(chunks))
        end = segment_start + duration * ((i + 1) / len(chunks))
        result.append({
            "start": start,
            "end": end,
            "text": " ".join(chunk),
        })
    return result


def split_segments(
    segments: Iterable[dict],
    preset: str = PRESET_SHORTFORM,
) -> list[dict]:
    """Split WhisperX segments according to the chosen preset.

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

    max_words, max_chars, pause_threshold = PRESET_TARGETS[preset]
    segments = list(segments)

    words = _collect_words(segments)

    if words:
        groups = _group_words(words, max_words, max_chars, pause_threshold)
        return [
            {
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": _prefix_speaker(
                    _sanitize_text(" ".join(w["word"].strip() for w in group)),
                    _dominant_speaker(group),
                ),
            }
            for group in groups
        ]

    # No timed words available — fall back to splitting by text.
    output: list[dict] = []
    for segment in segments:
        output.extend(_fallback_split_segment(segment, max_words))
    return output
