"""Split WhisperX segments into short-form or long-form subtitle chunks."""

from __future__ import annotations

import re
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

# Default ceiling for characters-per-second. 21 cps is a safe general-purpose
# value; documentary / kids content often targets 15-17 cps, fast speech up to
# ~25 cps. Users can override with --max-cps.
DEFAULT_MAX_CPS = 21.0

# Default minimum on-screen time per subtitle, in seconds. Sub-1s subtitles
# are typically unreadable.
DEFAULT_MIN_DURATION = 1.0

# Minimum gap left between two extended subtitles so they don't overlap.
MIN_GAP_SECONDS = 0.05

# Default pause threshold for the context-aware sentence splitter. A gap
# between a period-ending word and the next word of at least this many
# seconds is treated as a real sentence boundary. 0.2s catches a small but
# meaningful breath — the existing ``PRESET_TARGETS`` pause threshold of
# 0.45s handles the bigger pauses, this is for the in-between range.
SENTENCE_PAUSE_THRESHOLD = 0.2

# Hallucination patterns produced by Whisper / WhisperX. Matched
# case-insensitively and stripped (along with surrounding whitespace) from the
# segment text. Order matters: longer phrases are tried before their prefixes.
_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsubtitles?\s+by\s+[^\n\r.]+", re.IGNORECASE),
    re.compile(r"\btranscribed\s+by\s+[^\n\r.]+", re.IGNORECASE),
    re.compile(r"\bthanks?\s+for\s+watching\b\.?", re.IGNORECASE),
    re.compile(r"\bthank\s+you\s+for\s+watching\b\.?", re.IGNORECASE),
    re.compile(r"\bthank\s+you\s+for\s+listening\b\.?", re.IGNORECASE),
    re.compile(r"\bplease\s+subscribe\b\.?", re.IGNORECASE),
    re.compile(r"\bsee\s+you\s+in\s+the\s+next\s+(video|one)\b\.?", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:\[|\()\s*(music|applause|laughter|inaudible|crosstalk)\s*(?:\]|\))", re.IGNORECASE),
    re.compile(r"^\s*♪+\s*|\s*♪+\s*$"),
    re.compile(r"\.{3,}"),
)

# Characters that mark the end of a clause / sentence. Used by the
# clause-aware breaker to prefer a natural pause over a mid-clause split.
_CLAUSE_TERMINATORS = set(",.!?;:–—…)]\"'")

# Characters that end a sentence (hard split). Only these trigger the
# sentence-boundary breaker in :func:`_group_words`.
_SENTENCE_TERMINATORS = set(".!?")

# Lower-cased abbreviations that look like sentence terminators but should
# NOT trigger a split. Compared against the lowercased, stripped, and
# trailing-period-stripped form of the word.
#
# This list is intentionally short and hand-curated. Adding more entries
# is safe; missing an entry just means a name like "St. Petersburg" gets
# split (recoverable with --no-sentence-split).
_SENTENCE_ABBREVIATIONS: frozenset[str] = frozenset({
    # Titles / honorifics
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    # Latin / common shorthand
    "vs", "etc", "e.g", "i.e", "cf", "al",
    # Common US/UK abbreviations
    "u.s", "u.s.a", "u.k", "ph.d", "m.d", "a.m", "p.m",
    "vol", "inc", "ltd", "co",
    # Single-letter initials (A. B. Smith, J. R. R. Tolkien, etc.)
    *{chr(c) for c in range(ord("a"), ord("z") + 1)},
    *{chr(c) for c in range(ord("A"), ord("Z") + 1)},
})


def _is_sentence_terminator(word: str) -> bool:
    """Return True if ``word`` ends a sentence and should trigger a split.

    A word counts as a sentence terminator when its stripped form ends with
    ``.``, ``!`` or ``?`` and the word (with the trailing punctuation
    stripped) is not a known abbreviation.
    """
    stripped = word.rstrip()
    if not stripped or stripped[-1] not in _SENTENCE_TERMINATORS:
        return False
    # Strip the trailing terminator for the abbreviation check. We only
    # care about whether the *core* of the word is a known abbreviation,
    # not the punctuation.
    core = stripped.rstrip(".!?")
    return core.lower() not in _SENTENCE_ABBREVIATIONS


def _sanitize_text(text: str) -> str:
    """Return a cleaned version of the text for display.

    Collapses runs of whitespace into single spaces and strips the ends.
    Newlines are flattened to spaces — use :func:`_sanitize_lines` if the
    text contains intentional line breaks.
    """
    return " ".join(text.split())


def _sanitize_lines(text: str) -> str:
    """Clean a multi-line subtitle while preserving the line breaks.

    Runs of whitespace inside each line are collapsed; empty lines are
    dropped. The number of non-empty lines is returned unchanged.
    """
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _is_clause_boundary(word: str) -> bool:
    """Return True if ``word`` ends at a clause / sentence boundary.

    A word is treated as a clause boundary when its stripped form ends with one
    of ``_CLAUSE_TERMINATORS``. A trailing newline / whitespace is ignored.
    """
    stripped = word.rstrip()
    return bool(stripped) and stripped[-1] in _CLAUSE_TERMINATORS


def _strip_hallucinations(text: str) -> str:
    """Remove common WhisperX hallucinated text and music tags.

    The cleanup is intentionally conservative: it only matches well-known
    filler phrases and bracketed audio tags. Genuine transcript text that
    happens to contain a hallucinated phrase (e.g. an outro that really is
    "Thanks for watching") will also be removed — callers that want to keep
    the original text can pass ``cleanup=False`` to :func:`split_segments`.

    When the input contains a newline (i.e. a balanced two-line subtitle),
    the line structure is preserved.
    """
    had_newline = "\n" in text
    cleaned = text
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if had_newline:
        return _sanitize_lines(cleaned)
    return _sanitize_text(cleaned)


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
    use_clause_boundaries: bool = True,
    split_at_sentences: bool = True,
    sentence_pause_threshold: float = SENTENCE_PAUSE_THRESHOLD,
) -> list[list[dict]]:
    """Group timed words into subtitle-sized chunks.

    A new chunk is started when any of the following is true:
    - the current chunk already has ``max_words`` words,
    - the joined text would reach ``max_chars`` characters,
    - the pause between the current and previous word exceeds
      ``pause_threshold`` seconds,
    - the speaker changes,
    - (optional) the chunk exceeded its thresholds and a clause boundary is
      found by walking back from the current word,
    - (optional) a sentence-ending word (``!``/``?``/``.``) is found in the
      middle of the current chunk. The chunk is split at the latest such
      word, but only when at least one of these context signals also
      agrees:
        * the next word starts with an uppercase letter,
        * the gap between the period-ending word and the next word is at
          least ``sentence_pause_threshold`` seconds (a small breath),
        * the next word belongs to a different speaker.
      Without any of those signals the period is treated as a mid-thought
      pause and the chunk is kept whole. Known abbreviations (``Mr.``,
      ``Dr.``, ``U.S.``, ``e.g.``, single letters, …) are skipped before
      any of this runs.

    The clause-aware breaker only kicks in when the chunk is *already* over
    the word/char/pause threshold — it just picks a more natural break point
    than the last word. When ``use_clause_boundaries`` is ``False`` the
    behavior is identical to the original implementation.

    When ``split_at_sentences`` is ``False`` the sentence check is skipped
    entirely (legacy behavior).
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    def _flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in words:
        # Start a new group on speaker change.
        if current and current[-1].get("speaker") != word.get("speaker"):
            _flush()

        current.append(word)

        # Context-aware sentence-boundary split: if the chunk has a
        # sentence terminator in any position other than the very last
        # word, consider splitting at the latest such terminator. The
        # split is only committed when at least one supporting signal
        # is present (capital letter after, meaningful pause after, or a
        # speaker change at the boundary). This keeps mid-thought
        # periods ("I found a bug. reports say...") from prematurely
        # breaking the chunk.
        if split_at_sentences and len(current) > 1:
            for i in range(len(current) - 2, -1, -1):
                if not _is_sentence_terminator(current[i]["word"]):
                    continue
                next_word = current[i + 1]
                gap_after = next_word["start"] - current[i]["end"]
                starts_capital = next_word["word"][:1].isupper()
                speaker_changed = (
                    current[i].get("speaker") is not None
                    and next_word.get("speaker") is not None
                    and current[i]["speaker"] != next_word["speaker"]
                )
                if (
                    gap_after >= sentence_pause_threshold
                    or starts_capital
                    or speaker_changed
                ):
                    head = current[: i + 1]
                    tail = current[i + 1 :]
                    current = head
                    _flush()
                    current = tail
                    break

        text = " ".join(w["word"].strip() for w in current)

        pause = False
        if len(current) >= 2:
            gap = current[-1]["start"] - current[-2]["end"]
            if gap > pause_threshold:
                pause = True

        over_words = len(current) >= max_words
        over_chars = len(text) >= max_chars

        if not (over_words or over_chars or pause):
            continue

        if use_clause_boundaries and (over_words or over_chars) and len(current) > 1:
            # Look back from the second-to-last word for the last clause
            # boundary. If we find one, split immediately after it instead of
            # after the current word so we don't end a line mid-clause.
            split_at = 0
            for i in range(len(current) - 2, 0, -1):
                if _is_clause_boundary(current[i]["word"]):
                    split_at = i + 1
                    break
            if split_at > 0 and split_at < len(current):
                tail = current[split_at:]
                head = current[:split_at]
                current = head
                _flush()
                current = tail
                continue

        _flush()

    _flush()

    return groups


def _balance_lines(text: str, max_line: int = 36) -> str:
    """Split ``text`` into two lines at the clause boundary closest to the midpoint.

    Returns the text unchanged when it's already short enough, when there's
    no clause boundary, or when ``max_line`` is ``0`` / negative (disabled).
    The SRT format supports embedded newlines inside the text field, so the
    caller can pass the result through unchanged.
    """
    if max_line <= 0 or len(text) <= max_line:
        return text

    best_index = -1
    best_distance = len(text) + 1
    midpoint = len(text) / 2.0

    for i, ch in enumerate(text):
        if ch not in _CLAUSE_TERMINATORS:
            continue
        # Prefer to break *after* the terminator, so leave a small bias.
        candidate = i + 1
        distance = abs(candidate - midpoint)
        if distance < best_distance:
            best_distance = distance
            best_index = candidate

    if best_index <= 0 or best_index >= len(text):
        return text

    return text[:best_index] + "\n" + text[best_index:]


def _enforce_cps(
    groups: list[dict],
    max_cps: float,
    min_gap: float = MIN_GAP_SECONDS,
) -> list[dict]:
    """Extend each group's ``end`` so its characters-per-second is at most ``max_cps``.

    The end time is clamped to ``next_group.start - min_gap`` so an extension
    never overlaps the following subtitle. Returns the same list with
    ``end`` values updated in place. When ``max_cps`` is ``0`` or negative
    the function is a no-op.
    """
    if max_cps <= 0 or not groups:
        return groups

    for i, group in enumerate(groups):
        text = str(group.get("text", ""))
        if not text:
            continue
        # Newlines don't add reading time, so don't count them.
        chars = len(text.replace("\n", ""))
        duration = float(group["end"]) - float(group["start"])
        if duration <= 0:
            continue
        current_cps = chars / duration
        if current_cps <= max_cps:
            continue
        min_duration = chars / max_cps
        new_end = float(group["start"]) + min_duration
        if i + 1 < len(groups):
            ceiling = float(groups[i + 1]["start"]) - min_gap
            if new_end > ceiling:
                new_end = ceiling
        if new_end > float(group["end"]):
            group["end"] = new_end

    return groups


def _enforce_min_duration(
    groups: list[dict],
    min_duration: float,
    min_gap: float = MIN_GAP_SECONDS,
) -> list[dict]:
    """Extend each group's ``end`` so it stays on screen for at least ``min_duration``.

    The end time is clamped to ``next_group.start - min_gap`` so an extension
    never overlaps the following subtitle. Returns the same list with
    ``end`` values updated in place. When ``min_duration`` is ``0`` or
    negative the function is a no-op.
    """
    if min_duration <= 0 or not groups:
        return groups

    for i, group in enumerate(groups):
        duration = float(group["end"]) - float(group["start"])
        if duration >= min_duration:
            continue
        new_end = float(group["start"]) + min_duration
        if i + 1 < len(groups):
            ceiling = float(groups[i + 1]["start"]) - min_gap
            if new_end > ceiling:
                new_end = ceiling
        if new_end > float(group["end"]):
            group["end"] = new_end

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
    *,
    cleanup: bool = True,
    use_clause_boundaries: bool = True,
    split_at_sentences: bool = True,
    sentence_pause_threshold: float = SENTENCE_PAUSE_THRESHOLD,
    balance_lines: bool | None = None,
    max_cps: float = DEFAULT_MAX_CPS,
    min_duration: float = DEFAULT_MIN_DURATION,
    enforce_cps: bool = True,
    enforce_min_duration: bool = True,
) -> list[dict]:
    """Split WhisperX segments according to the chosen preset.

    Parameters
    ----------
    segments:
        WhisperX segments with ``start``, ``end``, ``text`` and optionally
        per-word timings.
    preset:
        ``shortform`` or ``longform``.
    cleanup:
        Strip common WhisperX hallucinated text (default: ``True``).
    use_clause_boundaries:
        Prefer splitting at clause boundaries over mid-clause breaks
        (default: ``True``).
    split_at_sentences:
        Always split at sentence boundaries (``!``/``?``/``.``) even when
        the chunk is well under the word/char limits, but only when the
        surrounding context (capital after, pause after, or speaker change)
        supports the split. Known abbreviations (``Mr.``, ``Dr.``,
        ``U.S.``, ``e.g.``, single letters) are skipped (default:
        ``True``).
    sentence_pause_threshold:
        Minimum gap (in seconds) between a period-ending word and the
        next word for the gap to count as supporting evidence of a real
        sentence boundary. Default: ``0.2``. Set to ``0`` to require a
        capital letter or speaker change instead.
    balance_lines:
        Insert a newline in long chunks to balance two-line display. When
        ``None`` (default), the longform preset enables it and the shortform
        preset disables it.
    max_cps:
        Maximum characters-per-second allowed per subtitle. Used only when
        ``enforce_cps`` is ``True`` (default: 21.0). Set to ``0`` to disable.
    min_duration:
        Minimum on-screen duration in seconds. Used only when
        ``enforce_min_duration`` is ``True`` (default: 1.0). Set to ``0``
        to disable.
    enforce_cps:
        Run the CPS post-processor (default: ``True``).
    enforce_min_duration:
        Run the minimum-duration post-processor (default: ``True``).

    Returns
    -------
    A flat list of segment dicts suitable for writing to SRT.
    """
    if preset not in VALID_PRESETS:
        valid = ", ".join(sorted(VALID_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {valid}")

    max_words, max_chars, pause_threshold = PRESET_TARGETS[preset]
    segments = list(segments)

    if balance_lines is None:
        balance_lines = preset == PRESET_LONGFORM

    words = _collect_words(segments)

    if words:
        groups = _group_words(
            words,
            max_words,
            max_chars,
            pause_threshold,
            use_clause_boundaries=use_clause_boundaries,
            split_at_sentences=split_at_sentences,
            sentence_pause_threshold=sentence_pause_threshold,
        )
        output: list[dict] = [
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
    else:
        # No timed words available — fall back to splitting by text.
        output = []
        for segment in segments:
            output.extend(_fallback_split_segment(segment, max_words))

    if balance_lines:
        for item in output:
            item["text"] = _sanitize_lines(_balance_lines(item["text"]))

    if cleanup:
        for item in output:
            cleaned = _strip_hallucinations(item["text"])
            item["text"] = _sanitize_lines(cleaned) if "\n" in item["text"] else _sanitize_text(cleaned)

    # Drop any group that became empty after cleanup.
    output = [item for item in output if item.get("text", "").strip()]

    if enforce_cps:
        _enforce_cps(output, max_cps=max_cps)
    if enforce_min_duration:
        _enforce_min_duration(output, min_duration=min_duration)

    return output
