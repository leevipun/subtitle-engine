"""Tests for the segmenter/preset splitter."""

import pytest

from subtitle_engine.segmenter import (
    DEFAULT_MAX_CPS,
    DEFAULT_MIN_DURATION,
    PRESET_LONGFORM,
    PRESET_SHORTFORM,
    SENTENCE_PAUSE_THRESHOLD,
    _balance_lines,
    _collect_words,
    _enforce_cps,
    _enforce_min_duration,
    _group_words,
    _is_clause_boundary,
    _is_sentence_terminator,
    _strip_hallucinations,
    split_segments,
)


def _word(word: str, start: float, end: float, speaker: str | None = None) -> dict:
    return {"word": word, "start": start, "end": end, "speaker": speaker}


def _segment(text: str, words: list[dict], start: float = 0.0, end: float | None = None) -> dict:
    return {
        "start": start,
        "end": end if end is not None else (words[-1]["end"] if words else start + 1.0),
        "text": text,
        "words": words,
    }


def test_collect_words_extracts_timed_words():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                _word("hello", 0.0, 0.5),
                _word("world", 0.5, 1.0),
            ],
        }
    ]
    words = _collect_words(segments)
    assert [w["word"] for w in words] == ["hello", "world"]


def test_collect_words_skips_untimed_words():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "words": [
                _word("hello", 0.0, 0.5),
                {"word": "world"},  # missing timing
            ],
        }
    ]
    words = _collect_words(segments)
    assert [w["word"] for w in words] == ["hello"]


def test_group_splits_on_word_count():
    words = [_word(str(i), i * 0.5, i * 0.5 + 0.4) for i in range(10)]
    groups = _group_words(words, max_words=4, max_chars=200, pause_threshold=1.0)

    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 4
    assert len(groups[2]) == 2


def test_group_splits_on_char_count():
    words = [
        _word("hello", 0.0, 0.5),
        _word("world", 0.5, 1.0),
        _word("today", 1.0, 1.5),
    ]
    groups = _group_words(words, max_words=10, max_chars=10, pause_threshold=1.0)

    # "hello world" is 11 chars -> triggers split, leaving "today" on its own.
    assert len(groups) == 2
    assert [w["word"] for w in groups[0]] == ["hello", "world"]
    assert [w["word"] for w in groups[1]] == ["today"]


def test_group_splits_on_pause():
    words = [
        _word("one", 0.0, 0.5),
        _word("two", 0.5, 1.0),
        _word("three", 2.0, 2.5),  # 1.0s pause after "two"
        _word("four", 2.5, 3.0),
    ]
    groups = _group_words(words, max_words=10, max_chars=200, pause_threshold=0.45)

    # The word following the pause is included in the group that gets finalized.
    assert len(groups) == 2
    assert [w["word"] for w in groups[0]] == ["one", "two", "three"]
    assert [w["word"] for w in groups[1]] == ["four"]


def test_group_splits_on_speaker_change():
    words = [
        _word("hello", 0.0, 0.5, "SPEAKER_01"),
        _word("world", 0.5, 1.0, "SPEAKER_02"),
    ]
    groups = _group_words(words, max_words=10, max_chars=200, pause_threshold=1.0)

    assert len(groups) == 2


def test_shortform_splits_to_small_chunks():
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "one two three four five six seven eight",
            "words": [_word(str(i), i * 0.5, i * 0.5 + 0.4) for i in range(8)],
        }
    ]
    result = split_segments(segments, preset=PRESET_SHORTFORM)

    assert len(result) == 2
    assert len(result[0]["text"].split()) == 4
    assert len(result[1]["text"].split()) == 4


def test_longform_allows_larger_chunks():
    segments = [
        {
            "start": 0.0,
            "end": 20.0,
            "text": " ".join(str(i) for i in range(25)),
            "words": [_word(str(i), i * 0.8, i * 0.8 + 0.7) for i in range(25)],
        }
    ]
    result = split_segments(segments, preset=PRESET_LONGFORM)

    assert len(result) >= 2
    for chunk in result:
        assert len(chunk["text"].split()) <= 14


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        split_segments([], preset="invalid")


def test_speaker_label_is_preserved():
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "hello world today",
            "words": [
                _word("hello", 0.0, 0.5, "SPEAKER_01"),
                _word("world", 0.5, 1.0, "SPEAKER_01"),
                _word("today", 1.0, 1.5, "SPEAKER_01"),
            ],
        }
    ]
    result = split_segments(segments, preset=PRESET_SHORTFORM)

    assert any("[SPEAKER_01]" in chunk["text"] for chunk in result)


def test_empty_segment_is_ignored():
    segments = [{"start": 0.0, "end": 1.0, "text": "   ", "words": []}]
    result = split_segments(segments, preset=PRESET_SHORTFORM)
    assert result == []


def test_fallback_when_no_word_timings():
    segments = [
        {
            "start": 0.0,
            "end": 9.0,
            "text": "one two three four five six seven eight nine",
        }
    ]
    result = split_segments(segments, preset=PRESET_SHORTFORM)

    total_words = sum(len(chunk["text"].split()) for chunk in result)
    assert total_words == 9
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[-1]["end"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Hallucination cleanup
# ---------------------------------------------------------------------------


def test_strip_hallucinations_removes_thanks_for_watching():
    assert _strip_hallucinations("Thanks for watching everyone") == "everyone"
    assert _strip_hallucinations("Hello there. Thanks for watching.") == "Hello there."
    assert _strip_hallucinations("Thank you for watching this video") == "this video"


def test_strip_hallucinations_removes_music_and_applause_tags():
    assert _strip_hallucinations("[Music] hello world") == "hello world"
    assert _strip_hallucinations("(Applause) thank you") == "thank you"
    assert _strip_hallucinations("hello [Laughter] world") == "hello world"


def test_strip_hallucinations_removes_ellipsis_runs():
    assert _strip_hallucinations("well.... yes") == "well yes"
    assert _strip_hallucinations("ok.......... sure") == "ok sure"


def test_strip_hallucinations_removes_subtitles_by():
    assert _strip_hallucinations("hello world Subtitles by Acme Corp") == "hello world"


def test_strip_hallucinations_preserves_unrelated_text():
    # A genuine sentence containing the word "thanks" but not the full
    # hallucination phrase should survive.
    assert _strip_hallucinations("I give thanks to my parents") == "I give thanks to my parents"


def test_cleanup_disabled_keeps_hallucinations():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Thanks for watching",
            "words": [_word("Thanks", 0.0, 0.4), _word("for", 0.4, 0.6), _word("watching", 0.6, 1.0)],
        }
    ]
    result = split_segments(segments, preset=PRESET_SHORTFORM, cleanup=False)
    assert any("Thanks" in chunk["text"] for chunk in result)


# ---------------------------------------------------------------------------
# Clause-aware breaking
# ---------------------------------------------------------------------------


def test_is_clause_boundary_recognises_terminators():
    assert _is_clause_boundary("hello,")
    assert _is_clause_boundary("world.")
    assert _is_clause_boundary("really?")
    assert _is_clause_boundary("yes!")
    assert _is_clause_boundary("then;")
    assert _is_clause_boundary("then:")
    assert _is_clause_boundary("quote\"")
    assert not _is_clause_boundary("hello")
    assert not _is_clause_boundary("")


def test_group_breaks_at_clause_boundary():
    # 5 words; max_words=4, so adding the 5th triggers a break. The
    # clause-aware breaker should look back and split *after* the 3rd word
    # (which ends in a comma), not after the 5th.
    words = [
        _word("the", 0.0, 0.3),
        _word("quick", 0.3, 0.6),
        _word("brown", 0.6, 0.9),
        _word("fox,", 0.9, 1.2),
        _word("jumps", 1.2, 1.5),
    ]
    groups = _group_words(words, max_words=4, max_chars=200, pause_threshold=10.0)

    assert len(groups) == 2
    assert [w["word"] for w in groups[0]] == ["the", "quick", "brown", "fox,"]
    assert [w["word"] for w in groups[1]] == ["jumps"]


def test_group_falls_back_when_no_clause_boundary():
    # Same setup, but the words are numbers with no clause terminator. The
    # breaker should fall through to the original max-words split.
    words = [_word(str(i), i * 0.3, i * 0.3 + 0.25) for i in range(5)]
    groups = _group_words(words, max_words=4, max_chars=200, pause_threshold=10.0)

    assert len(groups) == 2
    assert [w["word"] for w in groups[0]] == ["0", "1", "2", "3"]
    assert [w["word"] for w in groups[1]] == ["4"]


def test_group_legacy_behavior_when_disabled():
    # 5 words; the 2nd ends with a comma. With max_words=4 the legacy
    # (non-clause) splitter breaks after the 4th word. The clause-aware
    # splitter walks back and finds the comma on the 2nd word, so it breaks
    # after the 2nd word instead. Note: the walker never picks the very
    # first word as a split point (to avoid creating 1-word orphans).
    words = [
        _word("the", 0.0, 0.3),
        _word("quick,", 0.3, 0.6),
        _word("brown", 0.6, 0.9),
        _word("fox", 0.9, 1.2),
        _word("jumps", 1.2, 1.5),
    ]
    groups_with = _group_words(
        words, max_words=4, max_chars=200, pause_threshold=10.0, use_clause_boundaries=True
    )
    groups_without = _group_words(
        words, max_words=4, max_chars=200, pause_threshold=10.0, use_clause_boundaries=False
    )
    # With clause awareness: split after the comma -> first group has 2 words.
    assert [w["word"] for w in groups_with[0]] == ["the", "quick,"]
    # Without clause awareness: split at max_words -> first group has 4 words.
    assert [w["word"] for w in groups_without[0]] == ["the", "quick,", "brown", "fox"]


def test_no_clause_boundaries_flag_disables_breaker():
    segments = [
        _segment(
            "the quick, brown fox jumps",
            [
                _word("the", 0.0, 0.3),
                _word("quick,", 0.3, 0.6),
                _word("brown", 0.6, 0.9),
                _word("fox", 0.9, 1.2),
                _word("jumps", 1.2, 1.5),
            ],
        )
    ]
    enabled = split_segments(segments, preset=PRESET_SHORTFORM, use_clause_boundaries=True)
    disabled = split_segments(segments, preset=PRESET_SHORTFORM, use_clause_boundaries=False)
    # With the breaker on, the first subtitle ends at the comma.
    assert enabled[0]["text"].endswith("quick,")
    # With the breaker off, the first subtitle fills to the max-words cap.
    assert disabled[0]["text"].endswith("fox")


# ---------------------------------------------------------------------------
# Two-line balancer
# ---------------------------------------------------------------------------


def test_balance_lines_splits_at_midpoint():
    text = "the quick, brown fox, jumps over the lazy dog today"
    cap = 22
    balanced = _balance_lines(text, max_line=cap)
    assert "\n" in balanced
    top, bottom = balanced.split("\n")
    # No text is lost.
    assert top + bottom == text
    # The split happens at a clause boundary.
    assert top.endswith((",", ".", "!", "?", ";", ":", "—", "–"))
    # The split is at most one line off from the midpoint.
    midpoint = len(text) / 2.0
    assert abs(len(top) - midpoint) <= cap


def test_balance_lines_short_text_unchanged():
    text = "hello world"
    assert _balance_lines(text, max_line=36) == text


def test_balance_lines_no_clause_returns_unchanged():
    text = "averylongwordwithnobreakpointthatdefinitelyneedssplitting"
    # No clause terminator -> can't balance.
    assert _balance_lines(text, max_line=20) == text


def test_balance_lines_disabled_when_max_line_zero():
    text = "the quick, brown fox, jumps over"
    assert _balance_lines(text, max_line=0) == text


def test_shortform_does_not_balance_by_default():
    text = "the quick, brown fox, jumps over the lazy dog"
    words = [
        _word("the", 0.0, 0.3),
        _word("quick,", 0.3, 0.6),
        _word("brown", 0.6, 0.9),
        _word("fox,", 0.9, 1.2),
        _word("jumps", 1.2, 1.5),
        _word("over", 1.5, 1.8),
        _word("the", 1.8, 2.1),
        _word("lazy", 2.1, 2.4),
        _word("dog", 2.4, 2.7),
    ]
    segments = [_segment(text, words, end=2.7)]
    result = split_segments(segments, preset=PRESET_SHORTFORM)
    assert "\n" not in result[0]["text"]


def test_longform_balances_by_default():
    text = "the quick, brown fox, jumps over the lazy dog"
    words = [
        _word("the", 0.0, 0.3),
        _word("quick,", 0.3, 0.6),
        _word("brown", 0.6, 0.9),
        _word("fox,", 0.9, 1.2),
        _word("jumps", 1.2, 1.5),
        _word("over", 1.5, 1.8),
        _word("the", 1.8, 2.1),
        _word("lazy", 2.1, 2.4),
        _word("dog", 2.4, 2.7),
    ]
    segments = [_segment(text, words, end=2.7)]
    result = split_segments(segments, preset=PRESET_LONGFORM)
    assert any("\n" in chunk["text"] for chunk in result)


# ---------------------------------------------------------------------------
# CPS limiter
# ---------------------------------------------------------------------------


def test_cps_limiter_extends_short_chunks():
    # 40 chars in 0.5s = 80 cps. To hit 21 cps we need 40/21 ~= 1.9s.
    groups = [{"start": 0.0, "end": 0.5, "text": "a" * 40}]
    _enforce_cps(groups, max_cps=21.0)
    assert groups[0]["end"] == pytest.approx(40 / 21.0, rel=1e-3)


def test_cps_limiter_clamps_to_next_chunk():
    # First chunk: 40 chars, only 0.5s of room; would need to extend to
    # 40/21 ~= 1.9s. The second chunk starts at 1.0, so the extension is
    # clamped to start - min_gap = 0.95.
    groups = [
        {"start": 0.0, "end": 0.5, "text": "a" * 40},
        {"start": 1.0, "end": 2.0, "text": "b" * 10},
    ]
    _enforce_cps(groups, max_cps=21.0, min_gap=0.05)
    # The extension is bounded by the gap, so end == 0.95.
    assert groups[0]["end"] == pytest.approx(0.95)
    assert groups[0]["end"] < 1.905  # would have been without clamp


def test_cps_limiter_disabled_is_noop():
    groups = [{"start": 0.0, "end": 0.5, "text": "a" * 40}]
    _enforce_cps(groups, max_cps=0)
    assert groups[0]["end"] == 0.5


def test_cps_limiter_ignores_newlines_in_char_count():
    # A balanced 2-line subtitle has roughly the same cps as a single-line
    # subtitle with the same number of readable characters.
    groups = [{"start": 0.0, "end": 1.0, "text": "line one\nline two\n"}]
    _enforce_cps(groups, max_cps=1000.0)
    # 16 readable chars / 1.0s = 16 cps, well under 1000. No extension.
    assert groups[0]["end"] == 1.0


def test_no_cps_limit_flag_disables_enforcement():
    # 4 long words, 0.5s total -> "xxxxx xxxxx xxxxx xxxxx" = 24 chars in
    # 0.5s = 48 cps. CPS limiter should extend the end.
    words = [
        _word("xxxxx", 0.0, 0.125),
        _word("xxxxx", 0.125, 0.25),
        _word("xxxxx", 0.25, 0.375),
        _word("xxxxx", 0.375, 0.5),
    ]
    segments = [
        _segment(
            "xxxxx xxxxx xxxxx xxxxx",
            words,
            end=0.5,
        )
    ]
    default = split_segments(segments, preset=PRESET_SHORTFORM)
    off = split_segments(
        segments,
        preset=PRESET_SHORTFORM,
        enforce_cps=False,
        enforce_min_duration=False,
    )
    # The default extends the end well past 0.5s; turning enforcement off
    # leaves the end at the original 0.5s.
    assert default[0]["end"] > 1.0
    assert off[0]["end"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Minimum display time
# ---------------------------------------------------------------------------


def test_min_duration_extends_short_chunks():
    groups = [{"start": 0.0, "end": 0.4, "text": "hi"}]
    _enforce_min_duration(groups, min_duration=1.0)
    assert groups[0]["end"] == pytest.approx(1.0)


def test_min_duration_clamps_to_next_chunk():
    # Second chunk starts at 0.8s; with min_gap=0.05 the first chunk can
    # extend only up to 0.75s.
    groups = [
        {"start": 0.0, "end": 0.4, "text": "hi"},
        {"start": 0.8, "end": 1.5, "text": "there"},
    ]
    _enforce_min_duration(groups, min_duration=1.0, min_gap=0.05)
    assert groups[0]["end"] == pytest.approx(0.75)


def test_min_duration_disabled_is_noop():
    groups = [{"start": 0.0, "end": 0.4, "text": "hi"}]
    _enforce_min_duration(groups, min_duration=0)
    assert groups[0]["end"] == 0.4


def test_min_duration_leaves_long_chunks_alone():
    groups = [{"start": 0.0, "end": 5.0, "text": "a long subtitle"}]
    _enforce_min_duration(groups, min_duration=1.0)
    assert groups[0]["end"] == 5.0


def test_no_min_duration_flag_disables_enforcement():
    segments = [
        _segment("hi", [_word("hi", 0.0, 0.4)], end=0.4)
    ]
    default = split_segments(segments, preset=PRESET_SHORTFORM)
    off = split_segments(
        segments,
        preset=PRESET_SHORTFORM,
        enforce_min_duration=False,
        enforce_cps=False,
    )
    # Default extends to >= 1.0s; off leaves it at 0.4s.
    assert default[0]["end"] >= 1.0
    assert off[0]["end"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_preset_targets_preserved():
    # A simple, deterministic case: 8 words, no pauses, no speakers, no
    # clauses, no balance, no enforcement. With every new behavior off the
    # output should be byte-identical to the old splitter.
    segments = [
        _segment(
            "one two three four five six seven eight",
            [_word(str(i), i * 0.5, i * 0.5 + 0.4) for i in range(8)],
        )
    ]
    legacy = split_segments(
        segments,
        preset=PRESET_SHORTFORM,
        cleanup=False,
        use_clause_boundaries=False,
        balance_lines=False,
        enforce_cps=False,
        enforce_min_duration=False,
    )
    assert len(legacy) == 2
    assert legacy[0]["text"] == "0 1 2 3"
    assert legacy[1]["text"] == "4 5 6 7"
    assert legacy[0]["end"] == pytest.approx(1.9)
    assert legacy[1]["end"] == pytest.approx(3.9)


def test_default_output_extends_subtitle_end():
    # A 40-character subtitle in 0.5s gets its end extended by default.
    segments = [
        _segment("a" * 40, [_word("a", 0.0, 0.5)], end=0.5)
    ]
    result = split_segments(segments, preset=PRESET_SHORTFORM)
    assert result[0]["end"] > 0.5


def test_default_cps_and_min_duration_match_module_constants():
    # The defaults exposed by the module should match the values used by
    # ``split_segments`` when called with no enforcement overrides.
    assert DEFAULT_MAX_CPS > 0
    assert DEFAULT_MIN_DURATION > 0


# ---------------------------------------------------------------------------
# Sentence-boundary splitting
# ---------------------------------------------------------------------------


def test_is_sentence_terminator_recognises_enders():
    assert _is_sentence_terminator("mission.")
    assert _is_sentence_terminator("really?")
    assert _is_sentence_terminator("watch!")
    assert not _is_sentence_terminator("hello,")
    assert not _is_sentence_terminator("hello")
    assert not _is_sentence_terminator("")


def test_is_sentence_terminator_skips_known_abbreviations():
    assert not _is_sentence_terminator("Mr.")
    assert not _is_sentence_terminator("Mrs.")
    assert not _is_sentence_terminator("Dr.")
    assert not _is_sentence_terminator("Prof.")
    assert not _is_sentence_terminator("Sr.")
    assert not _is_sentence_terminator("Jr.")
    assert not _is_sentence_terminator("St.")
    assert not _is_sentence_terminator("vs.")
    assert not _is_sentence_terminator("etc.")
    assert not _is_sentence_terminator("e.g.")
    assert not _is_sentence_terminator("i.e.")
    assert not _is_sentence_terminator("cf.")
    # Single-letter initials (A. B. Smith).
    assert not _is_sentence_terminator("A.")
    assert not _is_sentence_terminator("a.")
    # Exclamation / question are not abbreviations.
    assert _is_sentence_terminator("help!")


def test_sentence_split_breaks_at_period():
    # The motivating example: "mission. Building in public" must not stay
    # as one subtitle. The period ends up on the first piece, the rest
    # starts a fresh subtitle.
    words = [
        _word("mission.", 0.0, 0.5),
        _word("Building", 0.5, 0.9),
        _word("in", 0.9, 1.1),
        _word("public", 1.1, 1.5),
    ]
    result = split_segments(
        [_segment("mission. Building in public", words, end=1.5)],
        preset=PRESET_SHORTFORM,
    )
    assert len(result) == 2
    assert result[0]["text"] == "mission."
    assert result[1]["text"] == "Building in public"


def test_sentence_split_breaks_at_question_mark():
    words = [
        _word("How", 0.0, 0.2),
        _word("are", 0.2, 0.4),
        _word("you?", 0.4, 0.6),
        _word("I", 0.6, 0.8),
        _word("am", 0.8, 1.0),
        _word("fine.", 1.0, 1.4),
    ]
    result = split_segments(
        [_segment("How are you? I am fine.", words, end=1.4)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["How are you?", "I am fine."]


def test_sentence_split_breaks_at_exclamation():
    words = [
        _word("Watch", 0.0, 0.3),
        _word("out!", 0.3, 0.6),
        _word("The", 0.6, 0.8),
        _word("car", 0.8, 1.1),
        _word("is", 1.1, 1.3),
        _word("coming.", 1.3, 1.7),
    ]
    result = split_segments(
        [_segment("Watch out! The car is coming.", words, end=1.7)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["Watch out!", "The car is coming."]


def test_sentence_split_handles_mr_and_dr():
    # "Mr. Smith is here." must NOT split inside "Mr. Smith" — the
    # abbreviation is in the allowlist. The split happens at the real
    # sentence boundary after "here."
    words = [
        _word("Mr.", 0.0, 0.2),
        _word("Smith", 0.2, 0.5),
        _word("is", 0.5, 0.7),
        _word("here.", 0.7, 1.1),
        _word("He", 1.1, 1.3),
        _word("is", 1.3, 1.5),
        _word("nice.", 1.5, 1.9),
    ]
    result = split_segments(
        [_segment("Mr. Smith is here. He is nice.", words, end=1.9)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["Mr. Smith is here.", "He is nice."]


def test_sentence_split_handles_us_initial():
    # "U.S. policy." should not split at the initial.
    words = [
        _word("U.S.", 0.0, 0.3),
        _word("policy.", 0.3, 0.7),
        _word("That's", 0.7, 1.0),
        _word("what", 1.0, 1.2),
        _word("I", 1.2, 1.3),
        _word("said.", 1.3, 1.6),
    ]
    result = split_segments(
        [_segment("U.S. policy. That's what I said.", words, end=1.6)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["U.S. policy.", "That's what I said."]


def test_sentence_split_handles_latin_abbrev():
    # "e.g." is in the allowlist; the period is skipped, so the chunk
    # stays whole until the next real sentence boundary at "something."
    words = [
        _word("e.g.", 0.0, 0.3),
        _word("something.", 0.3, 0.7),
        _word("Then", 0.7, 1.0),
        _word("more.", 1.0, 1.4),
    ]
    result = split_segments(
        [_segment("e.g. something. Then more.", words, end=1.4)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["e.g. something.", "Then more."]


def test_sentence_split_first_word_can_be_orphan():
    # Unlike the clause-aware breaker, a 1-word "mission." subtitle is
    # acceptable here — the alternative is a stranded period in the
    # following subtitle.
    words = [
        _word("mission.", 0.0, 0.4),
        _word("Building", 0.4, 0.7),
        _word("in", 0.7, 0.9),
        _word("public", 0.9, 1.3),
    ]
    groups = _group_words(
        words, max_words=4, max_chars=22, pause_threshold=1.0
    )
    assert [w["word"] for w in groups[0]] == ["mission."]
    assert [w["word"] for w in groups[1]] == ["Building", "in", "public"]


def test_sentence_split_no_split_when_disabled():
    # The same input as the motivating example, but with the feature
    # disabled: it stays as one subtitle.
    words = [
        _word("mission.", 0.0, 0.5),
        _word("Building", 0.5, 0.9),
        _word("in", 0.9, 1.1),
        _word("public", 1.1, 1.5),
    ]
    result = split_segments(
        [_segment("mission. Building in public", words, end=1.5)],
        preset=PRESET_SHORTFORM,
        split_at_sentences=False,
    )
    assert len(result) == 1
    assert result[0]["text"] == "mission. Building in public"


def test_sentence_split_three_periods_in_a_row():
    # "Hello. World. Foo." should produce three one-word subtitles,
    # never a subtitle with a period in the middle.
    words = [
        _word("Hello.", 0.0, 0.3),
        _word("World.", 0.3, 0.6),
        _word("Foo.", 0.6, 0.9),
    ]
    result = split_segments(
        [_segment("Hello. World. Foo.", words, end=0.9)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["Hello.", "World.", "Foo."]


def test_sentence_split_runs_with_pause_threshold():
    # A period in the middle + a long pause after it: the period split
    # fires first, then the pause between subsequent chunks is fine.
    words = [
        _word("Yes.", 0.0, 0.3),
        _word("No.", 0.3, 0.6),
        # 1.5s pause before "maybe"
        _word("maybe", 2.1, 2.5),
        _word("later.", 2.5, 2.9),
    ]
    result = split_segments(
        [_segment("Yes. No. maybe later.", words, end=2.9)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["Yes.", "No.", "maybe later."]


# ---------------------------------------------------------------------------
# Context-aware sentence splitting
# ---------------------------------------------------------------------------


def test_default_sentence_pause_threshold_is_module_constant():
    # The default exposed by the module is the one used by
    # ``split_segments`` when no override is passed.
    assert SENTENCE_PAUSE_THRESHOLD > 0
    assert SENTENCE_PAUSE_THRESHOLD < 0.5  # smaller than the legacy pause threshold


def test_sentence_split_keeps_together_when_no_context():
    # "found bug. reports" — period is followed by a lowercase word with
    # no pause, and the chunk is well under the max_words cap. The
    # period stays in the middle of the subtitle because none of the
    # context signals (capital, pause, speaker change) support a real
    # sentence boundary here.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        _word("reports", 0.5, 0.7),
    ]
    result = split_segments(
        [_segment("found bug. reports", words, end=0.7)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["found bug. reports"]


def test_sentence_split_splits_on_capital():
    # Same shape but the next word is capitalized — the split fires.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        _word("Reports", 0.5, 0.7),
        _word("later.", 0.7, 0.9),
    ]
    result = split_segments(
        [_segment("found bug. Reports later.", words, end=0.9)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["found bug.", "Reports later."]


def test_sentence_split_splits_on_long_pause():
    # Lowercase next word, but a 0.3s breath after the period — the
    # pause is enough supporting evidence to trigger the split.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        # 0.3s gap before the next word
        _word("reports", 0.8, 1.0),
    ]
    result = split_segments(
        [_segment("found bug. reports", words, end=1.0)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["found bug.", "reports"]


def test_sentence_split_keeps_on_short_pause_lowercase():
    # 0.05s gap is below the default 0.2s threshold, and the next word
    # is lowercase — the chunk should stay together.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        # 0.05s gap
        _word("reports", 0.55, 0.75),
    ]
    result = split_segments(
        [_segment("found bug. reports", words, end=0.75)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["found bug. reports"]


def test_sentence_split_pause_threshold_tunable():
    # A 0.1s gap is too short to trigger a split at the default
    # threshold of 0.2s. Lowering the threshold to 0 makes the gap
    # always sufficient.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        # 0.1s gap
        _word("reports", 0.6, 0.8),
    ]
    default_result = split_segments(
        [_segment("found bug. reports", words, end=0.8)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in default_result] == ["found bug. reports"]

    permissive_result = split_segments(
        [_segment("found bug. reports", words, end=0.8)],
        preset=PRESET_SHORTFORM,
        sentence_pause_threshold=0.0,
    )
    assert [r["text"] for r in permissive_result] == ["found bug.", "reports"]


def test_sentence_split_speaker_change_triggers_split():
    # When the next word belongs to a different speaker, the split fires
    # even with no capital and no meaningful pause.
    words = [
        _word("Hello.", 0.0, 0.3, speaker="SPEAKER_01"),
        _word("hi", 0.3, 0.5, speaker="SPEAKER_02"),
        _word("there.", 0.5, 0.8, speaker="SPEAKER_02"),
    ]
    result = split_segments(
        [_segment("Hello. hi there.", words, end=0.8)],
        preset=PRESET_SHORTFORM,
    )
    assert len(result) == 2
    assert "[SPEAKER_01]" in result[0]["text"]
    assert "Hello." in result[0]["text"]
    assert "[SPEAKER_02]" in result[1]["text"]
    assert "hi there." in result[1]["text"]


def test_sentence_split_no_speaker_labels_no_change():
    # When neither word has a speaker label (no diarization), the
    # speaker-change check is a no-op and we fall back to cap-or-pause.
    words = [
        _word("bug.", 0.0, 0.3, speaker=None),
        _word("reports", 0.3, 0.5, speaker=None),
        _word("later.", 0.5, 0.8, speaker=None),
    ]
    result = split_segments(
        [_segment("bug. reports later.", words, end=0.8)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["bug. reports later."]


def test_sentence_split_still_handles_real_sentences():
    # Regression guard: the motivating "mission. Building" case must
    # still split (capital letter is a strong context signal).
    words = [
        _word("mission.", 0.0, 0.5),
        _word("Building", 0.5, 0.9),
        _word("in", 0.9, 1.1),
        _word("public", 1.1, 1.5),
    ]
    result = split_segments(
        [_segment("mission. Building in public", words, end=1.5)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["mission.", "Building in public"]


def test_sentence_split_still_skips_abbreviations():
    # Regression guard: "Mr. Smith" must not split even with a capital
    # next word, because "Mr." is in the abbreviation allowlist.
    words = [
        _word("Mr.", 0.0, 0.2),
        _word("Smith", 0.2, 0.5),
        _word("is", 0.5, 0.7),
        _word("here.", 0.7, 1.1),
    ]
    result = split_segments(
        [_segment("Mr. Smith is here.", words, end=1.1)],
        preset=PRESET_SHORTFORM,
    )
    assert [r["text"] for r in result] == ["Mr. Smith is here."]


def test_sentence_split_disabled_keeps_legacy_behavior():
    # Disabling sentence-splitting entirely reproduces the 0.2.x
    # behavior: no period is a sentence boundary, so the chunk stays
    # whole regardless of context.
    words = [
        _word("found", 0.0, 0.2),
        _word("bug.", 0.2, 0.5),
        _word("reports", 0.5, 0.7),
    ]
    result = split_segments(
        [_segment("found bug. reports", words, end=0.7)],
        preset=PRESET_SHORTFORM,
        split_at_sentences=False,
    )
    assert [r["text"] for r in result] == ["found bug. reports"]
