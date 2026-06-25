"""Tests for the segmenter/preset splitter."""

import pytest

from subtitle_engine.segmenter import (
    PRESET_LONGFORM,
    PRESET_SHORTFORM,
    _collect_words,
    _group_words,
    split_segments,
)


def _word(word: str, start: float, end: float, speaker: str | None = None) -> dict:
    return {"word": word, "start": start, "end": end, "speaker": speaker}


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
