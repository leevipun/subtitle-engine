"""Tests for the segmenter/preset splitter."""

import pytest

from subtitle_engine.segmenter import (
    PRESET_LONGFORM,
    PRESET_SHORTFORM,
    split_segments,
)


def _segment(text: str, words: list[dict] | None = None, start: float = 0.0, end: float = 1.0) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": words if words is not None else [{"word": w} for w in text.split()],
    }


def test_shortform_splits_to_small_chunks():
    segment = _segment("one two three four five six seven eight", start=0.0, end=8.0)
    result = split_segments([segment], preset=PRESET_SHORTFORM)

    assert len(result) >= 2
    for chunk in result:
        word_count = len(chunk["text"].split())
        assert 1 <= word_count <= 5


def test_longform_allows_larger_chunks():
    segment = _segment(" ".join(str(i) for i in range(25)), start=0.0, end=25.0)
    result = split_segments([segment], preset=PRESET_LONGFORM)

    assert len(result) >= 1
    for chunk in result:
        word_count = len(chunk["text"].split())
        assert 1 <= word_count <= 14


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        split_segments([], preset="invalid")


def test_word_timings_are_used():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "today", "start": 1.0, "end": 1.5},
    ]
    segment = _segment("hello world today", words=words, start=0.0, end=1.5)
    result = split_segments([segment], preset=PRESET_SHORTFORM)

    assert result[0]["start"] == pytest.approx(0.0)
    assert result[-1]["end"] == pytest.approx(1.5)


def test_speaker_label_is_preserved():
    words = [
        {"word": "hello", "speaker": "SPEAKER_01"},
        {"word": "world", "speaker": "SPEAKER_01"},
        {"word": "today", "speaker": "SPEAKER_02"},
    ]
    segment = _segment("hello world today", words=words, start=0.0, end=3.0)
    result = split_segments([segment], preset=PRESET_SHORTFORM)

    assert any("[SPEAKER_01]" in chunk["text"] for chunk in result)


def test_empty_segment_is_ignored():
    segment = {"start": 0.0, "end": 1.0, "text": "   ", "words": []}
    result = split_segments([segment], preset=PRESET_SHORTFORM)
    assert result == []


def test_segment_without_words_splits_by_text():
    segment = {"start": 0.0, "end": 9.0, "text": "one two three four five six seven eight nine"}
    result = split_segments([segment], preset=PRESET_SHORTFORM)

    total_words = sum(len(chunk["text"].split()) for chunk in result)
    assert total_words == 9
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[-1]["end"] == pytest.approx(9.0)
