"""Tests for the SRT writer module."""

from pathlib import Path

import pytest

from subtitle_engine.srt_writer import (
    _format_segment,
    _format_time,
    extract_text_from_srt,
    segments_to_srt,
    write_srt,
)


def test_format_time_zero():
    assert _format_time(0.0) == "00:00:00,000"


def test_format_time_with_hours():
    assert _format_time(3661.123) == "01:01:01,123"


def test_format_time_milliseconds_rounding():
    assert _format_time(0.9996) == "00:00:01,000"


def test_format_time_millis_ceiling_guard():
    # 1.9999 rounds to 2.000 -> should not produce 1000 ms
    assert _format_time(1.9999) == "00:00:02,000"


def test_format_segment():
    block = _format_segment(1, 1.5, 4.25, "Hello world")
    assert block == "1\n00:00:01,500 --> 00:00:04,250\nHello world\n"


def test_segments_to_srt():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "First line"},
        {"start": 3.5, "end": 5.5, "text": "Second line"},
    ]
    srt = segments_to_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:02,000\nFirst line" in srt
    assert "2\n00:00:03,500 --> 00:00:05,500\nSecond line" in srt


def test_segments_to_srt_empty_text_falls_back():
    srt = segments_to_srt([{"start": 0.0, "end": 1.0, "text": "   "}])
    assert "..." in srt


def test_segments_to_srt_empty():
    assert segments_to_srt([]) == ""


def test_write_srt(tmp_path: Path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
    output = tmp_path / "subs.srt"
    write_srt(segments, output)
    assert output.exists()
    assert "00:00:00,000 --> 00:00:01,000" in output.read_text(encoding="utf-8")


def test_write_srt_creates_parent_dirs(tmp_path: Path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
    output = tmp_path / "nested" / "dir" / "subs.srt"
    write_srt(segments, output)
    assert output.exists()


def test_extract_text_from_srt(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nSecond line\n",
        encoding="utf-8",
    )
    assert extract_text_from_srt(srt) == "Hello world Second line"


def test_extract_text_from_srt_multiline_text(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:04,000\nFirst line\nSecond line\n",
        encoding="utf-8",
    )
    assert extract_text_from_srt(srt) == "First line Second line"


def test_extract_text_from_srt_ignores_blank_blocks(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\n   \n",
        encoding="utf-8",
    )
    assert extract_text_from_srt(srt) == "Hello"


def test_extract_text_from_srt_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.srt"
    with pytest.raises(FileNotFoundError):
        extract_text_from_srt(missing)


def test_extract_text_from_srt_no_text(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n   \n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No subtitle text"):
        extract_text_from_srt(srt)
