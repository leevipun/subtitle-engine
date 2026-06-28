"""Tests for the ffmpeg-based subtitle burner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitle_engine.burner import (
    BurnError,
    _build_filter,
    _escape_filter_path,
    _has_video_stream,
    _parse_ass_style,
    _parse_progress_line,
    _probe_duration,
    burn_subtitles,
)


# --- _parse_progress_line ---------------------------------------------------


def test_parse_progress_line_returns_fraction():
    total = 60.0
    line = "out_time_ms=30000000"  # 30 seconds
    assert _parse_progress_line(line, total) == pytest.approx(0.5)


def test_parse_progress_line_clamps_to_one():
    total = 10.0
    line = "out_time_ms=99999999"
    assert _parse_progress_line(line, total) == 1.0


def test_parse_progress_line_ignores_negative_time():
    total = 10.0
    line = "out_time_ms=-1000"
    assert _parse_progress_line(line, total) is None


def test_parse_progress_line_returns_none_when_unknown():
    assert _parse_progress_line("frame=123", 60.0) is None


def test_parse_progress_line_returns_none_when_no_total():
    assert _parse_progress_line("out_time_ms=1000", 0.0) is None


def test_parse_progress_line_handles_garbage_value():
    assert _parse_progress_line("out_time_ms=notanumber", 60.0) is None


# --- _escape_filter_path ---------------------------------------------------


def test_escape_filter_path_escapes_colon():
    result = _escape_filter_path(Path("/tmp/has:colon.srt"))
    assert "\\:" in result


def test_escape_filter_path_escapes_backslash():
    result = _escape_filter_path(Path(r"C:\has\backslash.srt"))
    assert "\\:" in result or "/" in result


# --- _parse_ass_style ------------------------------------------------------


def test_parse_ass_style_returns_fields_after_name(tmp_path: Path):
    ass = tmp_path / "style.ass"
    ass.write_text(
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour\n"
        "Style: Default,Arial,24,&H00FFFFFF\n",
        encoding="utf-8",
    )
    assert _parse_ass_style(ass) == "Arial,24,&H00FFFFFF"


def test_parse_ass_style_case_insensitive(tmp_path: Path):
    ass = tmp_path / "style.ass"
    ass.write_text(
        "style: Default,Arial,20,&H000000\n",
        encoding="utf-8",
    )
    assert _parse_ass_style(ass) == "Arial,20,&H000000"


def test_parse_ass_style_raises_when_missing(tmp_path: Path):
    ass = tmp_path / "style.ass"
    ass.write_text("[V4+ Styles]\nFormat: Name\n", encoding="utf-8")
    with pytest.raises(BurnError, match="No 'Style:' line"):
        _parse_ass_style(ass)


# --- _build_filter ---------------------------------------------------------


def test_build_filter_default(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    result = _build_filter(srt, None)
    assert result == f"subtitles={_escape_filter_path(srt)}"


def test_build_filter_with_style(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    style = tmp_path / "style.ass"
    style.write_text(
        "Style: Default,Arial,24,&H00FFFFFF\n",
        encoding="utf-8",
    )
    result = _build_filter(srt, style)
    assert result.startswith("subtitles=")
    assert "force_style=" in result
    assert "Arial" in result


# --- _has_video_stream -----------------------------------------------------


def test_has_video_stream_true():
    fake_result = MagicMock(stdout="0\n1\n", returncode=0)
    with patch("subtitle_engine.burner.subprocess.run", return_value=fake_result):
        with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffprobe"):
            assert _has_video_stream(Path("video.mp4")) is True


def test_has_video_stream_false():
    fake_result = MagicMock(stdout="", returncode=0)
    with patch("subtitle_engine.burner.subprocess.run", return_value=fake_result):
        with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffprobe"):
            assert _has_video_stream(Path("audio.mp3")) is False


def test_has_video_stream_without_ffprobe_assumes_video():
    with patch("subtitle_engine.burner.shutil.which", return_value=None):
        assert _has_video_stream(Path("video.mp4")) is True


# --- _probe_duration -------------------------------------------------------


def test_probe_duration_parses_value():
    fake_result = MagicMock(stdout="42.5\n", returncode=0)
    with patch("subtitle_engine.burner.subprocess.run", return_value=fake_result):
        with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffprobe"):
            assert _probe_duration(Path("video.mp4")) == 42.5


def test_probe_duration_returns_zero_without_ffprobe():
    with patch("subtitle_engine.burner.shutil.which", return_value=None):
        assert _probe_duration(Path("video.mp4")) == 0.0


def test_probe_duration_returns_zero_on_garbage():
    fake_result = MagicMock(stdout="not a number\n", returncode=0)
    with patch("subtitle_engine.burner.subprocess.run", return_value=fake_result):
        with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffprobe"):
            assert _probe_duration(Path("video.mp4")) == 0.0


# --- burn_subtitles: validation -------------------------------------------


def test_burn_subtitles_raises_when_input_missing(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    with pytest.raises(BurnError, match="Input video not found"):
        burn_subtitles(tmp_path / "missing.mp4", srt, tmp_path / "out.mp4")


def test_burn_subtitles_raises_when_srt_missing(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(BurnError, match="SRT file not found"):
        burn_subtitles(video, tmp_path / "missing.srt", tmp_path / "out.mp4")


def test_burn_subtitles_raises_when_style_file_missing(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    with pytest.raises(BurnError, match="Style file not found"):
        burn_subtitles(
            video,
            srt,
            tmp_path / "out.mp4",
            style_file=tmp_path / "missing.ass",
        )


def test_burn_subtitles_raises_when_style_file_wrong_ext(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    bogus = tmp_path / "style.txt"
    bogus.write_text("dummy", encoding="utf-8")
    with pytest.raises(BurnError, match="expects an .ass file"):
        burn_subtitles(
            video,
            srt,
            tmp_path / "out.mp4",
            style_file=bogus,
        )


def test_burn_subtitles_raises_when_ffmpeg_missing(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    with patch("subtitle_engine.burner.shutil.which", return_value=None):
        with pytest.raises(BurnError, match="ffmpeg not found"):
            burn_subtitles(video, srt, tmp_path / "out.mp4")


def test_burn_subtitles_raises_when_no_video_stream(tmp_path: Path):
    video = tmp_path / "audio.mp3"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch(
                "subtitle_engine.burner._has_video_stream", return_value=False
            ):
                with pytest.raises(BurnError, match="no video stream"):
                    burn_subtitles(video, srt, tmp_path / "out.mp4")


def test_burn_subtitles_raises_when_ffmpeg_lacks_libass(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=False):
            with pytest.raises(BurnError, match="libass"):
                burn_subtitles(video, srt, tmp_path / "out.mp4")


# --- burn_subtitles: success path -----------------------------------------


def _fake_proc_with_progress(progress_lines: list[str], returncode: int = 0):
    """Build a mock Popen whose stdout yields ``progress_lines`` then EOF."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.wait.return_value = None
    proc.stdout = iter(progress_lines + ["progress=end\n", ""])
    proc.stderr = iter([])
    return proc


def test_burn_subtitles_invokes_ffmpeg_with_subtitles_filter(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    out = tmp_path / "video.subtitled.mp4"

    proc = _fake_proc_with_progress([], returncode=0)

    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch("subtitle_engine.burner._has_video_stream", return_value=True):
                with patch("subtitle_engine.burner._probe_duration", return_value=60.0):
                    with patch("subtitle_engine.burner.subprocess.Popen", return_value=proc) as mock_popen:
                        burn_subtitles(video, srt, out)

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-vf" in cmd
    vf_index = cmd.index("-vf")
    assert cmd[vf_index + 1].startswith("subtitles=")
    # Audio is copied, not re-encoded.
    audio_index = cmd.index("-c:a")
    assert cmd[audio_index + 1] == "copy"


def test_burn_subtitles_passes_style_file_to_filter(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    style = tmp_path / "style.ass"
    style.write_text(
        "Style: Default,Arial,24,&H00FFFFFF\n",
        encoding="utf-8",
    )
    out = tmp_path / "video.subtitled.mp4"

    proc = _fake_proc_with_progress([], returncode=0)

    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch("subtitle_engine.burner._has_video_stream", return_value=True):
                with patch("subtitle_engine.burner._probe_duration", return_value=60.0):
                    with patch("subtitle_engine.burner.subprocess.Popen", return_value=proc) as mock_popen:
                        burn_subtitles(video, srt, out, style_file=style)

    cmd = mock_popen.call_args[0][0]
    vf = cmd[cmd.index("-vf") + 1]
    assert "force_style=" in vf
    assert "Arial" in vf


def test_burn_subtitles_creates_output_parent_directory(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    out = tmp_path / "nested" / "dir" / "video.subtitled.mp4"
    proc = _fake_proc_with_progress([], returncode=0)
    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch("subtitle_engine.burner._has_video_stream", return_value=True):
                with patch("subtitle_engine.burner._probe_duration", return_value=0.0):
                    with patch("subtitle_engine.burner.subprocess.Popen", return_value=proc):
                        burn_subtitles(video, srt, out)
    assert out.parent.exists()


def test_burn_subtitles_raises_on_ffmpeg_failure(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    out = tmp_path / "out.mp4"

    proc = MagicMock()
    proc.returncode = 1
    proc.wait.return_value = None
    proc.stdout = iter([""])
    proc.stderr = iter(
        ["ffmpeg version 6.0\n", "[error] something broke\n", "tail line\n"]
    )

    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch("subtitle_engine.burner._has_video_stream", return_value=True):
                with patch("subtitle_engine.burner._probe_duration", return_value=10.0):
                    with patch("subtitle_engine.burner.subprocess.Popen", return_value=proc):
                        with pytest.raises(BurnError, match="exit code 1"):
                            burn_subtitles(video, srt, out)


def test_burn_subtitles_emits_progress_callbacks(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    srt = tmp_path / "subs.srt"
    srt.write_text("dummy", encoding="utf-8")
    out = tmp_path / "out.mp4"

    progress_lines = [
        "out_time_ms=10000000\n",  # 10s / 50s = 0.2
        "out_time_ms=25000000\n",  # 25s / 50s = 0.5
        "out_time_ms=50000000\n",  # 50s / 50s = 1.0
    ]
    proc = _fake_proc_with_progress(progress_lines, returncode=0)
    seen: list[tuple[str, float]] = []

    with patch("subtitle_engine.burner.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subtitle_engine.burner._has_subtitle_filter", return_value=True):
            with patch("subtitle_engine.burner._has_video_stream", return_value=True):
                with patch("subtitle_engine.burner._probe_duration", return_value=50.0):
                    with patch("subtitle_engine.burner.subprocess.Popen", return_value=proc):
                        burn_subtitles(
                            video,
                            srt,
                            out,
                            progress_callback=lambda stage, fraction: seen.append(
                                (stage, fraction)
                            ),
                        )

    # First call is the "starting" 0.0, then updates, then 1.0 at the end.
    assert seen[0] == ("burning", 0.0)
    assert seen[-1] == ("burning", 1.0)
    # The intermediate updates should be increasing and the final one is 1.0.
    fractions = [f for _, f in seen[1:-1]]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
