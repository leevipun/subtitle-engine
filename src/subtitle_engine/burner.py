"""Burn (hardcode) subtitles into a video using ffmpeg."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional


class BurnError(Exception):
    """Raised when subtitle burning fails."""


def _check_ffmpeg() -> str:
    """Return the path to ffmpeg, or raise ``BurnError`` if it is missing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BurnError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/."
        )
    return ffmpeg


def _has_subtitle_filter(ffmpeg: str) -> bool:
    """Return True if ffmpeg was built with libass (provides the subtitles filter)."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    for line in output.splitlines():
        # Columns are: status flags name rest-of-line
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[1]
        if name in ("subtitles", "ass"):
            return True
    return False


def _has_video_stream(video_path: Path) -> bool:
    """Return True if the input has at least one video stream."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Without ffprobe we can't be sure; let ffmpeg surface the real error.
        return True
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _probe_duration(video_path: Path) -> float:
    """Return the video's duration in seconds, or 0 if it cannot be determined."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter graph.

    The ``subtitles=`` filter treats ``\\`` and ``:`` as special characters
    inside the value, so we sanitize the path by copying the SRT to a safe
    location. This helper is only used as a defense in depth.
    """
    s = str(path).replace("\\", "/")
    return s.replace(":", r"\:").replace("'", r"\'")


def _parse_ass_style(ass_path: Path) -> str:
    """Extract a ``force_style`` value from a .ass file.

    Returns the comma-separated libass fields (after the style name) from
    the first ``Style:`` line in the file, suitable for passing to
    ffmpeg's ``subtitles=`` filter as ``force_style=…``.
    """
    text = ass_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("style:"):
            continue
        after = stripped.split(":", 1)[1].strip()
        fields = [field.strip() for field in after.split(",")]
        if len(fields) < 2:
            continue
        # Drop the style name; the rest maps directly to libass fields.
        return ",".join(fields[1:])
    raise BurnError(f"No 'Style:' line found in {ass_path}")


def _build_filter(safe_srt: Path, style_file: Optional[Path]) -> str:
    """Build the ffmpeg ``-vf`` argument value."""
    escaped = _escape_filter_path(safe_srt)
    if style_file is None:
        return f"subtitles={escaped}"
    style = _parse_ass_style(style_file).replace("'", r"\'")
    return f"subtitles={escaped}:force_style='{style}'"


_OUT_TIME_MS_RE = re.compile(r"^out_time_ms=(\d+)$")


def _parse_progress_line(line: str, total_seconds: float) -> Optional[float]:
    """Return a 0..1 fraction if ``line`` is an ffmpeg progress time marker."""
    match = _OUT_TIME_MS_RE.match(line.strip())
    if not match:
        return None
    if total_seconds <= 0:
        return None
    try:
        micros = int(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, micros / 1_000_000 / total_seconds))


def _tail(text: str, max_lines: int = 12) -> str:
    """Return the last ``max_lines`` of ``text`` for inclusion in error messages."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def burn_subtitles(
    input_video: Path,
    srt_path: Path,
    output_video: Path,
    style_file: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> None:
    """Re-encode ``input_video`` with ``srt_path`` burned into the frames.

    The audio stream is copied where possible to keep the burn step fast.
    On success, the result is written to ``output_video``. The original
    input is never modified.

    Parameters
    ----------
    input_video:
        Source video file with at least one video stream.
    srt_path:
        Subtitle file to render into the video. May have any name; the
        file is copied to a sanitized path before being passed to ffmpeg.
    output_video:
        Destination path. The parent directory is created if it does not
        exist. The original ``input_video`` is never modified.
    style_file:
        Optional ``.ass`` style file. When supplied, the first ``Style:``
        line is converted to a ``force_style=`` argument for ffmpeg's
        ``subtitles=`` filter.
    progress_callback:
        Optional ``(stage, fraction)`` callback. The only stage used is
        ``"burning"`` and ``fraction`` is in ``[0.0, 1.0]``.
    """
    input_video = Path(input_video)
    srt_path = Path(srt_path)
    output_video = Path(output_video)
    style_file = Path(style_file) if style_file is not None else None

    if not input_video.exists():
        raise BurnError(f"Input video not found: {input_video}")
    if not srt_path.exists():
        raise BurnError(f"SRT file not found: {srt_path}")
    if style_file is not None and not style_file.exists():
        raise BurnError(f"Style file not found: {style_file}")
    if style_file is not None and style_file.suffix.lower() != ".ass":
        raise BurnError(
            f"--style-file expects an .ass file, got '{style_file.suffix}'"
        )

    ffmpeg = _check_ffmpeg()

    if not _has_subtitle_filter(ffmpeg):
        raise BurnError(
            "ffmpeg was built without libass, so the 'subtitles' filter is "
            "unavailable. Reinstall ffmpeg with libass support (e.g. "
            "'brew install ffmpeg' on macOS) to enable subtitle burning."
        )

    if not _has_video_stream(input_video):
        raise BurnError(
            f"Input has no video stream: {input_video}. "
            "Subtitle burning requires a video file."
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    total_seconds = _probe_duration(input_video)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the SRT to a sanitized path so the ``subtitles=`` filter is
        # never tripped by stray ``:`` or ``\`` characters in the filename.
        safe_srt = Path(tmpdir) / "subs.srt"
        safe_srt.write_text(
            srt_path.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

        vf = _build_filter(safe_srt, style_file)
        cmd = [
            ffmpeg,
            "-y",
            "-nostats",
            "-progress",
            "pipe:1",
            "-i",
            str(input_video),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_video),
        ]

        if progress_callback is not None:
            progress_callback("burning", 0.0)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_chunks: list[str] = []
        last_fraction = 0.0

        def _drain_stderr() -> None:
            if process.stderr is None:
                return
            for chunk in process.stderr:
                stderr_chunks.append(chunk)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            if process.stdout is not None:
                for line in process.stdout:
                    fraction = _parse_progress_line(line, total_seconds)
                    if fraction is None:
                        continue
                    if (
                        progress_callback is not None
                        and fraction - last_fraction >= 0.01
                    ):
                        progress_callback("burning", fraction)
                        last_fraction = fraction
        finally:
            process.wait()
            stderr_thread.join()

        if process.returncode != 0:
            raise BurnError(
                f"ffmpeg failed with exit code {process.returncode}:\n"
                f"{_tail(''.join(stderr_chunks))}"
            )

        if progress_callback is not None:
            progress_callback("burning", 1.0)
