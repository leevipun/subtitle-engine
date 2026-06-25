"""Tests for CLI helpers and argument parsing."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from subtitle_engine.cli import app
from subtitle_engine.utils import resolve_output_path, validate_media_file

runner = CliRunner()


def test_resolve_output_path_default():
    input_path = Path("movie.mp4")
    assert resolve_output_path(input_path) == Path("movie.srt")


def test_resolve_output_path_explicit():
    input_path = Path("movie.mp4")
    output = Path("custom.srt")
    assert resolve_output_path(input_path, output) == output


def test_validate_media_file_supported():
    validate_media_file(Path("video.mp4"))


def test_validate_media_file_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_media_file(Path("file.txt"))


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Generate SRT subtitles" in result.output


def test_cli_no_args():
    result = runner.invoke(app)
    assert result.exit_code != 0
    assert "Usage:" in result.output


def test_caption_requires_ollama_model(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, [str(media), "--caption"])
    assert result.exit_code != 0
    assert "--ollama-model is required" in result.output


def test_cli_version_long():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "subeng" in result.output
    assert "0.1.1" in result.output


def test_cli_version_short():
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert "subeng" in result.output


def test_cli_version_no_extra_output():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "subeng 0.1.1"


def test_cli_quiet_hides_status_but_keeps_errors(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, [str(media), "--caption", "-q"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Transcribing:" not in result.output


def test_cli_verbose_accepted(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, [str(media), "--caption", "--verbose"])
    assert result.exit_code != 0
    assert "--ollama-model is required" in result.output
