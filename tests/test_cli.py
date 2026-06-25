"""Tests for CLI helpers and argument parsing."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from subtitle_engine import __version__
from subtitle_engine.cli import _select_ollama_model, app, main_entry, update
from subtitle_engine.updater import UpdateCheckError, UpdateInfo
from subtitle_engine.utils import resolve_output_path, validate_media_file

runner = CliRunner()


@pytest.fixture(autouse=True)
def disable_update_check():
    """Prevent the CLI from hitting the network during transcription tests."""
    with patch("subtitle_engine.cli.check_for_update", return_value=None):
        yield


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


def test_caption_prompts_for_ollama_model(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    with patch("subtitle_engine.cli._select_ollama_model", return_value="qwen3.5:0.8b") as mock_select:
        with patch("subtitle_engine.cli.transcribe", return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}]):
            with patch("subtitle_engine.cli.split_segments", return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}]):
                with patch("subtitle_engine.cli.generate_caption", return_value="A caption"):
                    result = runner.invoke(app, ["main", str(media), "--caption"])
    assert result.exit_code == 0
    mock_select.assert_called_once_with("http://localhost:11434")
    assert "Wrote caption" in result.output


def test_cli_version_long():
    result = runner.invoke(app, ["main", "--version"])
    assert result.exit_code == 0
    assert "subeng" in result.output
    assert "0.1.3" in result.output


def test_cli_version_short():
    result = runner.invoke(app, ["main", "-v"])
    assert result.exit_code == 0
    assert "subeng" in result.output


def test_cli_version_no_extra_output():
    result = runner.invoke(app, ["main", "--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"subeng {__version__}"


def test_cli_quiet_hides_status_but_keeps_errors(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(
        app,
        ["main", str(media), "--caption", "--ollama-model", "qwen3.5:0.8b", "-q"],
    )
    assert result.exit_code != 0
    assert "Transcription failed:" in result.output
    assert "Transcribing:" not in result.output


def test_cli_verbose_accepted(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(
        app,
        ["main", str(media), "--caption", "--ollama-model", "qwen3.5:0.8b", "--verbose"],
    )
    assert result.exit_code != 0
    assert "Transcription failed:" in result.output
    assert "Transcribing:" in result.output


def test_update_command_shows_up_to_date():
    with patch("subtitle_engine.cli.check_for_update", return_value=None) as mock_check:
        result = runner.invoke(app, ["update"])
        # The Typer app itself does not register ``update`` as a command; it is
        # routed via ``main_entry``. Invoking the app directly with ``update``
        # should therefore fail as an unknown command.
        assert result.exit_code != 0
        mock_check.assert_not_called()


def test_update_function_runs_upgrade_when_available():
    update_info = UpdateInfo(current=__version__, latest="9.9.9")
    with patch("subtitle_engine.cli.check_for_update", return_value=update_info) as mock_check:
        with patch("subtitle_engine.cli.update_package") as mock_upgrade:
            update()
            mock_check.assert_called_once_with(force=True)
            mock_upgrade.assert_called_once()


def test_update_function_reports_up_to_date():
    with patch("subtitle_engine.cli.check_for_update", return_value=None) as mock_check:
        with patch("subtitle_engine.cli.update_package") as mock_upgrade:
            update()
            mock_check.assert_called_once_with(force=True)
            mock_upgrade.assert_not_called()


def test_update_function_handles_check_error():
    with patch("subtitle_engine.cli.check_for_update", side_effect=UpdateCheckError("no network")):
        with pytest.raises(typer.Exit) as exc_info:
            update()
        assert exc_info.value.exit_code == 1


def test_main_entry_routes_update_command():
    with patch("subtitle_engine.cli.update") as mock_update:
        with patch.object(sys, "argv", ["subeng", "update"]):
            main_entry()
        mock_update.assert_called_once()


def test_main_entry_runs_typer_app_for_transcription():
    with patch("subtitle_engine.cli.app") as mock_app:
        with patch.object(sys, "argv", ["subeng", "video.mp4"]):
            main_entry()
        mock_app.assert_called_once_with(["main", "video.mp4"])


def test_main_entry_routes_caption_command():
    with patch("subtitle_engine.cli.app") as mock_app:
        with patch.object(sys, "argv", ["subeng", "caption", "file.srt"]):
            main_entry()
        mock_app.assert_called_once_with(["caption", "file.srt"])


def test_main_entry_handles_version_flag(capsys):
    with patch.object(sys, "argv", ["subeng", "--version"]):
        assert main_entry() is None
    captured = capsys.readouterr()
    assert captured.out.strip() == f"subeng {__version__}"


def test_cli_preset_shortform_accepted(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, ["main", str(media), "--preset", "shortform"])
    # Validation passes; transcription fails because the file is fake.
    assert result.exit_code != 0
    assert "Preset: shortform" in result.output


def test_cli_preset_longform_accepted(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, ["main", str(media), "--preset", "longform"])
    assert result.exit_code != 0
    assert "Preset: longform" in result.output


def test_cli_invalid_preset_rejected(tmp_path: Path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")
    result = runner.invoke(app, ["main", str(media), "--preset", "invalid"])
    assert result.exit_code != 0
    assert "Unknown preset" in result.output


def test_select_ollama_model_returns_chosen_model():
    mock_select = Mock(ask=Mock(return_value="model-b"))
    with patch("subtitle_engine.cli.list_models", return_value=["model-a", "model-b"]):
        with patch("subtitle_engine.cli.questionary.select", return_value=mock_select):
            assert _select_ollama_model("http://localhost:11434") == "model-b"


def test_select_ollama_model_empty_list_raises():
    with patch("subtitle_engine.cli.list_models", return_value=[]):
        with pytest.raises(ConnectionError, match="No Ollama models"):
            _select_ollama_model("http://localhost:11434")


def test_select_ollama_model_no_selection_raises():
    mock_select = Mock(ask=Mock(return_value=None))
    with patch("subtitle_engine.cli.list_models", return_value=["model-a"]):
        with patch("subtitle_engine.cli.questionary.select", return_value=mock_select):
            with pytest.raises(ValueError, match="No model selected"):
                _select_ollama_model("http://localhost:11434")


def test_caption_command_generates_caption(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n",
        encoding="utf-8",
    )
    mock_select = Mock(ask=Mock(return_value="qwen3.5:0.8b"))
    with patch("subtitle_engine.cli.list_models", return_value=["qwen3.5:0.8b"]):
        with patch("subtitle_engine.cli.questionary.select", return_value=mock_select):
            with patch("subtitle_engine.cli.generate_caption", return_value="A caption") as mock_generate:
                result = runner.invoke(app, ["caption", str(srt)])
    assert result.exit_code == 0
    mock_generate.assert_called_once()
    assert (tmp_path / "subs.caption.txt").read_text(encoding="utf-8") == "A caption"


def test_caption_command_uses_explicit_model(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n",
        encoding="utf-8",
    )
    with patch("subtitle_engine.cli.generate_caption", return_value="A caption") as mock_generate:
        result = runner.invoke(app, ["caption", str(srt), "--ollama-model", "llama3.2"])
    assert result.exit_code == 0
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "llama3.2"


def test_caption_command_custom_output(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n",
        encoding="utf-8",
    )
    output = tmp_path / "custom.txt"
    with patch("subtitle_engine.cli.generate_caption", return_value="A caption"):
        result = runner.invoke(app, ["caption", str(srt), "--ollama-model", "llama3.2", "--output", str(output)])
    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8") == "A caption"


def test_caption_command_no_models_raises(tmp_path: Path):
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello world\n",
        encoding="utf-8",
    )
    with patch("subtitle_engine.cli.list_models", return_value=[]):
        result = runner.invoke(app, ["caption", str(srt)])
    assert result.exit_code != 0
    assert "No Ollama models" in result.output
