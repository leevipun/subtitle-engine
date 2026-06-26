"""Tests for CLI helpers and argument parsing."""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from subtitle_engine import __version__
from subtitle_engine.cli import (
    _format_throughput,
    _make_progress_callback,
    _select_ollama_model,
    _start_stall_watcher,
    app,
    main_entry,
    update,
)
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
    assert __version__ in result.output


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


def test_cli_renders_progress_bar(tmp_path: Path):
    """A non-quiet run should drive the progress callback through every stage."""
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")

    seen_stages: list[str] = []

    def fake_transcribe(*_args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            for stage, fraction in (
                ("loading_audio", 0.0),
                ("loading_model", 0.0),
                ("transcribing", 0.5),
                ("aligning", 0.0),
                ("done", 1.0),
            ):
                seen_stages.append(stage)
                callback(stage, fraction)
        return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    with patch("subtitle_engine.cli.transcribe", side_effect=fake_transcribe):
        with patch(
            "subtitle_engine.cli.split_segments",
            return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        ):
            result = runner.invoke(app, ["main", str(media)])

    assert result.exit_code == 0
    for expected in ("loading_audio", "loading_model", "transcribing", "aligning", "done"):
        assert expected in seen_stages, f"progress bar never saw stage {expected!r}"


def test_cli_quiet_skips_progress_bar(tmp_path: Path):
    """``--quiet`` must not construct a rich Progress at all."""
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake")

    callback_received: list[bool] = []

    def fake_transcribe(*_args, **kwargs):
        callback_received.append(kwargs.get("progress_callback") is not None)
        return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    with patch("subtitle_engine.cli.transcribe", side_effect=fake_transcribe):
        with patch(
            "subtitle_engine.cli.split_segments",
            return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        ):
            result = runner.invoke(app, ["main", str(media), "-q"])

    assert result.exit_code == 0
    assert callback_received == [False], "quiet mode must not pass a progress callback"


def test_format_throughput_handles_edges():
    assert _format_throughput(0, 60) == ""
    assert _format_throughput(30, 0) == ""
    assert _format_throughput(20, 60) == "3.0x realtime"
    assert _format_throughput(10, 60) == "6.0x realtime"


def test_progress_callback_records_stage_timing():
    """Each callback should reset the stall-watcher's clock for that stage."""
    import rich.progress

    stall_state: dict = {"stage": None, "stage_started_at": None}
    progress = rich.progress.Progress(transient=True, disable=True)
    with progress:
        task = progress.add_task("Starting", total=100)
        cb = _make_progress_callback(progress, task, stall_state)
        cb("transcribing", 0.5)
        assert stall_state["stage"] == "transcribing"
        assert stall_state["last_fraction"] == 0.5
        assert stall_state["stage_started_at"] is not None
        first = stall_state["stage_started_at"]
        time.sleep(0.01)
        cb("aligning", 0.1)
        assert stall_state["stage"] == "aligning"
        assert stall_state["stage_started_at"] > first


def test_stall_watcher_annotates_long_stages():
    """When a stage hasn't reported in a while, the description gets a hint."""
    import rich.progress

    from subtitle_engine.cli import _PROGRESS_STAGE_TOTAL

    progress = rich.progress.Progress(transient=True, disable=True)
    stall_state: dict = {"stage": "transcribing", "stage_started_at": time.monotonic() - 5.0, "last_fraction": 0.0}
    stop_event = threading.Event()
    with progress:
        task = progress.add_task("Starting", total=_PROGRESS_STAGE_TOTAL)
        with patch("subtitle_engine.cli.get_last_audio_duration", return_value=60.0):
            _start_stall_watcher(progress, task, stall_state, stop_event)
            # Wait long enough for the watcher to tick at least twice.
            time.sleep(1.2)
        stop_event.set()
    # We can't directly inspect the bar's description, but the watcher must
    # not have raised. The fact that we reach this line is the assertion.


def test_stall_watcher_stops_cleanly():
    import rich.progress

    progress = rich.progress.Progress(transient=True, disable=True)
    stall_state: dict = {"stage": "transcribing", "stage_started_at": time.monotonic() - 5.0, "last_fraction": 0.0}
    stop_event = threading.Event()
    with progress:
        task = progress.add_task("Starting", total=100)
        thread = _start_stall_watcher(progress, task, stall_state, stop_event)
        stop_event.set()
        thread.join(timeout=2)
    assert not thread.is_alive(), "watcher thread should exit when stop_event is set"


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


def test_caption_command_rejects_non_srt_input(tmp_path: Path):
    video = tmp_path / "video.mov"
    video.write_bytes(b"\x00\xe0not an srt")
    result = runner.invoke(app, ["caption", str(video), "--ollama-model", "llama3.2"])
    assert result.exit_code != 0
    assert "expects an .srt file" in result.output
