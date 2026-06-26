"""Tests for the WhisperX transcription wrapper."""

import io
import logging
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest
import tqdm

from subtitle_engine.transcriber import (
    _patch_tqdm_for_progress,
    _suppress_external_output,
    transcribe,
)


def test_suppress_external_output_hides_noise(capsys):
    with _suppress_external_output(verbose=False):
        print("stdout noise")
        print("stderr noise", flush=True)
        warnings.warn("warning noise")
        logging.getLogger("whisperx").warning("logger noise")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_suppress_external_output_verbose_keeps_noise(capsys):
    with _suppress_external_output(verbose=True):
        print("visible output")

    captured = capsys.readouterr()
    assert "visible output" in captured.out


def test_progress_tqdm_reports_fraction():
    """Iterating a progress-tqdm bar should fire the callback with a rising fraction."""
    seen: list[float] = []

    def cb(_stage: str, fraction: float) -> None:
        seen.append(fraction)

    with _patch_tqdm_for_progress(cb):
        for _ in tqdm.trange(10, file=io.StringIO()):
            pass

    assert seen, "callback was never invoked"
    assert all(0.0 <= value <= 1.0 for value in seen)
    # Should be monotonic non-decreasing.
    assert seen == sorted(seen)
    # Final call should be 1.0.
    assert seen[-1] == 1.0


def test_progress_tqdm_skips_when_no_total():
    """Bars with an unknown total should not fire the callback for partial updates."""
    seen: list[float] = []

    def cb(_stage: str, fraction: float) -> None:
        seen.append(fraction)

    with _patch_tqdm_for_progress(cb):
        bar = tqdm.tqdm(file=io.StringIO(), total=None)
        bar.update(5)
        bar.update(5)
        bar.close()

    # No update-path calls should have fired because total is None.
    for value in seen:
        assert value == 0.0


def test_patch_tqdm_restores_original_on_exit():
    original = tqdm.tqdm
    with _patch_tqdm_for_progress(lambda *_: None):
        assert tqdm.tqdm is not original
    assert tqdm.tqdm is original


def test_patch_tqdm_restores_on_exception():
    original = tqdm.tqdm
    with pytest.raises(RuntimeError):
        with _patch_tqdm_for_progress(lambda *_: None):
            assert tqdm.tqdm is not original
            raise RuntimeError("boom")
    assert tqdm.tqdm is original


def test_patch_tqdm_is_noop_without_callback():
    original = tqdm.tqdm
    with _patch_tqdm_for_progress(None):
        assert tqdm.tqdm is original
    assert tqdm.tqdm is original


def test_transcribe_invokes_callback_with_stages(tmp_path):
    """transcribe() should report each stage to the progress callback."""
    audio = tmp_path / "video.mp4"
    audio.write_bytes(b"fake")

    events: list[tuple[str, float]] = []

    def callback(stage: str, fraction: float) -> None:
        events.append((stage, fraction))

    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"language": "en", "segments": []}

    with patch("subtitle_engine.transcriber.whisperx") as fake_whisperx:
        fake_whisperx.load_audio.return_value = MagicMock()
        fake_whisperx.load_model.return_value = fake_model
        fake_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        with patch("subtitle_engine.transcriber._suppress_external_output", _passthrough):
            transcribe(
                audio,
                model_name="small",
                progress_callback=callback,
            )

    stages = [stage for stage, _ in events]
    # Each pipeline stage should have been announced.
    for expected in (
        "loading_audio",
        "loading_model",
        "transcribing",
        "aligning",
        "done",
    ):
        assert expected in stages, f"missing stage {expected!r} in {stages!r}"

    # The first announce of each stage should be 0.0.
    for stage in ("loading_audio", "loading_model", "transcribing", "aligning"):
        assert (stage, 0.0) in events
    # Final announce should be 'done' at 1.0.
    assert events[-1] == ("done", 1.0)


def test_transcribe_forwards_native_callback_to_whisperx(tmp_path):
    """The transcribe/align progress callbacks passed to WhisperX must
    convert the 0-100 percent back into (stage, fraction) events."""
    audio = tmp_path / "video.mp4"
    audio.write_bytes(b"fake")

    forwarded: dict[str, list] = {"transcribe": [], "align": []}

    fake_model = MagicMock()

    def fake_transcribe(_audio, **kwargs):
        cb = kwargs.get("progress_callback")
        forwarded["transcribe"].append(cb)
        if cb is not None:
            cb(25.0)
            cb(75.0)
        return {"language": "en", "segments": []}

    fake_model.transcribe.side_effect = fake_transcribe

    fake_align = MagicMock()

    def fake_align_call(_segments, _model, _meta, _audio, _device, **kwargs):
        cb = kwargs.get("progress_callback")
        forwarded["align"].append(cb)
        if cb is not None:
            cb(50.0)
        return {"segments": []}

    with patch("subtitle_engine.transcriber.whisperx") as fake_whisperx:
        fake_whisperx.load_audio.return_value = MagicMock()
        fake_whisperx.load_model.return_value = fake_model
        fake_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        fake_whisperx.align.side_effect = fake_align_call
        with patch("subtitle_engine.transcriber._suppress_external_output", _passthrough):
            transcribe(
                audio,
                model_name="small",
                progress_callback=lambda *_: None,
            )

    assert forwarded["transcribe"][0] is not None, "transcribe should receive a progress_callback"
    assert forwarded["align"][0] is not None, "align should receive a progress_callback"


def test_transcribe_uninstalls_tqdm_on_error(tmp_path):
    audio = tmp_path / "video.mp4"
    audio.write_bytes(b"fake")

    original = tqdm.tqdm

    def boom(*_args, **_kwargs):
        raise RuntimeError("explode")

    with patch("subtitle_engine.transcriber.whisperx") as fake_whisperx:
        fake_whisperx.load_audio.return_value = MagicMock()
        fake_model = MagicMock()
        fake_model.transcribe.side_effect = boom
        fake_whisperx.load_model.return_value = fake_model
        with patch("subtitle_engine.transcriber._suppress_external_output", _passthrough):
            with pytest.raises(RuntimeError):
                transcribe(
                    audio,
                    model_name="small",
                    progress_callback=lambda *_: None,
                )

    assert tqdm.tqdm is original


def test_patch_tqdm_patches_lazy_faster_whisper_imports():
    """``faster_whisper`` submodules must end up using the progress subclass.

    Libraries like faster-whisper do ``from tqdm import tqdm`` (or the
    ``tqdm.auto`` variant) on first import, which would otherwise cache the
    *original* ``tqdm`` class on the submodule. The patch must force-load
    those submodules and re-bind their ``tqdm`` attribute so the subclass is
    used when whisperx lazily imports faster-whisper.
    """
    import importlib

    original_tqdm_class = tqdm.tqdm

    # Simulate a lazy faster-whisper import: none of these modules are loaded
    # at patch time, so the patch has to force-import them to swap their
    # cached ``tqdm`` references.
    for name in (
        "faster_whisper",
        "faster_whisper.transcribe",
        "faster_whisper.utils",
        "tqdm.asyncio",
        "tqdm.auto",
    ):
        sys.modules.pop(name, None)

    with _patch_tqdm_for_progress(lambda *_: None):
        faster_whisper = importlib.import_module("faster_whisper")
        faster_whisper_transcribe = importlib.import_module(
            "faster_whisper.transcribe"
        )
        faster_whisper_utils = importlib.import_module("faster_whisper.utils")
        tqdm_auto = importlib.import_module("tqdm.auto")

        # Inside the patch, the global ``tqdm.tqdm`` is the progress subclass.
        # The bindings cached on each submodule must also be the subclass, not
        # the original ``tqdm.std.tqdm`` (which is what ``from tqdm import
        # tqdm`` would have captured without our force-load).
        assert faster_whisper_transcribe.tqdm is not original_tqdm_class
        assert faster_whisper_utils.tqdm is not original_tqdm_class
        assert tqdm_auto.tqdm is not original_tqdm_class

    # After the patch, the global binding is restored to the original.
    assert tqdm.tqdm is original_tqdm_class


def test_transcribe_skips_patching_when_no_callback(tmp_path):
    audio = tmp_path / "video.mp4"
    audio.write_bytes(b"fake")

    original = tqdm.tqdm

    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"language": "en", "segments": []}

    with patch("subtitle_engine.transcriber.whisperx") as fake_whisperx:
        fake_whisperx.load_audio.return_value = MagicMock()
        fake_whisperx.load_model.return_value = fake_model
        fake_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        # While inside transcribe, tqdm.tqdm must equal the original — the
        # patch should be a no-op when no callback is supplied.
        with patch("subtitle_engine.transcriber._suppress_external_output", _passthrough):
            transcribe(audio, model_name="small")

    assert tqdm.tqdm is original


def _passthrough(verbose):  # noqa: ARG001
    """Context manager stand-in that does no suppression for tests."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    return _cm()
