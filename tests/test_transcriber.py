"""Tests for the WhisperX transcription wrapper."""

import logging
import warnings

from subtitle_engine.transcriber import _suppress_external_output


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
