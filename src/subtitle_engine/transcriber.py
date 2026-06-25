"""WhisperX transcription wrapper."""

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Iterator, Optional

import torch
import whisperx


VALID_MODELS = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
VALID_DEVICES = {"cpu", "cuda"}


def _detect_device(device: Optional[str]) -> str:
    """Pick a device if none was specified.

    Note: MPS is intentionally excluded because the WhisperX backend
    (faster-whisper / CTranslate2) only supports CPU and CUDA.
    """
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _default_compute_type(device: str) -> str:
    """Pick a safe compute type for the device."""
    if device == "cpu":
        return "int8"
    return "float16"


def _validate_model(model_name: str) -> None:
    """Raise a ValueError if the model name is unknown."""
    if model_name not in VALID_MODELS:
        joined = ", ".join(sorted(VALID_MODELS))
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {joined}")


def _validate_device(device: str) -> None:
    """Raise a ValueError if the device name is unknown."""
    if device not in VALID_DEVICES:
        joined = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"Unknown device '{device}'. Choose from: {joined}")


@contextlib.contextmanager
def _suppress_external_output(verbose: bool) -> Iterator[None]:
    """Silence tqdm, warnings, and noisy loggers unless verbose is True."""
    if verbose:
        yield
        return

    original_tqdm_disable: Optional[bool] = None
    original_tqdm_env = os.environ.get("TQDM_DISABLE")
    try:
        import tqdm as _tqdm

        original_tqdm_disable = _tqdm.tqdm.disable
        _tqdm.tqdm.disable = True
    except Exception:  # noqa: BLE001
        _tqdm = None

    os.environ["TQDM_DISABLE"] = "1"

    loggers = [
        "whisperx",
        "faster_whisper",
        "pyannote.audio",
        "transformers",
        "torch",
    ]
    original_levels: dict[str, int] = {}
    for name in loggers:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(logging.WARNING)

    buf = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                yield
    finally:
        if original_tqdm_env is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = original_tqdm_env
        if original_tqdm_disable is not None and _tqdm is not None:
            _tqdm.tqdm.disable = original_tqdm_disable
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def transcribe(
    audio_path: Path,
    *,
    model_name: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    diarize: bool = False,
    hf_token: Optional[str] = None,
    verbose: bool = False,
) -> list[dict]:
    """Transcribe an audio/video file and return SRT-ready segments.

    Parameters
    ----------
    audio_path:
        Path to the media file to transcribe.
    model_name:
        WhisperX model size. One of: tiny, base, small, medium, large-v2, large-v3.
    language:
        ISO language code, e.g. ``en`` or ``fi``. If ``None``, WhisperX auto-detects.
    device:
        ``cpu``, ``cuda`` or ``mps``. Auto-detected if omitted.
    batch_size:
        WhisperX batch size for transcription.
    compute_type:
        ``int8`` or ``float16``. Auto-selected per device if omitted.
    diarize:
        Whether to run speaker diarization.
    hf_token:
        Hugging Face token required for diarization.

    Returns
    -------
    A list of segment dicts with ``start``, ``end`` and ``text`` keys.
    """
    _validate_model(model_name)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = _detect_device(device)
    _validate_device(device)

    if compute_type is None:
        compute_type = _default_compute_type(device)

    if diarize and not hf_token:
        raise ValueError("--hf-token is required when using --diarize")

    with _suppress_external_output(verbose):
        audio = whisperx.load_audio(str(audio_path))

        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        # Free transcription model memory before alignment
        del model

        detected_language = result.get("language")
        if detected_language:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language, device=device
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            del align_model

        if diarize:
            diarize_model = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return result["segments"]
