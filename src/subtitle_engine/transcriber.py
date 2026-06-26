"""WhisperX transcription wrapper."""

import contextlib
import contextvars
import io
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import whisperx


VALID_MODELS = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
VALID_DEVICES = {"cpu", "cuda"}


# A callback signature for reporting transcription progress.
# The first argument is a pipeline stage label, the second is the
# intra-stage fraction in ``[0.0, 1.0]``.
ProgressCallback = Callable[[str, float], None]


# Modules that may have already imported ``tqdm`` (via ``from tqdm import tqdm``
# or ``from tqdm.auto import tqdm``) and cached a local reference. The progress
# patch must rebind the ``tqdm`` attribute on each loaded module so the
# subclass is actually used. Submodules matter because libraries like
# faster-whisper keep the binding on the submodule, not the top-level
# package.
_PROGRESS_TQDM_PATCH_MODULES = (
    "faster_whisper",
    "faster_whisper.transcribe",
    "faster_whisper.utils",
    "whisperx",
    "pyannote.audio",
    "transformers",
    "pyannote",
    "tqdm.auto",
)


def _iter_tqdm_modules():
    """Yield every loaded module that has a ``tqdm`` attribute to patch."""
    for name in _PROGRESS_TQDM_PATCH_MODULES:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "tqdm", None) is not None:
            yield mod

    # tqdm's own submodules cache ``tqdm`` as a local class reference
    # (e.g. ``tqdm.trange`` resolves ``tqdm`` through ``tqdm.std``'s scope).
    # Patch ``tqdm.std`` if it has been imported so internal helpers use
    # the progress subclass.
    std = sys.modules.get("tqdm.std")
    if std is not None and getattr(std, "tqdm", None) is not None:
        yield std


# Tracks the pipeline stage that the next progress callback should be labelled
# with. Read by the patched tqdm subclass so the callback can be labelled
# without threading the stage label through every callsite.
_progress_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "subtitle_engine_progress_stage", default=""
)


# Mutable side-channel for callers that need extra context (e.g. the audio
# duration so the CLI can compute a throughput hint during long stalls).
# Populated by ``transcribe`` and read by the CLI between stages.
_progress_meta: dict[str, float] = {}


def get_last_audio_duration() -> Optional[float]:
    """Return the audio duration of the most recent ``transcribe`` call, in seconds.

    The transcriber populates this once the audio is loaded. Returns ``None``
    if no transcription has run yet (or if the audio length is unknown).
    """
    return _progress_meta.get("audio_duration")


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


@contextlib.contextmanager
def _patch_tqdm_for_progress(callback: Optional[ProgressCallback]) -> Iterator[None]:
    """Install a progress-reporting tqdm subclass for the duration of the block.

    Yields immediately if ``callback`` is ``None``. Otherwise ``tqdm.tqdm`` is
    replaced (on the global ``tqdm`` module and on already-loaded modules that
    may have cached a reference via ``from tqdm import tqdm``) with a subclass
    that forwards every ``n / total`` update to ``callback``. The original
    ``tqdm`` is always restored on exit, even if the wrapped block raises.
    """
    if callback is None:
        yield
        return

    import tqdm as _tqdm

    class _ProgressTqdm(_tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._progress_callback = callback
            try:
                self._progress_callback(_progress_stage.get(), 0.0)
            except Exception:  # noqa: BLE001
                pass

        def _report_fraction(self) -> None:
            if self._progress_callback is None or not self.total:
                return
            try:
                self._progress_callback(
                    _progress_stage.get(),
                    min(self.n / self.total, 1.0),
                )
            except Exception:  # noqa: BLE001
                pass

        def update(self, n=1):
            result = super().update(n)
            self._report_fraction()
            return result

        def close(self):
            if self._progress_callback is not None and self.total:
                try:
                    self._progress_callback(_progress_stage.get(), 1.0)
                except Exception:  # noqa: BLE001
                    pass
            super().close()

        def __iter__(self):
            # tqdm's stock ``__iter__`` skips ``update`` for speed, which means
            # we would otherwise see no intra-bar progress for fast loops. Walk
            # the iterable manually so we can fire the callback on every step
            # while still delegating final display to ``close``.
            if self.disable or self.total is None:
                for obj in self.iterable:
                    yield obj
                return
            n = self.n
            total = self.total
            try:
                for obj in self.iterable:
                    yield obj
                    n += 1
                    self.n = n
                    try:
                        self._progress_callback(
                            _progress_stage.get(), min(n / total, 1.0)
                        )
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self.n = n
                self.close()

    original = _tqdm.tqdm
    _tqdm.tqdm = _ProgressTqdm

    # Force-load tqdm submodules that may be imported lazily by libraries like
    # faster-whisper. If they aren't in ``sys.modules`` yet, their ``from
    # tqdm.auto import tqdm`` line will see the *original* (unpatched) class
    # and cache it on the importing module, escaping our patch.
    for module_name in ("tqdm.asyncio", "tqdm.auto", "tqdm.autonotebook"):
        try:
            __import__(module_name)
        except Exception:  # noqa: BLE001
            pass

    patched_modules: list[tuple[object, object]] = []
    for mod in _iter_tqdm_modules():
        patched_modules.append((mod, mod.tqdm))
        mod.tqdm = _ProgressTqdm

    try:
        yield
    finally:
        _tqdm.tqdm = original
        for mod, original_attr in patched_modules:
            mod.tqdm = original_attr


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
    progress_callback: Optional[ProgressCallback] = None,
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
    progress_callback:
        Optional ``(stage: str, fraction: float) -> None`` callback. The
        ``stage`` is one of ``"loading_audio"``, ``"loading_model"``,
        ``"transcribing"``, ``"aligning"``, ``"diarizing"``. The ``fraction``
        is the intra-stage progress in ``[0.0, 1.0]`` and is sourced from
        WhisperX's native ``progress_callback`` on ``model.transcribe``,
        ``whisperx.align`` and the diarization pipeline, so it reflects the
        pipeline's true progress. Note: WhisperX's transcribe callback
        fires once per VAD-detected segment, so a single contiguous speech
        region will jump from 0% to 100% in one step.

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

    def _report_stage(stage: str) -> None:
        if progress_callback is None:
            return
        _progress_stage.set(stage)
        try:
            progress_callback(stage, 0.0)
        except Exception:  # noqa: BLE001
            pass

    def _emit(stage: str, percent: float) -> None:
        if progress_callback is None:
            return
        _progress_stage.set(stage)
        try:
            progress_callback(stage, max(0.0, min(percent / 100.0, 1.0)))
        except Exception:  # noqa: BLE001
            pass

    def _make_native_callback(stage: str):
        """Wrap a WhisperX-native callback (which gets 0-100) into our (stage, fraction) format."""
        def cb(percent: float) -> None:
            _emit(stage, float(percent))
        return cb

    with _suppress_external_output(verbose), _patch_tqdm_for_progress(progress_callback):
        _report_stage("loading_audio")
        audio = whisperx.load_audio(str(audio_path))
        audio_duration = float(audio.shape[0]) / 16000.0
        _emit("loading_audio", 1.0)
        if progress_callback is not None:
            _progress_meta["audio_duration"] = audio_duration

        _report_stage("loading_model")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

        _report_stage("transcribing")
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
            progress_callback=_make_native_callback("transcribing")
            if progress_callback is not None
            else None,
        )

        # Free transcription model memory before alignment
        del model

        detected_language = result.get("language")
        if detected_language:
            _report_stage("aligning")
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
                progress_callback=_make_native_callback("aligning")
                if progress_callback is not None
                else None,
            )
            del align_model

        if diarize:
            _report_stage("diarizing")
            diarize_model = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(
                audio,
                progress_callback=_make_native_callback("diarizing")
                if progress_callback is not None
                else None,
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

        if progress_callback is not None:
            try:
                progress_callback("done", 1.0)
            except Exception:  # noqa: BLE001
                pass

        return result["segments"]
