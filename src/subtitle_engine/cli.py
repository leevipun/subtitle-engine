"""Command-line interface for subtitle-engine."""

import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Callable, Optional

import questionary
import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from subtitle_engine import __version__
from subtitle_engine.captioner import generate_caption, list_models
from subtitle_engine.segmenter import (
    DEFAULT_MAX_CPS,
    DEFAULT_MIN_DURATION,
    SENTENCE_PAUSE_THRESHOLD,
    VALID_PRESETS,
    split_segments,
)
from subtitle_engine.srt_writer import extract_text_from_srt, write_srt
from subtitle_engine.transcriber import get_last_audio_duration, transcribe
from subtitle_engine.updater import UpdateCheckError, check_for_update, update_package
from subtitle_engine.utils import resolve_output_path, validate_media_file

app = typer.Typer(
    help="Generate SRT subtitles from audio/video files using WhisperX",
    no_args_is_help=True,
)
console = Console()


# Stage weights for the overall progress bar. The transcribe stage dominates
# so the bar moves most of its life during actual inference. These are rough
# defaults; adjust here if real-world profiling says otherwise.
_PROGRESS_STAGES: tuple[tuple[str, int], ...] = (
    ("loading_audio", 1),
    ("loading_model", 4),
    ("transcribing", 85),
    ("aligning", 7),
    ("diarizing", 3),
)
_PROGRESS_STAGE_WEIGHTS: dict[str, int] = dict(_PROGRESS_STAGES)
_PROGRESS_STAGE_CUMULATIVE: dict[str, float] = {}
_running = 0.0
for _name, _weight in _PROGRESS_STAGES:
    _PROGRESS_STAGE_CUMULATIVE[_name] = _running
    _running += _weight
_PROGRESS_STAGE_TOTAL = _running

_PROGRESS_DONE_LABEL = "Done"
_PROGRESS_FAILED_LABEL = "Failed"

# How long a stage can go without a callback before we add a throughput hint
# to the description. The bar still shows the *true* percentage; this just
# keeps the user informed that the process is alive during long stalls
# (e.g. WhisperX fires its transcribe callback only once per VAD segment).
_STALL_HINT_AFTER_SECONDS = 1.5
_STALL_TICK_SECONDS = 0.5


def _stage_display_name(stage: str) -> str:
    """Format a stage label for display in the progress bar."""
    if stage == "done":
        return _PROGRESS_DONE_LABEL
    return stage.replace("_", " ").capitalize()


def _format_throughput(processing_seconds: float, audio_seconds: float) -> str:
    """Return a compact ``Nx realtime`` hint for the stage label."""
    if processing_seconds <= 0 or audio_seconds <= 0:
        return ""
    ratio = audio_seconds / processing_seconds
    return f"{ratio:.1f}x realtime"


def _make_progress_callback(
    progress: Progress,
    task_id,
    stall_state: dict,
) -> Callable[[str, float], None]:
    """Build a (stage, fraction) -> None callback that drives ``progress``.

    ``stall_state`` is a shared dict the stall watcher reads to decide when to
    annotate a long-running stage with throughput information.
    """

    def callback(stage: str, fraction: float) -> None:
        if stage == "done":
            completed = _PROGRESS_STAGE_TOTAL
        else:
            base = _PROGRESS_STAGE_CUMULATIVE.get(stage, 0.0)
            weight = _PROGRESS_STAGE_WEIGHTS.get(stage, 0)
            # Clamp so the bar can never report more than 100% before the
            # pipeline actually completes.
            completed = min(base + max(0.0, min(fraction, 1.0)) * weight, 99.0)
        stall_state["stage"] = stage
        stall_state["stage_started_at"] = time.monotonic()
        stall_state["last_fraction"] = fraction
        progress.update(
            task_id,
            completed=completed,
            description=_stage_display_name(stage),
        )

    return callback


def _start_stall_watcher(
    progress: Progress,
    task_id,
    stall_state: dict,
    stop_event: threading.Event,
) -> threading.Thread:
    """Start a background thread that annotates long-stalled stages.

    The watcher only updates the *description*; the bar's *completed* value
    keeps reflecting the true (last-reported) progress so we never lie about
    how much of the work is actually done.
    """

    def run() -> None:
        while not stop_event.is_set():
            stop_event.wait(_STALL_TICK_SECONDS)
            if stop_event.is_set():
                return
            stage = stall_state.get("stage")
            if not stage or stage in ("done", "loading_audio", "loading_model"):
                continue
            started = stall_state.get("stage_started_at")
            if started is None:
                continue
            elapsed = time.monotonic() - started
            if elapsed < _STALL_HINT_AFTER_SECONDS:
                continue
            # Re-read on every tick: the audio duration is published by
            # ``transcribe`` *after* this thread starts, so we have to keep
            # polling for it to get a throughput ratio instead of a plain
            # seconds counter.
            audio_duration = get_last_audio_duration() or 0.0
            hint = _format_throughput(elapsed, audio_duration)
            base_label = _stage_display_name(stage)
            if hint:
                new_description = f"{base_label} ({hint})"
            else:
                new_description = f"{base_label} ({elapsed:.0f}s)"
            try:
                progress.update(task_id, description=new_description)
            except Exception:  # noqa: BLE001
                # The bar may have closed if the user hit Ctrl-C; ignore.
                return

    thread = threading.Thread(target=run, daemon=True, name="stall-watcher")
    thread.start()
    return thread


def _select_ollama_model(host: str) -> str:
    """List available Ollama models and prompt the user to pick one."""
    try:
        models = list_models(host)
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Could not connect to Ollama at {host}. Is Ollama running?"
        ) from exc

    if not models:
        raise ConnectionError(f"No Ollama models found at {host}.")

    choice = questionary.select(
        "Select an Ollama model:",
        choices=models,
        default=models[0],
    ).ask()

    if choice is None:
        raise ValueError("No model selected")

    return choice


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"subeng {__version__}")
        raise typer.Exit()


def update() -> None:
    """Update subeng to the latest version from PyPI."""
    try:
        update_info = check_for_update(force=True)
    except UpdateCheckError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if update_info is None:
        console.print(f"[green]subeng is up to date ({__version__}).[/green]")
        return

    console.print(
        f"[bold]A new version is available:[/bold] "
        f"{update_info.current} → {update_info.latest}"
    )
    console.print("[bold]Updating...[/bold]")
    try:
        update_package()
    except UpdateCheckError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]Update complete.[/green]")


def main_entry() -> None:
    """Route subcommands; default to the ``main`` transcription command."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-v", "--version"):
        console.print(f"subeng {__version__}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        update()
        return

    # If the user did not supply a subcommand (or global option), default to ``main``.
    args = sys.argv.copy()
    if (
        len(args) > 1
        and not args[1].startswith("-")
        and args[1] not in ("main", "caption")
    ):
        args.insert(1, "main")
    app(args[1:])


@app.command(
    name="main",
    epilog="Run 'subeng update' to update to the latest version.",
)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Audio or video file to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output SRT file (default: <input>.srt)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="WhisperX model: tiny, base, small, medium, large-v2, large-v3",
        ),
    ] = "small",
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Language code, e.g. en, fi. Auto-detected if omitted.",
        ),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option(
            "--device",
            "-d",
            help="Device: cpu or cuda. Auto-detected if omitted.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="WhisperX inference batch size",
            min=1,
        ),
    ] = 16,
    compute_type: Annotated[
        Optional[str],
        typer.Option(
            "--compute-type",
            "-c",
            help="Compute type: int8 or float16. Auto-selected if omitted.",
        ),
    ] = None,
    diarize: Annotated[
        bool,
        typer.Option(
            "--diarize",
            help="Run speaker diarization (requires --hf-token)",
        ),
    ] = False,
    hf_token: Annotated[
        Optional[str],
        typer.Option(
            "--hf-token",
            help="Hugging Face token for diarization",
            envvar="HF_TOKEN",
        ),
    ] = None,
    caption: Annotated[
        bool,
        typer.Option(
            "--caption",
            help="Generate a caption from the transcript using Ollama",
        ),
    ] = False,
    ollama_model: Annotated[
        Optional[str],
        typer.Option(
            "--ollama-model",
            help="Ollama model for caption generation. If omitted, installed models are listed.",
        ),
    ] = None,
    ollama_host: Annotated[
        str,
        typer.Option(
            "--ollama-host",
            help="Ollama API host",
            envvar="OLLAMA_HOST",
        ),
    ] = "http://localhost:11434",
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            "-p",
            help="Subtitle style: shortform (2-5 words) or longform (10-14 words).",
        ),
    ] = "shortform",
    no_cleanup: Annotated[
        bool,
        typer.Option(
            "--no-cleanup",
            help="Disable hallucination cleanup (e.g. 'Thanks for watching', [Music]).",
        ),
    ] = False,
    no_clause_boundaries: Annotated[
        bool,
        typer.Option(
            "--no-clause-boundaries",
            help="Disable clause-aware line breaking.",
        ),
    ] = False,
    no_sentence_split: Annotated[
        bool,
        typer.Option(
            "--no-sentence-split",
            help=(
                "Disable splitting at sentence boundaries (. ! ?). "
                "When off, a subtitle will not contain a stranded period "
                "in the middle (e.g. 'mission. Building in public' becomes "
                "'mission.' / 'Building in public'). Known abbreviations "
                "(Mr., Dr., U.S., e.g., …) are skipped."
            ),
        ),
    ] = False,
    sentence_pause_threshold: Annotated[
        float,
        typer.Option(
            "--sentence-pause-threshold",
            help=(
                "Minimum gap (seconds) between a period-ending word and the "
                "next word for the gap to count as evidence of a real "
                "sentence boundary. Lower = more aggressive splitting. "
                "Set to 0 to require a capital letter or speaker change."
            ),
            min=0.0,
        ),
    ] = SENTENCE_PAUSE_THRESHOLD,
    no_line_balance: Annotated[
        bool,
        typer.Option(
            "--no-line-balance",
            help="Disable two-line balancing (enabled by default for longform).",
        ),
    ] = False,
    max_cps: Annotated[
        float,
        typer.Option(
            "--max-cps",
            help="Maximum characters per second per subtitle. Use 0 to disable.",
            min=0.0,
        ),
    ] = DEFAULT_MAX_CPS,
    no_cps_limit: Annotated[
        bool,
        typer.Option(
            "--no-cps-limit",
            help="Disable the characters-per-second post-processor.",
        ),
    ] = False,
    min_duration: Annotated[
        float,
        typer.Option(
            "--min-duration",
            help="Minimum on-screen duration per subtitle, in seconds. Use 0 to disable.",
            min=0.0,
        ),
    ] = DEFAULT_MIN_DURATION,
    no_min_duration: Annotated[
        bool,
        typer.Option(
            "--no-min-duration",
            help="Disable the minimum-duration post-processor.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Only print errors.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show WhisperX progress bars and warnings.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Generate SRT subtitles from a media file."""
    try:
        if preset not in VALID_PRESETS:
            valid = ", ".join(sorted(VALID_PRESETS))
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {valid}")

        validate_media_file(input_file)
        output_path = resolve_output_path(input_file, output)

        if caption and not ollama_model:
            ollama_model = _select_ollama_model(ollama_host)

        if not quiet:
            update_info = check_for_update()
            if update_info:
                console.print(
                    f"[yellow]A new version of subeng is available:[/yellow] "
                    f"{update_info.current} → {update_info.latest}. "
                    f"Run [bold]subeng update[/bold] to install it."
                )

        if not quiet:
            console.print(f"[bold]Transcribing:[/bold] {input_file}")
            console.print(f"[bold]Model:[/bold] {model}")
            console.print(f"[bold]Preset:[/bold] {preset}")
            if language:
                console.print(f"[bold]Language:[/bold] {language}")
            if device:
                console.print(f"[bold]Device:[/bold] {device}")

        if quiet:
            segments = transcribe(
                input_file,
                model_name=model,
                language=language,
                device=device,
                batch_size=batch_size,
                compute_type=compute_type,
                diarize=diarize,
                hf_token=hf_token,
                verbose=verbose,
            )
        else:
            # Build a Console that targets the *original* stdout. The
            # transcriber's ``_suppress_external_output`` swaps ``sys.stdout``
            # for a buffer to hide WhisperX noise, and Rich's default Console
            # follows ``sys.stdout``, which would make the bar invisible. By
            # capturing the file beforehand we keep the bar visible on the
            # terminal while the noise still gets swallowed.
            progress_console = Console(file=sys.stdout, force_terminal=True)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=30),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=progress_console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("Starting", total=_PROGRESS_STAGE_TOTAL)
                stall_state: dict = {"stage": None, "stage_started_at": None}
                stop_event = threading.Event()
                _start_stall_watcher(progress, task_id, stall_state, stop_event)
                progress_callback = _make_progress_callback(
                    progress, task_id, stall_state
                )
                try:
                    segments = transcribe(
                        input_file,
                        model_name=model,
                        language=language,
                        device=device,
                        batch_size=batch_size,
                        compute_type=compute_type,
                        diarize=diarize,
                        hf_token=hf_token,
                        verbose=verbose,
                        progress_callback=progress_callback,
                    )
                except Exception:
                    stop_event.set()
                    progress.update(
                        task_id,
                        description=_PROGRESS_FAILED_LABEL,
                        completed=_PROGRESS_STAGE_TOTAL,
                    )
                    # Switch off transient so the failed bar remains visible.
                    progress.transient = False
                    raise
                stop_event.set()
                progress.update(
                    task_id,
                    completed=_PROGRESS_STAGE_TOTAL,
                    description=_PROGRESS_DONE_LABEL,
                )

        splitted_segments = split_segments(
            segments,
            preset=preset,
            cleanup=not no_cleanup,
            use_clause_boundaries=not no_clause_boundaries,
            split_at_sentences=not no_sentence_split,
            sentence_pause_threshold=sentence_pause_threshold,
            balance_lines=not no_line_balance,
            max_cps=max_cps,
            min_duration=min_duration,
            enforce_cps=not no_cps_limit,
            enforce_min_duration=not no_min_duration,
        )
        write_srt(splitted_segments, output_path)
        if not quiet:
            console.print(f"[green]Wrote subtitles to:[/green] {output_path}")

        if caption:
            transcript = " ".join(str(segment.get("text", "")).strip() for segment in splitted_segments)
            caption_text = generate_caption(
                transcript,
                model=ollama_model,
                host=ollama_host,
            )
            caption_path = output_path.with_suffix(".caption.txt")
            caption_path.write_text(caption_text, encoding="utf-8")
            if not quiet:
                console.print(f"[green]Wrote caption to:[/green] {caption_path}")
    except (ValueError, FileNotFoundError, ConnectionError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Transcription failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="caption")
def caption_command(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="SRT file to generate a caption from",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output caption file (default: <input>.caption.txt)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    ollama_model: Annotated[
        Optional[str],
        typer.Option(
            "--ollama-model",
            "-m",
            help="Ollama model for caption generation",
        ),
    ] = None,
    ollama_host: Annotated[
        str,
        typer.Option(
            "--ollama-host",
            help="Ollama API host",
            envvar="OLLAMA_HOST",
        ),
    ] = "http://localhost:11434",
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Only print errors.",
        ),
    ] = False,
) -> None:
    """Generate a caption from an existing SRT file."""
    try:
        if input_file.suffix.lower() != ".srt":
            raise ValueError(
                f"caption expects an .srt file, got '{input_file.suffix}'. "
                "Use 'subeng main <video> --caption' to generate a caption from a video file."
            )

        if not ollama_model:
            ollama_model = _select_ollama_model(ollama_host)

        transcript = extract_text_from_srt(input_file)
        caption_text = generate_caption(
            transcript,
            model=ollama_model,
            host=ollama_host,
        )

        caption_path = output or input_file.with_suffix(".caption.txt")
        caption_path.write_text(caption_text, encoding="utf-8")
        if not quiet:
            console.print(f"[green]Wrote caption to:[/green] {caption_path}")
    except (ValueError, FileNotFoundError, ConnectionError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Caption generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main_entry()
