"""Command-line interface for subtitle-engine."""

import sys
from pathlib import Path
from typing import Annotated, Optional

import questionary
import requests
import typer
from rich.console import Console

from subtitle_engine import __version__
from subtitle_engine.captioner import generate_caption, list_models
from subtitle_engine.segmenter import VALID_PRESETS, split_segments
from subtitle_engine.srt_writer import extract_text_from_srt, write_srt
from subtitle_engine.transcriber import transcribe
from subtitle_engine.updater import UpdateCheckError, check_for_update, update_package
from subtitle_engine.utils import resolve_output_path, validate_media_file

app = typer.Typer(
    help="Generate SRT subtitles from audio/video files using WhisperX",
    no_args_is_help=True,
)
console = Console()


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

        splitted_segments = split_segments(segments, preset=preset)
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
