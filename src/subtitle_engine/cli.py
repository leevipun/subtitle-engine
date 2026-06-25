"""Command-line interface for subtitle-engine."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from subtitle_engine.captioner import generate_caption
from subtitle_engine.srt_writer import write_srt
from subtitle_engine.transcriber import transcribe
from subtitle_engine.utils import resolve_output_path, validate_media_file

app = typer.Typer(
    help="Generate SRT subtitles from audio/video files using WhisperX",
    no_args_is_help=True,
)
console = Console()


@app.command()
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
            help="Device: cpu, cuda or mps. Auto-detected if omitted.",
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
            help="Ollama model for caption generation. Required if --caption is set.",
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
) -> None:
    """Generate SRT subtitles from a media file."""
    try:
        validate_media_file(input_file)
        output_path = resolve_output_path(input_file, output)

        if caption and not ollama_model:
            raise ValueError("--ollama-model is required when using --caption")

        console.print(f"[bold]Transcribing:[/bold] {input_file}")
        console.print(f"[bold]Model:[/bold] {model}")
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
        )

        write_srt(segments, output_path)
        console.print(f"[green]Wrote subtitles to:[/green] {output_path}")

        if caption:
            transcript = " ".join(str(segment.get("text", "")).strip() for segment in segments)
            caption_text = generate_caption(
                transcript,
                model=ollama_model,
                host=ollama_host,
            )
            caption_path = output_path.with_suffix(".caption.txt")
            caption_path.write_text(caption_text, encoding="utf-8")
            console.print(f"[green]Wrote caption to:[/green] {caption_path}")
    except (ValueError, FileNotFoundError, ConnectionError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Transcription failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
