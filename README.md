# subtitle-engine

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/subtitle-engine?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/subtitle-engine)

Generate `.srt` subtitle files from audio or video files using [WhisperX](https://github.com/m-bain/whisperX). Optionally generate a caption from the transcript with a local [Ollama](https://ollama.com/) LLM.

## Installation

Requires Python 3.12 or newer.

```bash
pip install subtitle-engine
```

Or install from source:

```bash
git clone https://github.com/leevipuntanen/subtitle-engine.git
cd subtitle-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Basic usage — writes <input>.srt next to the source file
subeng video.mp4

# Specify output file
subeng video.mp4 --output subtitles.srt

# Use a different model or language
subeng video.mp4 --model medium --language fi

# Force CPU / CUDA
subeng video.mp4 --device cpu

# Speaker diarization (requires a Hugging Face token)
subeng video.mp4 --diarize --hf-token $HF_TOKEN

# Generate a caption from the transcript using Ollama
subeng video.mp4 --caption --ollama-model qwen3.5:0.6b

# Generate a caption from an existing SRT file
subeng caption subtitles.srt

# Short-form subtitles (2-5 words per line, default)
subeng video.mp4 --preset shortform

# Long-form subtitles (10-14 words per line)
subeng video.mp4 --preset longform
```

## Subtitle quality

The segmenter applies several quality passes by default. Each can be turned off
or tuned via a CLI flag.

| Pass | Default | Flag to disable | Flag to tune |
|------|---------|-----------------|--------------|
| Hallucination cleanup (`Thanks for watching`, `[Music]`, `Subtitles by …`) | on | `--no-cleanup` | — |
| Clause-aware line breaking (prefer splits at `,` `.` `?` `!` `;` `:`) | on | `--no-clause-boundaries` | — |
| Sentence-boundary splitting (split at `.` `?` `!` when at least one of: next word capitalized, ≥ 0.2s pause after, or speaker change) | on | `--no-sentence-split` | `--sentence-pause-threshold 0.3` |
| Two-line balancing (longform only — splits long lines at the midpoint clause) | on (longform) | `--no-line-balance` | — |
| Characters-per-second limit | 21 cps | `--no-cps-limit` | `--max-cps 17` |
| Minimum on-screen duration | 1.0 s | `--no-min-duration` | `--min-duration 1.5` |

The sentence splitter now considers context before splitting, so a mid-thought
period like `I found a bug. reports say...` stays in the same subtitle
(no capital, no pause, no speaker change → no split). To get the old
aggressive behavior back, set `--sentence-pause-threshold 0`.

Disable every quality pass to reproduce the legacy behavior:

```bash
subeng video.mp4 \
  --no-cleanup \
  --no-clause-boundaries \
  --no-sentence-split \
  --no-line-balance \
  --no-cps-limit \
  --no-min-duration
```

## Options

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output SRT file path |
| `--model`, `-m` | WhisperX model: `tiny`, `base`, `small` (default), `medium`, `large-v2`, `large-v3` |
| `--language`, `-l` | ISO language code, e.g. `en`, `fi`. Auto-detected if omitted. |
| `--device`, `-d` | `cpu` or `cuda`. Auto-detected if omitted. |
| `--batch-size`, `-b` | Inference batch size (default: 16) |
| `--compute-type`, `-c` | `int8` or `float16`. Auto-selected if omitted. |
| `--diarize` | Enable speaker diarization |
| `--hf-token` | Hugging Face token for diarization (or set `HF_TOKEN` env var) |
| `--caption` | Generate a caption from the transcript via Ollama |
| `--ollama-model` | Ollama model name. If omitted, installed models are listed and you can pick one. |
| `--ollama-host` | Ollama API host (default: `http://localhost:11434`) |
| `caption` | Generate a caption from an existing SRT file (e.g. `subeng caption file.srt`) |
| `--preset`, `-p` | Subtitle style: `shortform` (2-5 words, default) or `longform` (10-14 words) |
| `--no-cleanup` | Disable hallucination cleanup (e.g. `Thanks for watching`, `[Music]`) |
| `--no-clause-boundaries` | Disable clause-aware line breaking |
| `--no-sentence-split` | Disable sentence-boundary splitting (so `mission. Building in public` stays as one subtitle) |
| `--sentence-pause-threshold <float>` | Minimum pause (seconds) after a period for the gap to count as evidence of a real sentence boundary (default: 0.2). Set to `0` to rely on capital letters / speaker change only. |
| `--no-line-balance` | Disable two-line balancing (enabled by default for longform) |
| `--max-cps <float>` | Maximum characters per second per subtitle (default: 21). Use `0` to disable. |
| `--no-cps-limit` | Disable the CPS post-processor |
| `--min-duration <float>` | Minimum on-screen duration in seconds (default: 1.0). Use `0` to disable. |
| `--no-min-duration` | Disable the minimum-duration post-processor |

## Development

Run the test suite:

```bash
pytest
```

## License

MIT
