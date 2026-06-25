# subtitle-engine

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

## Development

Run the test suite:

```bash
pytest
```

## License

MIT
