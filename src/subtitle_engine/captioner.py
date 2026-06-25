"""Generate captions from transcripts using a local Ollama instance."""

import json
from typing import Optional

import requests


def _default_prompt(transcript: str) -> str:
    """Build the prompt sent to the LLM."""
    return (
        "Create a short, engaging caption (1-2 sentences) for a video based on the following transcript. "
        "Write the caption in the same language as the transcript. "
        "Answer directly with the caption only, without any thinking or explanation.\n\n"
        f"Transcript:\n{transcript}"
    )


def generate_caption(
    transcript: str,
    *,
    model: str,
    host: str = "http://localhost:11434",
    prompt: Optional[str] = None,
) -> str:
    """Generate a caption from a transcript via Ollama.

    Parameters
    ----------
    transcript:
        The transcript text to summarize.
    model:
        Name of the Ollama model to use.
    host:
        Base URL of the Ollama API.
    prompt:
        Custom prompt. A default prompt is used if omitted.

    Returns
    -------
    The generated caption string.
    """
    if not transcript.strip():
        raise ValueError("Cannot generate a caption from an empty transcript")

    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt or _default_prompt(transcript),
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to Ollama at {host}. Is Ollama running?"
        ) from exc

    response.raise_for_status()
    data = response.json()
    caption = data.get("response", "").strip()

    if not caption:
        raise ValueError(
            "Ollama returned an empty caption. "
            "This can happen with some models or languages — try a different --ollama-model."
        )

    return caption


def list_models(host: str = "http://localhost:11434") -> list[str]:
    """Return the names of models available in the local Ollama instance."""
    url = f"{host.rstrip('/')}/api/tags"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]
