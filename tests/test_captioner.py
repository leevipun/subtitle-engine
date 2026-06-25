"""Tests for the Ollama captioner module."""

from unittest.mock import Mock, patch

import pytest
import requests

from subtitle_engine.captioner import generate_caption, list_models


@patch("subtitle_engine.captioner.requests.post")
def test_generate_caption_success(mock_post):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"response": "  A short caption.  "},
        raise_for_status=lambda: None,
    )

    caption = generate_caption(
        "hello world",
        model="qwen3.5:0.8b",
        host="http://localhost:11434",
    )

    assert caption == "A short caption."
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "qwen3.5:0.8b"
    assert "hello world" in kwargs["json"]["prompt"]


@patch("subtitle_engine.captioner.requests.post")
def test_generate_caption_empty_response(mock_post):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"response": "   "},
        raise_for_status=lambda: None,
    )

    with pytest.raises(ValueError, match="empty caption"):
        generate_caption("hello", model="qwen3.5:0.8b")


def test_generate_caption_empty_transcript():
    with pytest.raises(ValueError, match="empty transcript"):
        generate_caption("   ", model="qwen3.5:0.8b")


@patch(
    "subtitle_engine.captioner.requests.post",
    side_effect=requests.ConnectionError("connection refused"),
)
def test_generate_caption_connection_error(mock_post):
    with pytest.raises(ConnectionError):
        generate_caption("hello", model="qwen3.5:0.8b")


@patch("subtitle_engine.captioner.requests.get")
def test_list_models(mock_get):
    mock_get.return_value = Mock(
        status_code=200,
        json=lambda: {"models": [{"name": "qwen3.5:0.8b"}, {"name": "llama3.2"}]},
        raise_for_status=lambda: None,
    )

    models = list_models()
    assert models == ["qwen3.5:0.8b", "llama3.2"]
