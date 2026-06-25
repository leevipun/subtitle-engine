"""Tests for the update checker and updater."""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitle_engine import __version__
from subtitle_engine import updater
from subtitle_engine.updater import (
    UpdateCheckError,
    UpdateInfo,
    check_for_update,
    fetch_latest_version,
    is_update_available,
    update_package,
)


@pytest.fixture
def fresh_cache(tmp_path: Path, monkeypatch):
    """Redirect the update-check cache to a temporary directory."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(updater, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(updater, "CACHE_FILE", cache_dir / "update_check.json")
    return cache_dir


def test_is_update_available_when_latest_is_newer():
    assert is_update_available("0.1.0", "0.2.0") is True
    assert is_update_available("0.1.1", "0.1.2") is True


def test_is_update_available_when_latest_is_older_or_equal():
    assert is_update_available("0.2.0", "0.1.0") is False
    assert is_update_available("0.1.1", "0.1.1") is False


def test_is_update_available_ignores_non_numeric_suffixes():
    assert is_update_available("0.1.1", "0.1.2a1") is True
    assert is_update_available("0.1.2", "0.1.2a1") is False


def test_fetch_latest_version_parses_pypi_response():
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"info": {"version": "9.9.9"}}).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("subtitle_engine.updater.urlopen", return_value=mock_context):
        assert fetch_latest_version() == "9.9.9"


def test_fetch_latest_version_raises_on_bad_response():
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"info": {}}).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("subtitle_engine.updater.urlopen", return_value=mock_context):
        with pytest.raises(UpdateCheckError):
            fetch_latest_version()


def test_fetch_latest_version_raises_on_network_error():
    with patch("subtitle_engine.updater.urlopen", side_effect=OSError("no network")):
        with pytest.raises(UpdateCheckError):
            fetch_latest_version()


def test_check_for_update_returns_info_when_update_available(fresh_cache: Path):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"info": {"version": "9.9.9"}}).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("subtitle_engine.updater.urlopen", return_value=mock_context):
        result = check_for_update()

    assert result == UpdateInfo(current=__version__, latest="9.9.9")
    assert (fresh_cache / "update_check.json").exists()


def test_check_for_update_uses_cache_instead_of_fetching(fresh_cache: Path):
    fresh_cache.mkdir(parents=True, exist_ok=True)
    cache = fresh_cache / "update_check.json"
    cache.write_text(
        json.dumps(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "latest": "9.9.9",
            }
        ),
        encoding="utf-8",
    )

    with patch("subtitle_engine.updater.urlopen") as mock_urlopen:
        result = check_for_update()
        mock_urlopen.assert_not_called()

    assert result == UpdateInfo(current=__version__, latest="9.9.9")


def test_check_for_update_cache_ignores_stale_entries(fresh_cache: Path):
    fresh_cache.mkdir(parents=True, exist_ok=True)
    cache = fresh_cache / "update_check.json"
    stale_time = datetime.now(timezone.utc) - timedelta(days=2)
    cache.write_text(
        json.dumps(
            {
                "checked_at": stale_time.isoformat(),
                "latest": "9.9.9",
            }
        ),
        encoding="utf-8",
    )

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"info": {"version": "9.9.10"}}).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("subtitle_engine.updater.urlopen", return_value=mock_context):
        result = check_for_update()

    assert result == UpdateInfo(current=__version__, latest="9.9.10")


def test_check_for_update_returns_none_when_up_to_date(fresh_cache: Path):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"info": {"version": __version__}}).encode("utf-8")
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("subtitle_engine.updater.urlopen", return_value=mock_context):
        result = check_for_update()

    assert result is None


def test_check_for_update_swallows_network_errors_by_default(fresh_cache: Path):
    with patch("subtitle_engine.updater.urlopen", side_effect=OSError("no network")):
        result = check_for_update()
    assert result is None


def test_check_for_update_force_re_raises_network_errors(fresh_cache: Path):
    with patch("subtitle_engine.updater.urlopen", side_effect=OSError("no network")):
        with pytest.raises(UpdateCheckError):
            check_for_update(force=True)


def test_update_package_runs_pip_upgrade():
    with patch("subtitle_engine.updater.subprocess.run") as mock_run:
        update_package()
        mock_run.assert_called_once()
        command = mock_run.call_args[0][0]
        assert command[0] == sys.executable
        assert command[1:] == ["-m", "pip", "install", "--upgrade", "subtitle-engine"]


def test_update_package_raises_on_pip_failure():
    with patch(
        "subtitle_engine.updater.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["pip"]),
    ):
        with pytest.raises(UpdateCheckError):
            update_package()
