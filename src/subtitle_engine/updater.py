"""Update checker and in-place updater for subtitle-engine."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from subtitle_engine import __version__

PYPI_PACKAGE_NAME = "subtitle-engine"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
CACHE_DIR = Path.home() / ".cache" / "subeng"
CACHE_FILE = CACHE_DIR / "update_check.json"
CACHE_TTL = timedelta(days=1)
REQUEST_TIMEOUT = 5  # seconds


@dataclass(frozen=True)
class UpdateInfo:
    """Information about an available update."""

    current: str
    latest: str


class UpdateCheckError(Exception):
    """Raised when the update check fails."""


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a version string to a comparable tuple of integers.

    Non-numeric parts are stripped, so ``1.2.3a1`` is treated like ``1.2.3``.
    """
    parts: list[int] = []
    for part in version.split("."):
        numeric = ""
        for ch in part:
            if ch.isdigit():
                numeric += ch
            else:
                break
        parts.append(int(numeric) if numeric else 0)
    return tuple(parts)


def is_update_available(current: str, latest: str) -> bool:
    """Return ``True`` if ``latest`` is newer than ``current``."""
    return _version_key(latest) > _version_key(current)


def _cache_path() -> Path:
    """Return the path to the update-check cache file, creating the directory if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_FILE


def _read_cached_check() -> Optional[UpdateInfo]:
    """Read the cached update check if it exists and is still fresh."""
    cache = _cache_path()
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        checked_at = datetime.fromisoformat(data["checked_at"])
        if datetime.now(timezone.utc) - checked_at > CACHE_TTL:
            return None
        latest = data["latest"]
        if is_update_available(__version__, latest):
            return UpdateInfo(current=__version__, latest=latest)
        return None
    except (KeyError, ValueError, OSError):
        return None


def _write_cached_check(latest: str) -> None:
    """Write the result of an update check to the cache."""
    cache = _cache_path()
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "latest": latest,
    }
    try:
        cache.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        # Caching is best-effort; never fail the CLI because of it.
        pass


def fetch_latest_version(timeout: int = REQUEST_TIMEOUT) -> str:
    """Fetch the latest released version from PyPI.

    Raises:
        UpdateCheckError: if the request fails or the response is unusable.
    """
    request = Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"subeng/{__version__}"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateCheckError(f"Could not check for updates: {exc}") from exc

    try:
        return str(data["info"]["version"])
    except (KeyError, TypeError) as exc:
        raise UpdateCheckError(f"Unexpected PyPI response: {exc}") from exc


def check_for_update(force: bool = False) -> Optional[UpdateInfo]:
    """Check whether a newer version is available on PyPI.

    The result is cached for ``CACHE_TTL`` (one day) unless ``force`` is ``True``.
    Network failures are swallowed unless ``force`` is ``True``.

    Returns:
        ``UpdateInfo`` if a newer version exists, otherwise ``None``.
    """
    if not force:
        cached = _read_cached_check()
        if cached is not None:
            return cached

    try:
        latest = fetch_latest_version()
    except UpdateCheckError:
        if force:
            raise
        return None

    _write_cached_check(latest)

    if is_update_available(__version__, latest):
        return UpdateInfo(current=__version__, latest=latest)
    return None


def update_package() -> None:
    """Upgrade subtitle-engine in the current Python environment using pip.

    Raises:
        UpdateCheckError: if the pip command fails.
    """
    command = [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE_NAME]
    try:
        subprocess.run(command, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        raise UpdateCheckError(f"Update failed (exit code {exc.returncode}).") from exc
    except FileNotFoundError as exc:
        raise UpdateCheckError(f"Could not run pip: {exc}") from exc
