"""Bounded content-addressed cache for validated scene-ranking responses."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from app.utils import utils

CACHE_VERSION = "scene-ranking-cache-v1"
MAX_CACHE_BYTES = 64_000
_NAME = re.compile(r"^ranking-v1-[0-9a-f]{64}\.json$")


def cache_dir(create: bool = True) -> Path:
    return Path(utils.storage_dir("cache_scene_rankings", create=create))


def identity(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def object_name(digest: str) -> str:
    return f"ranking-v1-{digest}.json"


def _remove(path: Path) -> None:
    try:
        if path.parent == cache_dir(create=False) and (
            path.is_file() or path.is_symlink()
        ):
            path.unlink(missing_ok=True)
    except OSError:
        pass


def load(digest: str, validator):
    directory = cache_dir(create=False)
    if directory.is_symlink():
        return None, True
    path = directory / object_name(digest)
    try:
        stat = path.lstat()
        if path.is_symlink() or not path.is_file() or stat.st_size > MAX_CACHE_BYTES:
            _remove(path)
            return None, True
        with path.open("rb") as stream:
            raw = stream.read(MAX_CACHE_BYTES + 1)
        if len(raw) > MAX_CACHE_BYTES:
            _remove(path)
            return None, True
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        if (
            set(payload) != {"cache_version", "identity", "response"}
            or payload["cache_version"] != CACHE_VERSION
            or payload["identity"] != digest
        ):
            raise ValueError("invalid cache record")
        return validator(payload["response"]), False
    except FileNotFoundError:
        return None, False
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        _remove(path)
        return None, True


def store(digest: str, response: dict) -> None:
    directory = cache_dir(create=True)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("ranking cache directory is invalid")
    data = json.dumps(
        {"cache_version": CACHE_VERSION, "identity": digest, "response": response},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(data) > MAX_CACHE_BYTES:
        raise ValueError("ranking cache record exceeds size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / object_name(digest))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
