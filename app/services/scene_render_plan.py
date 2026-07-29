"""Bind narration timing to selected scene footage without downloading media."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
TIMING_TOLERANCE = 0.001
SELECTED_STATUSES = {
    "provider_rank_selected",
    "provider_rank_fallback",
    "vlm_selected",
}
UNRESOLVED_STATUSES = {"no_candidates", "no_safe_candidate"}
KNOWN_STATUSES = SELECTED_STATUSES | UNRESOLVED_STATUSES | {"hold_no_search"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SceneRenderBinding(str, Enum):
    selected = "selected"
    reused = "reused"
    fallback_required = "fallback_required"


class SceneFallbackReason(str, Enum):
    no_candidates = "no_candidates"
    no_safe_candidate = "no_safe_candidate"
    missing_selection = "missing_selection"
    unavailable_url = "unavailable_url"
    reuse_target_unresolved = "reuse_target_unresolved"


def _fail(message: str) -> None:
    raise ValueError(message)


def _strict_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} is invalid")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} is invalid")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} is invalid")
    return result


def _string(value, name: str, maximum: int, *, allow_empty: bool = False) -> str:
    minimum = 0 if allow_empty else 1
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or _CONTROL.search(value)
    ):
        _fail(f"{name} is invalid")
    return value


def _read_manifest(path: str, name: str) -> tuple[Path, bytes]:
    if not isinstance(path, str) or not path or "\x00" in path:
        _fail(f"{name} path is invalid")
    source = Path(path).absolute()
    try:
        before = source.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if source.is_symlink() or not stat.S_ISREG(before.st_mode):
        _fail(f"{name} must be a regular file")
    if before.st_size <= 0 or before.st_size > MAX_MANIFEST_BYTES:
        _fail(f"{name} size is outside the supported range")
    try:
        with source.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                _fail(f"{name} changed while being opened")
            data = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if not data or len(data) > MAX_MANIFEST_BYTES:
        _fail(f"{name} size is outside the supported range")
    return source, data


def _parse_json(data: bytes, name: str):
    def reject_constant(_value: str):
        _fail(f"{name} contains a non-finite number")

    try:
        return json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc


def _validate_timeline(payload) -> list[dict]:
    if not isinstance(payload, list) or not payload:
        _fail("scene manifest must contain a non-empty array")
    scenes: list[dict] = []
    previous_end: float | None = None
    for position, item in enumerate(payload, 1):
        if not isinstance(item, dict) or set(item) != {
            "index",
            "start_time",
            "end_time",
            "duration",
            "text",
        }:
            _fail("scene manifest has invalid fields")
        index = _strict_int(item["index"], "scene index")
        if index != position:
            _fail("scene indexes must be contiguous and in source order")
        start = _number(item["start_time"], "scene start_time")
        end = _number(item["end_time"], "scene end_time")
        duration = _number(item["duration"], "scene duration")
        text = _string(item["text"], "scene text", 20_000, allow_empty=True)
        if start < 0 or end <= start or duration <= 0:
            _fail("scene timing is invalid")
        if abs(duration - (end - start)) > TIMING_TOLERANCE:
            _fail("scene duration is inconsistent")
        if previous_end is None:
            if abs(start) > TIMING_TOLERANCE:
                _fail("scene coverage must begin at zero")
        elif abs(start - previous_end) > TIMING_TOLERANCE:
            _fail("scene coverage has a gap or overlap")
        scenes.append(
            {
                "scene_index": index,
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "duration": item["duration"],
                "text": text,
            }
        )
        previous_end = end
    return scenes


def _validate_binding(value, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"version", "sha256"}:
        _fail(f"{name} is invalid")
    if _strict_int(value["version"], f"{name} version") != 1:
        _fail(f"{name} version is unsupported")
    digest = _string(value["sha256"], f"{name} sha256", 64)
    if not _SHA256.fullmatch(digest):
        _fail(f"{name} sha256 is invalid")


def _candidate_identity(scene: dict) -> dict:
    selected_id = scene.get("selected_candidate_id")
    selected = scene.get("selected_candidate")
    candidates = scene.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        _fail("selected candidate identity is incomplete")
    candidate_id = _string(selected_id, "selected_candidate_id", 320)
    if _string(selected.get("candidate_id"), "candidate_id", 320) != candidate_id:
        _fail("selected candidate IDs are inconsistent")
    provider = _string(selected.get("provider"), "selected provider", 32)
    provider_video_id = _string(
        selected.get("provider_video_id"), "provider_video_id", 256
    )
    matches = 0
    for candidate in candidates:
        if not isinstance(candidate, dict) or "candidate_id" not in candidate:
            _fail("candidate row is structurally invalid")
        row_id = _string(candidate["candidate_id"], "candidate row ID", 320)
        matches += row_id == candidate_id
    if matches != 1:
        _fail("selected candidate row is inconsistent")
    return {
        "provider": provider,
        "provider_video_id": provider_video_id,
        "selected_candidate_id": candidate_id,
        "video_url": selected.get("video_url"),
    }


def _validate_selections(
    payload: object, timeline_indexes: set[int]
) -> dict[int, dict]:
    if not isinstance(payload, dict):
        _fail("selection manifest must contain an object")
    if _strict_int(payload.get("version"), "selection manifest version") != 1:
        _fail("selection manifest version is unsupported")
    _validate_binding(payload.get("source_candidate_manifest"), "candidate binding")
    _validate_binding(payload.get("source_preview_manifest"), "preview binding")
    rows = payload.get("scenes")
    if not isinstance(rows, list):
        _fail("selection scenes must contain an array")
    by_index: dict[int, dict] = {}
    previous = 0
    for row in rows:
        if not isinstance(row, dict):
            _fail("selection scene is invalid")
        index = _strict_int(row.get("scene_index"), "selection scene_index")
        if index in by_index or index <= previous:
            _fail("selection indexes must be unique and in source order")
        if index not in timeline_indexes:
            _fail("selection scene does not exist in the timeline")
        status_value = row.get("status")
        status_name = _string(status_value, "selection status", 32)
        if status_name not in KNOWN_STATUSES:
            _fail("selection status is unsupported")
        reuse = row.get("reuse_scene_index")
        if reuse is not None:
            reuse = _strict_int(reuse, "reuse_scene_index")
            if reuse == index or reuse not in timeline_indexes:
                _fail("reuse_scene_index is invalid")
        if status_name in SELECTED_STATUSES:
            identity = _candidate_identity(row)
        else:
            identity = None
        by_index[index] = {"status": status_name, "reuse": reuse, "identity": identity}
        previous = index
    return by_index


def _usable_url(value) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if (
        len(value) > 4096
        or _CONTROL.search(value)
        or any(char.isspace() for char in value)
    ):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            return None
        # Accessing port also validates malformed or out-of-range port values.
        parsed.port
    except ValueError:
        return None
    return value


def _fallback(scene: dict, reason: SceneFallbackReason) -> dict:
    return {
        **scene,
        "binding": SceneRenderBinding.fallback_required.value,
        "visual_source_scene_index": None,
        "provider": None,
        "provider_video_id": None,
        "selected_candidate_id": None,
        "video_url": None,
        "fallback_reason": reason.value,
    }


def _selected(scene: dict, identity: dict) -> dict:
    video_url = _usable_url(identity["video_url"])
    if video_url is None:
        return _fallback(scene, SceneFallbackReason.unavailable_url)
    return {
        **scene,
        "binding": SceneRenderBinding.selected.value,
        "visual_source_scene_index": scene["scene_index"],
        "provider": identity["provider"],
        "provider_video_id": identity["provider_video_id"],
        "selected_candidate_id": identity["selected_candidate_id"],
        "video_url": video_url,
        "fallback_reason": None,
    }


def _build_rows(timeline: list[dict], selections: dict[int, dict]) -> list[dict]:
    results: dict[int, dict] = {}
    holds: list[dict] = []
    for scene in timeline:
        selection = selections.get(scene["scene_index"])
        if not scene["text"].strip():
            if selection is not None and selection["status"] != "hold_no_search":
                _fail("hold scene has a non-hold selection")
            holds.append(scene)
            continue
        if selection is None:
            results[scene["scene_index"]] = _fallback(
                scene, SceneFallbackReason.missing_selection
            )
        elif selection["status"] == "hold_no_search":
            _fail("meaningful scene has a hold selection")
        elif selection["status"] in SELECTED_STATUSES:
            results[scene["scene_index"]] = _selected(scene, selection["identity"])
        else:
            results[scene["scene_index"]] = _fallback(
                scene, SceneFallbackReason(selection["status"])
            )

    meaningful_indexes = [
        scene["scene_index"] for scene in timeline if scene["text"].strip()
    ]
    for scene in holds:
        index = scene["scene_index"]
        selection = selections.get(index)
        requested = selection["reuse"] if selection is not None else None
        ordered: list[int] = []
        if requested is not None:
            ordered.append(requested)
        ordered.extend(
            candidate for candidate in reversed(meaningful_indexes) if candidate < index
        )
        ordered.extend(
            candidate for candidate in meaningful_indexes if candidate > index
        )
        source = next(
            (
                results[candidate]
                for candidate in ordered
                if candidate in results
                and results[candidate]["binding"] == SceneRenderBinding.selected.value
            ),
            None,
        )
        if source is None:
            results[index] = _fallback(
                scene, SceneFallbackReason.reuse_target_unresolved
            )
        else:
            results[index] = {
                **scene,
                "binding": SceneRenderBinding.reused.value,
                "visual_source_scene_index": source["scene_index"],
                "provider": source["provider"],
                "provider_video_id": source["provider_video_id"],
                "selected_candidate_id": source["selected_candidate_id"],
                "video_url": source["video_url"],
                "fallback_reason": None,
            }
    return [results[scene["scene_index"]] for scene in timeline]


def _atomic_write(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            _fail("render plan target must be a regular file")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_scene_render_plan(
    scene_manifest_path: str, selection_manifest_path: str
) -> str:
    """Validate source artifacts and atomically publish ``scene_render_plan.json``."""
    scene_path, scene_bytes = _read_manifest(scene_manifest_path, "scene manifest")
    selection_path, selection_bytes = _read_manifest(
        selection_manifest_path, "selection manifest"
    )
    if scene_path.parent != selection_path.parent:
        _fail("source manifests must share a task directory")
    parent = scene_path.parent
    if parent.is_symlink() or not parent.is_dir():
        _fail("task directory is invalid")

    timeline = _validate_timeline(_parse_json(scene_bytes, "scene manifest"))
    selections = _validate_selections(
        _parse_json(selection_bytes, "selection manifest"),
        {scene["scene_index"] for scene in timeline},
    )
    rows = _build_rows(timeline, selections)
    if len(rows) != len(timeline):
        _fail("render plan scene coverage is incomplete")

    payload = {
        "version": MANIFEST_VERSION,
        "source_scene_manifest": {"sha256": hashlib.sha256(scene_bytes).hexdigest()},
        "source_selection_manifest": {
            "version": 1,
            "sha256": hashlib.sha256(selection_bytes).hexdigest(),
        },
        "scenes": rows,
    }
    target = parent / "scene_render_plan.json"
    _atomic_write(target, payload)
    return str(target)
