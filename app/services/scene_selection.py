"""Validate scene artifacts and publish deterministic provider-rank selections."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.services import scene_preview_cache

MANIFEST_VERSION = 1
SELECTION_POLICY_VERSION = "provider-rank-v1"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_NORMALIZED_PREVIEW_OBJECT_BYTES = 4 * 1024 * 1024
MAX_CANDIDATES_PER_SCENE = 12
MAX_PROVIDER_SEARCH_BUDGET = 60
MAX_WARNINGS = 8
VALID_ASPECTS = {"16:9", "9:16", "1:1"}
SUPPORTED_PROVIDERS = frozenset(scene_preview_cache.ALLOWED_HOSTS)
RETRIEVAL_STATUSES = {
    "complete",
    "no_results_or_error",
    "budget_exhausted",
    "partial_budget_exhausted",
    "hold_no_search",
}
QUERY_SOURCES = {"llm", "fallback", "none"}
PREVIEW_STATUSES = {
    "ready",
    "budget_exhausted",
    "missing_url",
    "invalid_url",
    "http_rejected",
    "unsupported_format",
    "size_limit_exceeded",
    "invalid_image",
    "download_failed",
    "storage_failed",
}
PREVIEW_SCENE_STATUSES = {
    "complete",
    "partial",
    "budget_exhausted",
    "no_previews",
    "hold_no_search",
}
WARNING_CODES = {
    "source_query_generation_warning",
    "source_provider_search_warning",
    "source_provider_budget_warning",
    "candidate_preview_not_ready",
    "candidate_preview_object_missing",
    "candidate_preview_object_invalid",
    "selected_preview_not_ready",
    "selected_preview_object_missing",
    "selected_preview_object_invalid",
    "incomplete_preview_coverage",
    "ranking_derivative_too_large",
    "ranking_cache_corrupt",
    "vlm_request_failed",
    "vlm_response_invalid",
    "additional_warnings_omitted",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SceneSelectionStatus(str, Enum):
    provider_rank_selected = "provider_rank_selected"
    provider_rank_fallback = "provider_rank_fallback"
    vlm_selected = "vlm_selected"
    no_safe_candidate = "no_safe_candidate"
    hold_no_search = "hold_no_search"
    no_candidates = "no_candidates"


class SelectionPreviewStatus(str, Enum):
    valid = "valid"
    not_ready = "not_ready"
    object_missing = "object_missing"
    object_invalid = "object_invalid"


def _fail(message: str) -> None:
    raise ValueError(message)


def _strict_int(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        _fail(f"{name} is outside the supported range")
    return value


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        _fail(f"{name} is invalid")
    return result


def _string(
    value,
    name: str,
    *,
    minimum: int = 0,
    maximum: int,
    trimmed: bool = False,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        _fail(f"{name} is invalid")
    if _CONTROL.search(value) or (trimmed and value != value.strip()):
        _fail(f"{name} is invalid")
    return value


def _optional_string(value, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _string(value, name, minimum=1, maximum=maximum, trimmed=True)


def _object(value, name: str, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{name} has invalid fields")
    return value


def _array(value, name: str) -> list:
    if not isinstance(value, list):
        _fail(f"{name} must be an array")
    return value


def _read_manifest(path: str, name: str) -> bytes:
    source = Path(path)
    try:
        stat = source.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if source.is_symlink() or not source.is_file():
        _fail(f"{name} must be a regular file")
    if stat.st_size <= 0 or stat.st_size > MAX_MANIFEST_BYTES:
        _fail(f"{name} size is outside the supported range")
    try:
        with source.open("rb") as stream:
            data = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if not data or len(data) > MAX_MANIFEST_BYTES:
        _fail(f"{name} size is outside the supported range")
    return data


def _parse_json(data: bytes, name: str) -> dict:
    try:
        text = data.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{name} must contain a JSON object")
    return payload


def _validate_candidate(candidate: dict, group: dict, provider: str) -> None:
    _object(
        candidate,
        "candidate",
        {
            "candidate_id",
            "provider",
            "provider_video_id",
            "provider_page_url",
            "matched_query",
            "provider_rank",
            "preview_url",
            "video_url",
            "duration",
            "width",
            "height",
        },
    )
    item_provider = _string(candidate["provider"], "candidate provider", minimum=1, maximum=32)
    if item_provider != provider:
        _fail("candidate provider differs from manifest provider")
    video_id = _string(candidate["provider_video_id"], "provider_video_id", minimum=1, maximum=256)
    candidate_id = _string(candidate["candidate_id"], "candidate_id", minimum=3, maximum=320)
    if candidate_id != f"{provider}:{video_id}":
        _fail("candidate_id is inconsistent with provider identity")
    matched = _string(candidate["matched_query"], "matched_query", minimum=1, maximum=80, trimmed=True)
    if matched not in group["queries"]:
        _fail("candidate matched_query is not a source query")
    _strict_int(candidate["provider_rank"], "provider_rank", 0)
    _optional_string(candidate["provider_page_url"], "provider_page_url", 4096)
    _optional_string(candidate["preview_url"], "preview_url", 4096)
    _string(candidate["video_url"], "video_url", minimum=1, maximum=4096, trimmed=True)
    _number(candidate["duration"], "candidate duration", positive=True)
    _strict_int(candidate["width"], "candidate width", 1, 100_000)
    _strict_int(candidate["height"], "candidate height", 1, 100_000)


def _validate_candidate_manifest(payload: dict) -> dict:
    _object(
        payload,
        "candidate manifest",
        {
            "version",
            "provider",
            "video_aspect",
            "candidates_per_scene",
            "provider_search_budget",
            "remote_searches_used",
            "query_generation_warning",
            "scenes",
        },
    )
    if _strict_int(payload["version"], "candidate manifest version", 1, 1) != 1:
        _fail("unsupported candidate manifest version")
    provider = _string(payload["provider"], "provider", minimum=1, maximum=32)
    if provider not in SUPPORTED_PROVIDERS:
        _fail("unsupported candidate provider")
    aspect = _string(payload["video_aspect"], "video_aspect", minimum=3, maximum=4)
    if aspect not in VALID_ASPECTS:
        _fail("unsupported video_aspect")
    limit = _strict_int(payload["candidates_per_scene"], "candidates_per_scene", 1, MAX_CANDIDATES_PER_SCENE)
    budget = _strict_int(payload["provider_search_budget"], "provider_search_budget", 0, MAX_PROVIDER_SEARCH_BUDGET)
    used = _strict_int(payload["remote_searches_used"], "remote_searches_used", 0, budget)
    del used
    _optional_string(payload["query_generation_warning"], "query_generation_warning", 500)
    scenes = _array(payload["scenes"], "candidate scenes")
    seen_indexes: set[int] = set()
    previous = 0
    for group in scenes:
        _object(
            group,
            "candidate scene",
            {
                "scene_index",
                "start_time",
                "end_time",
                "duration",
                "text",
                "queries",
                "query_source",
                "status",
                "warning",
                "reuse_scene_index",
                "candidates",
            },
        )
        index = _strict_int(group["scene_index"], "scene_index", 1)
        if index in seen_indexes or index <= previous:
            _fail("scene indexes must be unique and in source order")
        seen_indexes.add(index)
        previous = index
        start = _number(group["start_time"], "start_time")
        if start < 0:
            _fail("start_time is invalid")
        end = _number(group["end_time"], "end_time", positive=True)
        duration = _number(group["duration"], "duration", positive=True)
        if end <= start or abs(duration - (end - start)) > 0.001:
            _fail("scene timing is inconsistent")
        text = _string(group["text"], "scene text", maximum=20_000)
        queries = _array(group["queries"], "queries")
        if len(queries) > 3:
            _fail("scene query count exceeds the supported range")
        for query in queries:
            _string(query, "query", minimum=1, maximum=80, trimmed=True)
        if len(set(queries)) != len(queries):
            _fail("scene queries must be unique")
        query_source = _string(group["query_source"], "query_source", minimum=3, maximum=8)
        status = _string(group["status"], "retrieval status", minimum=8, maximum=32)
        if query_source not in QUERY_SOURCES or status not in RETRIEVAL_STATUSES:
            _fail("scene retrieval status is invalid")
        warning = _optional_string(group["warning"], "scene warning", 500)
        candidates = _array(group["candidates"], "candidates")
        if len(candidates) > limit:
            _fail("source candidate count exceeds its declared limit")
        if status == "hold_no_search":
            if text.strip() or queries or query_source != "none" or candidates or warning is not None:
                _fail("hold_no_search scene has searchable material")
            if group["reuse_scene_index"] is not None:
                _strict_int(group["reuse_scene_index"], "reuse_scene_index", 1)
        else:
            if not text.strip() or query_source not in {"llm", "fallback"} or not queries:
                _fail("non-hold scene has invalid query metadata")
            if group["reuse_scene_index"] is not None:
                _fail("non-hold scene cannot reuse another scene")
            if status == "complete" and (not candidates or warning is not None):
                _fail("complete scene is inconsistent")
            if status in {"no_results_or_error", "budget_exhausted"} and candidates:
                _fail("empty retrieval status contains candidates")
            if status in {"no_results_or_error", "budget_exhausted", "partial_budget_exhausted"} and warning is None:
                _fail("incomplete retrieval status requires a warning")
            if status in {"budget_exhausted", "partial_budget_exhausted"} and not warning.startswith(
                "Provider search budget exhausted;"
            ):
                _fail("budget retrieval status has an invalid warning")
        seen_candidates: set[str] = set()
        for candidate in candidates:
            _validate_candidate(candidate, group, provider)
            if candidate["candidate_id"] in seen_candidates:
                _fail("candidate IDs must be unique within a scene")
            seen_candidates.add(candidate["candidate_id"])
    scene_by_index = {scene["scene_index"]: scene for scene in scenes}
    for group in scenes:
        reuse = group["reuse_scene_index"]
        if reuse is None:
            continue
        referenced = scene_by_index.get(reuse)
        if referenced is None or referenced is group or referenced["status"] == "hold_no_search" or not referenced["text"].strip():
            _fail("hold reuse_scene_index does not reference a meaningful scene")
    return payload


def _validate_limits_usage(preview: dict) -> None:
    limits = _object(
        preview["limits"],
        "preview limits",
        {
            "candidates_per_scene",
            "max_remote_downloads_per_task",
            "max_download_bytes_per_image",
            "max_aggregate_download_bytes",
            "max_decoded_pixels",
            "max_normalized_edge",
            "download_concurrency",
            "max_redirects",
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "total_deadline_seconds",
        },
    )
    ranges = {
        "candidates_per_scene": (1, MAX_CANDIDATES_PER_SCENE),
        "max_remote_downloads_per_task": (0, 240),
        "max_download_bytes_per_image": (1, 3 * 1024 * 1024),
        "max_aggregate_download_bytes": (0, 256 * 1024 * 1024),
        "max_decoded_pixels": (1, scene_preview_cache.MAX_DECODED_PIXELS),
        "max_normalized_edge": (1, scene_preview_cache.MAX_NORMALIZED_EDGE),
        "download_concurrency": (1, 8),
        "max_redirects": (0, scene_preview_cache.MAX_REDIRECTS),
        "connect_timeout_seconds": (1, scene_preview_cache.CONNECT_TIMEOUT_SECONDS),
        "read_timeout_seconds": (1, scene_preview_cache.READ_TIMEOUT_SECONDS),
        "total_deadline_seconds": (1, scene_preview_cache.TOTAL_DEADLINE_SECONDS),
    }
    for key, value in limits.items():
        minimum, maximum = ranges[key]
        _strict_int(value, f"preview limit {key}", minimum, maximum)
    usage = _object(
        preview["usage"],
        "preview usage",
        {"remote_downloads_started", "remote_download_bytes", "cache_hits", "in_task_reuses"},
    )
    for key, value in usage.items():
        _strict_int(value, f"preview usage {key}", 0)


def _expected_preview_scene_status(source: dict, results: list[dict]) -> str:
    if source["status"] == "hold_no_search":
        return "hold_no_search"
    if not results:
        return "no_previews"
    ready = sum(result["status"] == "ready" for result in results)
    if ready == len(results):
        return "complete"
    if ready:
        return "partial"
    if all(result["status"] == "budget_exhausted" for result in results):
        return "budget_exhausted"
    return "no_previews"


def _validate_preview_manifest(preview: dict, source: dict, source_digest: str) -> dict:
    _object(
        preview,
        "preview manifest",
        {"version", "source_candidate_manifest", "normalization_version", "provider", "video_aspect", "limits", "usage", "scenes"},
    )
    _strict_int(preview["version"], "preview manifest version", 1, 1)
    binding = _object(preview["source_candidate_manifest"], "source candidate binding", {"version", "sha256"})
    _strict_int(binding["version"], "source candidate version", 1, 1)
    digest = _string(binding["sha256"], "source candidate sha256", minimum=64, maximum=64)
    if not _SHA256.fullmatch(digest) or digest != source_digest:
        _fail("preview manifest source binding is invalid")
    normalization = _string(preview["normalization_version"], "normalization_version", minimum=1, maximum=64)
    if normalization != scene_preview_cache.NORMALIZATION_VERSION:
        _fail("unsupported preview normalization version")
    if preview["provider"] != source["provider"] or preview["video_aspect"] != source["video_aspect"]:
        _fail("preview provider or aspect differs from candidate manifest")
    _string(preview["provider"], "preview provider", minimum=1, maximum=32)
    if preview["video_aspect"] not in VALID_ASPECTS:
        _fail("unsupported preview video_aspect")
    _validate_limits_usage(preview)
    preview_scenes = _array(preview["scenes"], "preview scenes")
    if len(preview_scenes) != len(source["scenes"]):
        _fail("preview scene coverage is incomplete")
    for source_scene, preview_scene in zip(source["scenes"], preview_scenes, strict=True):
        _object(preview_scene, "preview scene", {"scene_index", "status", "reuse_scene_index", "candidates"})
        if type(preview_scene["scene_index"]) is not int or preview_scene["scene_index"] != source_scene["scene_index"]:
            _fail("preview scene order differs from candidate manifest")
        status = _string(preview_scene["status"], "preview scene status", minimum=7, maximum=32)
        if status not in PREVIEW_SCENE_STATUSES:
            _fail("preview scene status is invalid")
        if preview_scene["reuse_scene_index"] != source_scene["reuse_scene_index"] or (
            preview_scene["reuse_scene_index"] is not None and type(preview_scene["reuse_scene_index"]) is not int
        ):
            _fail("preview hold reuse differs from candidate manifest")
        results = _array(preview_scene["candidates"], "preview candidates")
        if len(results) != len(source_scene["candidates"]):
            _fail("preview candidate coverage is incomplete")
        for candidate, result in zip(source_scene["candidates"], results, strict=True):
            _object(result, "candidate preview", {"candidate_id", "status", "failure_reason", "preview"})
            if result["candidate_id"] != candidate["candidate_id"]:
                _fail("preview candidate order differs from candidate manifest")
            _string(result["candidate_id"], "preview candidate_id", minimum=3, maximum=320)
            result_status = _string(result["status"], "candidate preview status", minimum=5, maximum=32)
            if result_status not in PREVIEW_STATUSES:
                _fail("candidate preview status is invalid")
            if result_status == "ready":
                if result["failure_reason"] is not None or not isinstance(result["preview"], dict):
                    _fail("ready candidate preview is incomplete")
                _validate_preview_metadata(result["preview"], normalization)
            else:
                _optional_string(result["failure_reason"], "preview failure_reason", 120)
                if result["failure_reason"] is None or result["preview"] is not None:
                    _fail("failed candidate preview is inconsistent")
        if status != _expected_preview_scene_status(source_scene, results):
            _fail("preview scene status is inconsistent with candidate results")
    return preview


def _validate_preview_metadata(preview: dict, normalization: str) -> None:
    _object(preview, "preview metadata", {"normalization_version", "image_sha256", "media_type", "width", "height", "byte_size", "cache_reference"})
    if preview["normalization_version"] != normalization:
        _fail("candidate preview normalization version differs from manifest")
    digest = _string(preview["image_sha256"], "image_sha256", minimum=64, maximum=64)
    if not _SHA256.fullmatch(digest) or preview["media_type"] != "image/jpeg":
        _fail("candidate preview media metadata is invalid")
    width = _strict_int(preview["width"], "preview width", 1, scene_preview_cache.MAX_NORMALIZED_EDGE)
    height = _strict_int(preview["height"], "preview height", 1, scene_preview_cache.MAX_NORMALIZED_EDGE)
    if width * height > scene_preview_cache.MAX_DECODED_PIXELS:
        _fail("candidate preview pixel count is invalid")
    _strict_int(preview["byte_size"], "preview byte_size", 1, MAX_NORMALIZED_PREVIEW_OBJECT_BYTES)
    reference = _string(preview["cache_reference"], "cache_reference", minimum=1, maximum=256)
    if reference != scene_preview_cache.cache_reference(digest):
        _fail("candidate preview cache reference is not canonical")


def _validate_preview_object(metadata: dict) -> str:
    digest = metadata["image_sha256"]
    try:
        path = scene_preview_cache.object_dir() / scene_preview_cache.object_name(
            digest
        )
        stat = path.lstat()
        if path.is_symlink() or not path.is_file():
            return "object_invalid"
        if stat.st_size != metadata["byte_size"] or stat.st_size > MAX_NORMALIZED_PREVIEW_OBJECT_BYTES:
            return "object_invalid"
        with path.open("rb") as stream:
            data = stream.read(MAX_NORMALIZED_PREVIEW_OBJECT_BYTES + 1)
        if len(data) != metadata["byte_size"] or hashlib.sha256(data).hexdigest() != digest:
            return "object_invalid"
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
                return "object_invalid"
            image.load()
            if image.mode != "RGB" or image.size != (metadata["width"], metadata["height"]):
                return "object_invalid"
            if image.width * image.height > scene_preview_cache.MAX_DECODED_PIXELS or max(image.size) > scene_preview_cache.MAX_NORMALIZED_EDGE:
                return "object_invalid"
        return "valid"
    except FileNotFoundError:
        return "object_missing"
    except (
        OSError,
        ValueError,
        TypeError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ):
        return "object_invalid"


def _warnings(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in WARNING_CODES or value in unique:
            continue
        if len(unique) == MAX_WARNINGS - 1:
            unique.append("additional_warnings_omitted")
            break
        unique.append(value)
    return unique


def _atomic_write(path: Path, payload: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_scene_selections(candidate_manifest_path: str, preview_manifest_path: str) -> str:
    """Validate upstream artifacts and atomically publish deterministic selections."""
    candidate_bytes = _read_manifest(candidate_manifest_path, "candidate manifest")
    preview_bytes = _read_manifest(preview_manifest_path, "preview manifest")
    candidate_manifest = _validate_candidate_manifest(_parse_json(candidate_bytes, "candidate manifest"))
    candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
    preview_manifest = _validate_preview_manifest(
        _parse_json(preview_bytes, "preview manifest"), candidate_manifest, candidate_digest
    )
    preview_digest = hashlib.sha256(preview_bytes).hexdigest()

    scenes = []
    valid_objects = 0
    unavailable_objects = 0
    selected_count = hold_count = empty_count = 0
    manifest_warnings = []
    if candidate_manifest["query_generation_warning"] is not None:
        manifest_warnings.append("source_query_generation_warning")

    for source_scene, preview_scene in zip(candidate_manifest["scenes"], preview_manifest["scenes"], strict=True):
        candidate_rows = []
        scene_warning_values = []
        preview_states = []
        for position, (candidate, result) in enumerate(zip(source_scene["candidates"], preview_scene["candidates"], strict=True)):
            if result["status"] != "ready":
                preview_state = "not_ready"
                preview_sha = None
                unavailable_objects += 1
                scene_warning_values.append("candidate_preview_not_ready")
            else:
                preview_state = _validate_preview_object(result["preview"])
                preview_sha = result["preview"]["image_sha256"] if preview_state == "valid" else None
                if preview_state == "valid":
                    valid_objects += 1
                else:
                    unavailable_objects += 1
                    scene_warning_values.append(f"candidate_preview_{preview_state}")
            preview_states.append((preview_state, preview_sha))
            candidate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "provider_rank": candidate["provider_rank"],
                    "manifest_position": position,
                    "local_order": None,
                    "preview_status": preview_state,
                    "preview_sha256": preview_sha,
                    "assessment": None,
                    "score_basis_points": None,
                    "safety_excluded": None,
                }
            )
        selected_candidate = None
        selected_id = None
        selected_sha = None
        if source_scene["status"] == "hold_no_search":
            status = "hold_no_search"
            hold_count += 1
        elif not source_scene["candidates"]:
            status = "no_candidates"
            empty_count += 1
        else:
            status = "provider_rank_selected"
            selected_count += 1
            order = sorted(
                range(len(source_scene["candidates"])),
                key=lambda position: (
                    source_scene["candidates"][position]["provider_rank"],
                    position,
                    source_scene["candidates"][position]["candidate_id"],
                ),
            )
            for local_order, position in enumerate(order, 1):
                candidate_rows[position]["local_order"] = local_order
            position = order[0]
            selected = source_scene["candidates"][position]
            selected_id = selected["candidate_id"]
            selected_sha = preview_states[position][1]
            selected_candidate = {
                key: selected[key]
                for key in (
                    "candidate_id",
                    "provider",
                    "provider_video_id",
                    "provider_page_url",
                    "video_url",
                    "provider_rank",
                )
            }
            if preview_states[position][0] != "valid":
                scene_warning_values.append(f"selected_preview_{preview_states[position][0]}")
        if source_scene["warning"] is not None:
            scene_warning_values.append(
                "source_provider_budget_warning"
                if source_scene["status"] in {"budget_exhausted", "partial_budget_exhausted"}
                else "source_provider_search_warning"
            )
        scenes.append(
            {
                "scene_index": source_scene["scene_index"],
                "status": status,
                "reuse_scene_index": source_scene["reuse_scene_index"],
                "visual_safety_evaluated": False,
                "selected_candidate_id": selected_id,
                "selected_candidate": selected_candidate,
                "selected_preview_sha256": selected_sha,
                "fallback_reason": None,
                "candidates": candidate_rows,
                "warnings": _warnings(scene_warning_values),
            }
        )

    payload = {
        "version": MANIFEST_VERSION,
        "source_candidate_manifest": {"version": 1, "sha256": candidate_digest},
        "source_preview_manifest": {"version": 1, "sha256": preview_digest},
        "provider": candidate_manifest["provider"],
        "video_aspect": candidate_manifest["video_aspect"],
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "ranking": {
            "provider": None,
            "model": None,
            "prompt_version": None,
            "response_schema_version": None,
            "scoring_policy_version": None,
            "representation_version": None,
        },
        "usage": {
            "scene_count": len(scenes),
            "meaningful_scene_count": len(scenes) - hold_count,
            "provider_rank_selected_scenes": selected_count,
            "provider_rank_fallback_scenes": 0,
            "vlm_selected_scenes": 0,
            "no_safe_candidate_scenes": 0,
            "hold_scenes": hold_count,
            "no_candidate_scenes": empty_count,
            "valid_preview_objects": valid_objects,
            "unavailable_preview_objects": unavailable_objects,
            "vlm_requests_started": None,
            "vlm_attempts_started": None,
            "ranking_cache_hits": None,
            "ranking_cache_misses": None,
        },
        "scenes": scenes,
        "warnings": _warnings(manifest_warnings),
    }
    target = Path(candidate_manifest_path).parent / "scene_selections.json"
    _atomic_write(target, payload)
    return str(target)
