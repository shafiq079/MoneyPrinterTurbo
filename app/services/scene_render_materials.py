"""Acquire selected scene footage and publish a validated local material map."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import stat
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.services import material, scene_render_plan

MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_SELECTED_VIDEO_BYTES = 512 * 1024 * 1024
MAX_TOTAL_SELECTED_VIDEO_BYTES = 8 * 1024 * 1024 * 1024
MAX_UNIQUE_SELECTED_DOWNLOADS = 250
DOWNLOAD_CHUNK_BYTES = 64 * 1024
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
MAX_REDIRECTS = 5

ACQUISITION_ERRORS = {
    "download_failed",
    "download_too_large",
    "redirect_limit_exceeded",
    "unsafe_destination",
    "invalid_media",
    "cache_validation_failed",
    "source_download_failed",
}
RESOLUTIONS = {
    "selected",
    "reused",
    "fallback_previous_selected",
    "fallback_next_selected",
    "fallback_legacy",
}
ORIGINS = {"selected_url", "legacy_fallback"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AcquisitionLimitError(ValueError):
    """A task-wide acquisition bound was exceeded."""


class NoSelectedSceneCoverageError(ValueError):
    """Selected downloads cannot cover every scene without a legacy fallback."""


class AcquisitionError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _regular_bytes(path: Path, maximum: int, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if not 0 < before.st_size <= maximum:
        raise ValueError(f"{name} size is outside the supported range")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{name} changed while being opened")
        data = stream.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError(f"{name} size is outside the supported range")
    return data


def _real_root(root: str, name: str, *, create: bool = False) -> Path:
    candidate = Path(root).absolute()
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{name} must be a real directory")
    return candidate.resolve(strict=True)


def _contained_file(value: str, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} path is invalid")
    candidate = Path(value).absolute()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} is outside its trusted root") from exc
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes its trusted root") from exc
    return resolved


def _probe(path: Path, root: Path, *, selected: bool) -> dict:
    candidate = _contained_file(str(path), root, "video material")
    maximum = (
        MAX_SELECTED_VIDEO_BYTES if selected else MAX_TOTAL_SELECTED_VIDEO_BYTES
    )
    before = candidate.stat()
    if not 0 < before.st_size <= maximum:
        raise ValueError("video material size is outside the supported range")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("video material changed while being opened")
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    clip = None
    try:
        clip = VideoFileClip(str(candidate))
        duration = float(clip.duration)
        fps = float(clip.fps)
        width, height = (int(value) for value in clip.size)
        if not all(math.isfinite(value) and value > 0 for value in (duration, fps)):
            raise ValueError("video material has invalid timing")
        if width <= 0 or height <= 0:
            raise ValueError("video material has invalid dimensions")
    finally:
        if clip is not None:
            clip.close()
    after = candidate.lstat()
    if candidate.is_symlink() or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise ValueError("video material changed during validation")
    return {
        "local_path": str(candidate),
        "content_sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _validate_url(url: str) -> tuple[str, int]:
    if not isinstance(url, str) or not url or len(url) > 4096:
        raise AcquisitionError("unsafe_destination")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in url
    ):
        raise AcquisitionError("unsafe_destination")
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise AcquisitionError("unsafe_destination")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise AcquisitionError("unsafe_destination") from exc
    return parsed.hostname, port


def _validate_destination(url: str) -> None:
    hostname, port = _validate_url(url)
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise AcquisitionError("unsafe_destination") from exc
    if not addresses:
        raise AcquisitionError("unsafe_destination")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address[4][0])
        except ValueError as exc:
            raise AcquisitionError("unsafe_destination") from exc
        if not ip.is_global:
            raise AcquisitionError("unsafe_destination")


def _response_for(url: str):
    current = url
    visited = set()
    for redirect_count in range(MAX_REDIRECTS + 1):
        _validate_destination(current)
        if current in visited:
            raise AcquisitionError("redirect_limit_exceeded")
        visited.add(current)
        try:
            response = requests.get(
                current,
                headers={"User-Agent": "MoneyPrinterTurbo scene material acquisition"},
                proxies=config.proxy,
                verify=material._get_tls_verify(),
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise AcquisitionError("download_failed") from exc
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise AcquisitionError("download_failed")
            if redirect_count >= MAX_REDIRECTS:
                raise AcquisitionError("redirect_limit_exceeded")
            current = urljoin(current, location)
            _validate_url(current)
            continue
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            response.close()
            raise AcquisitionError("download_failed") from exc
        return response
    raise AcquisitionError("redirect_limit_exceeded")


def _download(url: str, destination: Path, root: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        try:
            return _probe(destination, root, selected=True)
        except Exception:
            if destination.is_symlink() or not destination.is_file():
                raise AcquisitionError("cache_validation_failed")
            destination.unlink(missing_ok=True)
    response = _response_for(url)
    temporary = None
    try:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise AcquisitionError("download_failed") from exc
            if declared_size > MAX_SELECTED_VIDEO_BYTES:
                raise AcquisitionLimitError("selected video exceeds size limit")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            total = 0
            try:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_SELECTED_VIDEO_BYTES:
                        raise AcquisitionLimitError(
                            "selected video exceeds size limit"
                        )
                    output.write(chunk)
            except requests.RequestException as exc:
                raise AcquisitionError("download_failed") from exc
            output.flush()
            os.fsync(output.fileno())
        if total <= 0:
            raise AcquisitionError("invalid_media")
        _probe(temporary, root, selected=True)
        os.replace(temporary, destination)
        temporary = None
        return _probe(destination, root, selected=True)
    finally:
        response.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _material_row(metadata: dict, **identity) -> dict:
    return {**identity, **metadata}


def _selected_destination(selected_root: Path, source_url_sha256: str) -> Path:
    """Return the sole cache path authorized for an exact selected URL hash."""
    return selected_root / f"selected-{source_url_sha256}.mp4"


def _atomic_write(path: Path, payload: dict, validate_temporary) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("material manifest target is unsafe")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        validate_temporary(str(temporary))
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_scene_render_materials(
    render_plan_path: str,
    scene_manifest_path: str,
    selection_manifest_path: str,
    legacy_material_paths: list[str],
    legacy_material_root: str,
    task_dir: str,
) -> str:
    """Acquire selected URLs and atomically publish a complete scene material map."""
    task_root = _real_root(task_dir, "task directory")
    plan = scene_render_plan.load_scene_render_plan(
        render_plan_path, scene_manifest_path, selection_manifest_path
    )
    plan_path = _contained_file(render_plan_path, task_root, "render plan")
    plan_bytes = _regular_bytes(plan_path, MAX_MANIFEST_BYTES, "render plan")
    selected_root = _real_root(
        str(task_root / "scene_materials"), "selected material directory", create=True
    )
    legacy_root = (
        _real_root(legacy_material_root, "legacy material root")
        if legacy_material_paths
        else None
    )

    selected_requests = {}
    selected_identities = {}
    for scene in plan["scenes"]:
        if scene["binding"] != "selected":
            continue
        url = scene["video_url"]
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        identity = (scene["provider"], scene["provider_video_id"])
        if selected_identities.setdefault(url_hash, identity) != identity:
            raise ValueError("one selected URL has inconsistent provider identities")
        selected_requests.setdefault(url_hash, scene)
    if len(selected_requests) > MAX_UNIQUE_SELECTED_DOWNLOADS:
        raise AcquisitionLimitError("too many unique selected videos")

    selected_materials = {}
    selected_errors = {}
    aggregate = 0
    for url_hash, scene in selected_requests.items():
        try:
            metadata = _download(
                scene["video_url"],
                _selected_destination(selected_root, url_hash),
                selected_root,
            )
            aggregate += metadata["size_bytes"]
            if aggregate > MAX_TOTAL_SELECTED_VIDEO_BYTES:
                raise AcquisitionLimitError("selected video aggregate exceeds limit")
            material_id = f"selected:{url_hash}"
            selected_materials[url_hash] = _material_row(
                metadata,
                material_id=material_id,
                origin="selected_url",
                source_url_sha256=url_hash,
                provider=scene["provider"],
                provider_video_id=scene["provider_video_id"],
            )
        except AcquisitionLimitError:
            raise
        except AcquisitionError as exc:
            selected_errors[url_hash] = exc.code
        except Exception:
            selected_errors[url_hash] = "invalid_media"

    legacy_rows = []
    seen_legacy_paths = set()
    for legacy_path in legacy_material_paths:
        try:
            candidate = _contained_file(legacy_path, legacy_root, "legacy material")
            if candidate in seen_legacy_paths:
                continue
            seen_legacy_paths.add(candidate)
            metadata = _probe(candidate, legacy_root, selected=False)
            legacy_rows.append(
                _material_row(
                    metadata,
                    material_id=f"legacy:{metadata['content_sha256']}",
                    origin="legacy_fallback",
                    source_url_sha256=None,
                    provider=None,
                    provider_video_id=None,
                )
            )
        except Exception:
            continue

    acquired_by_scene = {}
    for scene in plan["scenes"]:
        if scene["binding"] != "selected":
            continue
        url_hash = hashlib.sha256(scene["video_url"].encode("utf-8")).hexdigest()
        if url_hash in selected_materials:
            acquired_by_scene[scene["scene_index"]] = selected_materials[url_hash]

    def fallback(index: int):
        prior = [value for key, value in acquired_by_scene.items() if key < index]
        if prior:
            source_index = max(key for key in acquired_by_scene if key < index)
            return (
                "fallback_previous_selected",
                source_index,
                acquired_by_scene[source_index],
            )
        later = [key for key in acquired_by_scene if key > index]
        if later:
            source_index = min(later)
            return (
                "fallback_next_selected",
                source_index,
                acquired_by_scene[source_index],
            )
        if legacy_rows:
            return "fallback_legacy", None, legacy_rows[0]
        if not legacy_material_paths:
            raise NoSelectedSceneCoverageError(
                "no selected material can provide complete scene coverage and no "
                "legacy fallback was supplied"
            )
        raise ValueError("scene has no validated fallback material")

    scene_rows = []
    used_materials = {}
    plan_by_index = {row["scene_index"]: row for row in plan["scenes"]}
    for scene in plan["scenes"]:
        index = scene["scene_index"]
        requested = scene["binding"]
        requested_source = scene["visual_source_scene_index"]
        acquisition_error = None
        if requested == "selected" and index in acquired_by_scene:
            resolution, resolved_source, chosen = (
                "selected",
                index,
                acquired_by_scene[index],
            )
        elif requested == "reused" and requested_source in acquired_by_scene:
            resolution = "reused"
            resolved_source = requested_source
            chosen = acquired_by_scene[requested_source]
        else:
            resolution, resolved_source, chosen = fallback(index)
            if requested == "selected":
                url_hash = hashlib.sha256(
                    scene["video_url"].encode("utf-8")
                ).hexdigest()
                acquisition_error = selected_errors.get(url_hash, "download_failed")
            elif requested == "reused":
                source = plan_by_index.get(requested_source)
                if source and source.get("video_url"):
                    source_hash = hashlib.sha256(
                        source["video_url"].encode("utf-8")
                    ).hexdigest()
                    if source_hash in selected_errors:
                        acquisition_error = "source_download_failed"
        used_materials[chosen["material_id"]] = chosen
        scene_rows.append(
            {
                "scene_index": index,
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "duration": scene["duration"],
                "requested_binding": requested,
                "resolution": resolution,
                "requested_visual_source_scene_index": requested_source,
                "resolved_visual_source_scene_index": resolved_source,
                "material_id": chosen["material_id"],
                "fallback_reason": scene["fallback_reason"],
                "acquisition_error": acquisition_error,
            }
        )

    payload = {
        "version": MANIFEST_VERSION,
        "source_render_plan": {
            "version": plan["version"],
            "sha256": hashlib.sha256(plan_bytes).hexdigest(),
        },
        "source_scene_manifest": plan["source_scene_manifest"],
        "source_selection_manifest": plan["source_selection_manifest"],
        "materials": list(used_materials.values()),
        "scenes": scene_rows,
    }
    target = task_root / "scene_render_materials.json"
    validate_args = (
        render_plan_path,
        scene_manifest_path,
        selection_manifest_path,
        legacy_material_paths,
        legacy_material_root,
        task_dir,
    )
    _atomic_write(
        target,
        payload,
        lambda temporary_path: load_scene_render_materials(
            temporary_path, *validate_args
        ),
    )
    try:
        load_scene_render_materials(str(target), *validate_args)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return str(target)


def load_scene_render_materials(
    manifest_path: str,
    render_plan_path: str,
    scene_manifest_path: str,
    selection_manifest_path: str,
    legacy_material_paths: list[str],
    legacy_material_root: str,
    task_dir: str,
) -> dict:
    """Strictly validate a material artifact, its sources, and its exact bindings."""
    task_root = _real_root(task_dir, "task directory")
    manifest = _contained_file(manifest_path, task_root, "material manifest")
    data = _regular_bytes(manifest, MAX_MANIFEST_BYTES, "material manifest")

    def reject_constant(_value):
        raise ValueError("material manifest contains a non-finite number")

    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("material manifest is invalid JSON") from exc
    plan = scene_render_plan.load_scene_render_plan(
        render_plan_path, scene_manifest_path, selection_manifest_path
    )
    plan_bytes = _regular_bytes(
        _contained_file(render_plan_path, task_root, "render plan"),
        MAX_MANIFEST_BYTES,
        "render plan",
    )
    expected_fields = {
        "version", "source_render_plan", "source_scene_manifest",
        "source_selection_manifest", "materials", "scenes",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("material manifest fields are invalid")
    if payload["version"] != MANIFEST_VERSION or payload["source_render_plan"] != {
        "version": plan["version"],
        "sha256": hashlib.sha256(plan_bytes).hexdigest(),
    }:
        raise ValueError("material manifest source binding is invalid")
    if (
        payload["source_scene_manifest"] != plan["source_scene_manifest"]
        or payload["source_selection_manifest"] != plan["source_selection_manifest"]
    ):
        raise ValueError("material manifest source hashes are invalid")
    if not isinstance(payload["materials"], list) or not isinstance(
        payload["scenes"], list
    ):
        raise ValueError("material manifest arrays are invalid")

    selected_root = _real_root(
        str(task_root / "scene_materials"), "selected material directory"
    )
    legacy_root = (
        _real_root(legacy_material_root, "legacy material root")
        if legacy_material_paths
        else None
    )
    validated_legacy = []
    seen_legacy_paths = set()
    seen_legacy_ids = set()
    for legacy_path in legacy_material_paths:
        try:
            candidate = _contained_file(legacy_path, legacy_root, "legacy material")
            if candidate in seen_legacy_paths:
                continue
            seen_legacy_paths.add(candidate)
            metadata = _probe(candidate, legacy_root, selected=False)
            material_id = f"legacy:{metadata['content_sha256']}"
            if material_id in seen_legacy_ids:
                continue
            seen_legacy_ids.add(material_id)
            validated_legacy.append({**metadata, "material_id": material_id})
        except Exception:
            continue

    materials = {}
    selected_count = 0
    selected_total = 0
    material_fields = {
        "material_id", "origin", "source_url_sha256", "provider",
        "provider_video_id", "local_path", "content_sha256", "size_bytes",
        "duration", "fps", "width", "height",
    }
    for row in payload["materials"]:
        if (
            not isinstance(row, dict)
            or set(row) != material_fields
            or row["origin"] not in ORIGINS
        ):
            raise ValueError("material row is invalid")
        if row["origin"] == "legacy_fallback" and legacy_root is None:
            raise ValueError("legacy material was not supplied by the task")
        root = selected_root if row["origin"] == "selected_url" else legacy_root
        metadata = _probe(
            Path(row["local_path"]), root, selected=row["origin"] == "selected_url"
        )
        for key in (
            "local_path", "content_sha256", "size_bytes", "duration", "fps",
            "width", "height",
        ):
            if metadata[key] != row[key]:
                raise ValueError("material file does not match its manifest")
        if row["origin"] == "selected_url":
            digest = row["source_url_sha256"]
            if (
                not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                or row["material_id"] != f"selected:{digest}"
                or not isinstance(row["provider"], str)
                or not row["provider"]
                or not isinstance(row["provider_video_id"], str)
                or not row["provider_video_id"]
            ):
                raise ValueError("selected material identity is invalid")
            try:
                expected_path = _selected_destination(selected_root, digest).resolve(
                    strict=True
                )
            except OSError as exc:
                raise ValueError("selected material cache path is unavailable") from exc
            if metadata["local_path"] != str(expected_path):
                raise ValueError("selected material path does not match its URL hash")
            selected_count += 1
            selected_total += row["size_bytes"]
        else:
            if (
                row["source_url_sha256"] is not None
                or row["provider"] is not None
                or row["provider_video_id"] is not None
                or row["material_id"] != f"legacy:{row['content_sha256']}"
            ):
                raise ValueError("legacy material identity is invalid")
            if not any(
                item["material_id"] == row["material_id"]
                and item["local_path"] == row["local_path"]
                for item in validated_legacy
            ):
                raise ValueError("legacy material was not supplied by the task")
        if row["material_id"] in materials:
            raise ValueError("material IDs must be unique")
        materials[row["material_id"]] = row
    if (
        selected_count > MAX_UNIQUE_SELECTED_DOWNLOADS
        or selected_total > MAX_TOTAL_SELECTED_VIDEO_BYTES
    ):
        raise ValueError("selected material limits are exceeded")

    plan_by_index = {row["scene_index"]: row for row in plan["scenes"]}
    available_selected = {}
    selected_identity_by_hash = {}
    for requested in plan["scenes"]:
        if requested["binding"] != "selected":
            continue
        digest = hashlib.sha256(requested["video_url"].encode("utf-8")).hexdigest()
        identity = (requested["provider"], requested["provider_video_id"])
        if selected_identity_by_hash.setdefault(digest, identity) != identity:
            raise ValueError("one selected URL has inconsistent provider identities")
        material_id = f"selected:{digest}"
        candidate = materials.get(material_id)
        if candidate is None:
            continue
        if (
            candidate["source_url_sha256"] != digest
            or candidate["provider"] != requested["provider"]
            or candidate["provider_video_id"] != requested["provider_video_id"]
        ):
            raise ValueError("selected material does not match its render-plan source")
        available_selected[requested["scene_index"]] = candidate

    first_legacy = validated_legacy[0] if validated_legacy else None

    def expected_fallback(index):
        prior = [source for source in available_selected if source < index]
        if prior:
            source = max(prior)
            return "fallback_previous_selected", source, available_selected[source]
        later = [source for source in available_selected if source > index]
        if later:
            source = min(later)
            return "fallback_next_selected", source, available_selected[source]
        if first_legacy is not None:
            legacy_row = materials.get(first_legacy["material_id"])
            if legacy_row is None:
                raise ValueError("first supplied legacy fallback material is missing")
            return "fallback_legacy", None, legacy_row
        raise ValueError("scene has no validated fallback material")

    scene_fields = {
        "scene_index", "start_time", "end_time", "duration",
        "requested_binding", "resolution", "requested_visual_source_scene_index",
        "resolved_visual_source_scene_index", "material_id", "fallback_reason",
        "acquisition_error",
    }
    if len(payload["scenes"]) != len(plan["scenes"]):
        raise ValueError("material scene coverage is incomplete")
    used_material_ids = set()
    for actual, requested in zip(payload["scenes"], plan["scenes"]):
        if not isinstance(actual, dict) or set(actual) != scene_fields:
            raise ValueError("material scene row is invalid")
        if actual["scene_index"] != requested["scene_index"] or any(
            actual[key] != requested[key]
            for key in ("start_time", "end_time", "duration")
        ):
            raise ValueError("material scene timing is invalid")
        if (
            actual["requested_binding"] != requested["binding"]
            or actual["requested_visual_source_scene_index"]
            != requested["visual_source_scene_index"]
            or actual["fallback_reason"] != requested["fallback_reason"]
        ):
            raise ValueError("material scene request is invalid")
        if actual["material_id"] not in materials:
            raise ValueError("material scene references a missing material")
        error = actual["acquisition_error"]
        if error is not None and error not in ACQUISITION_ERRORS:
            raise ValueError("material acquisition error is invalid")

        binding = requested["binding"]
        index = requested["scene_index"]
        if binding == "selected" and index in available_selected:
            expected = ("selected", index, available_selected[index])
            expected_error = None
        elif binding == "reused":
            source_index = requested["visual_source_scene_index"]
            source = plan_by_index.get(source_index)
            if source is None or source["binding"] != "selected":
                raise ValueError("reused scene source is not selected")
            if source_index in available_selected:
                expected = ("reused", source_index, available_selected[source_index])
                expected_error = None
            else:
                expected = expected_fallback(index)
                expected_error = "source_download_failed"
        else:
            expected = expected_fallback(index)
            expected_error = None if binding == "fallback_required" else "failure"

        expected_resolution, expected_source, expected_material = expected
        if (
            actual["resolution"] != expected_resolution
            or actual["resolved_visual_source_scene_index"] != expected_source
            or actual["material_id"] != expected_material["material_id"]
        ):
            raise ValueError("material scene does not use its deterministic resolution")
        if expected_error == "failure":
            if error is None or error == "source_download_failed":
                raise ValueError("selected fallback acquisition error is invalid")
        elif error != expected_error:
            raise ValueError("material acquisition error semantics are invalid")
        used_material_ids.add(actual["material_id"])
    if used_material_ids != set(materials):
        raise ValueError("material manifest contains unused material rows")
    return payload
