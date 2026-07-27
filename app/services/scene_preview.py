"""Prepare bounded, provider-poster previews for scene candidates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from app.services import scene_preview_cache
from app.services.scene_candidate import SceneCandidateManifest, SceneRetrievalStatus

MANIFEST_VERSION = 1
DEFAULT_CANDIDATES_PER_SCENE = 6
MAX_CANDIDATES_PER_SCENE = 12
DEFAULT_MAX_REMOTE_DOWNLOADS = 120
MAX_REMOTE_DOWNLOADS = 240
DEFAULT_MAX_AGGREGATE_BYTES = 128 * 1024 * 1024
MAX_AGGREGATE_BYTES = 256 * 1024 * 1024
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 8


class PreviewFile(BaseModel):
    normalization_version: str
    image_sha256: str
    media_type: str
    width: int
    height: int
    byte_size: int
    cache_reference: str


class CandidatePreviewResult(BaseModel):
    candidate_id: str
    status: str
    failure_reason: str | None = None
    preview: PreviewFile | None = None


class ScenePreviewResult(BaseModel):
    scene_index: int
    status: str
    reuse_scene_index: int | None = None
    candidates: list[CandidatePreviewResult]


class ScenePreviewManifest(BaseModel):
    version: int = MANIFEST_VERSION
    source_candidate_manifest: dict
    normalization_version: str
    provider: str
    video_aspect: str
    limits: dict
    usage: dict
    scenes: list[ScenePreviewResult]


def centered_stratified_indices(population: int, selections: int) -> list[int]:
    if population <= 0 or selections <= 0:
        return []
    if selections >= population:
        return list(range(population))
    values = {
        min(population - 1, max(0, int((index + 0.5) * population / selections)))
        for index in range(selections)
    }
    # The midpoint formula is unique for N <= M; the fill is defensive against
    # future arithmetic changes while preserving centered, source-ordered output.
    if len(values) < selections:
        for index in range(population):
            if index not in values:
                values.add(index)
                if len(values) == selections:
                    break
    return sorted(values)


def _failure(candidate_id: str, status: str, reason: str) -> CandidatePreviewResult:
    return CandidatePreviewResult(
        candidate_id=candidate_id, status=status, failure_reason=reason
    )


def _scene_status(group, results: list[CandidatePreviewResult]) -> str:
    if group.status == SceneRetrievalStatus.hold_no_search:
        return "hold_no_search"
    ready = sum(item.status == "ready" for item in results)
    if ready == len(results) and results:
        return "complete"
    if ready:
        return "partial"
    if results and all(item.status == "budget_exhausted" for item in results):
        return "budget_exhausted"
    return "no_previews"


def _validate_source_manifest(source: SceneCandidateManifest) -> None:
    if not 1 <= source.candidates_per_scene <= MAX_CANDIDATES_PER_SCENE:
        raise ValueError("source candidates_per_scene is outside the supported range")
    previous_index = None
    seen_indexes = set()
    for group in source.scenes:
        if group.scene_index in seen_indexes or (
            previous_index is not None and group.scene_index <= previous_index
        ):
            raise ValueError("scene indexes must be unique and in source order")
        seen_indexes.add(group.scene_index)
        previous_index = group.scene_index
        is_hold = group.status == SceneRetrievalStatus.hold_no_search
        if is_hold and (
            group.candidates or group.queries or group.query_source != "none"
        ):
            raise ValueError("hold_no_search scene has searchable material")
        if (
            len(group.candidates) > source.candidates_per_scene
            or len(group.candidates) > MAX_CANDIDATES_PER_SCENE
        ):
            raise ValueError("source candidate count exceeds its declared limit")
        for candidate in group.candidates:
            if candidate.provider != source.provider:
                raise ValueError("candidate provider differs from manifest provider")
            expected_id = f"{candidate.provider}:{candidate.provider_video_id}"
            if candidate.candidate_id != expected_id:
                raise ValueError("candidate_id is inconsistent with provider identity")


def _complete_scene_results(source, results) -> list[ScenePreviewResult]:
    if len(results) != len(source.scenes):
        raise ValueError("preview result scene count is incomplete")
    scene_results = []
    for group, group_results in zip(source.scenes, results, strict=True):
        if len(group_results) != len(group.candidates) or any(
            item is None for item in group_results
        ):
            raise ValueError("preview candidate result matrix is incomplete")
        finalized = list(group_results)
        if [item.candidate_id for item in finalized] != [
            candidate.candidate_id for candidate in group.candidates
        ]:
            raise ValueError("preview candidate result order is inconsistent")
        scene_results.append(
            ScenePreviewResult(
                scene_index=group.scene_index,
                status=_scene_status(group, finalized),
                reuse_scene_index=group.reuse_scene_index,
                candidates=finalized,
            )
        )
    return scene_results


def _atomic_manifest(path: Path, manifest: ScenePreviewManifest) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(
                manifest.model_dump(mode="json"), output, ensure_ascii=False, indent=2
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_scene_previews(
    candidate_manifest_path: str,
    *,
    candidates_per_scene: int = DEFAULT_CANDIDATES_PER_SCENE,
    max_remote_downloads: int = DEFAULT_MAX_REMOTE_DOWNLOADS,
    max_aggregate_bytes: int = DEFAULT_MAX_AGGREGATE_BYTES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> str:
    if not 1 <= candidates_per_scene <= MAX_CANDIDATES_PER_SCENE:
        raise ValueError("candidates_per_scene is outside the supported range")
    if not 0 <= max_remote_downloads <= MAX_REMOTE_DOWNLOADS:
        raise ValueError("max_remote_downloads is outside the supported range")
    if not 0 <= max_aggregate_bytes <= MAX_AGGREGATE_BYTES:
        raise ValueError("max_aggregate_bytes is outside the supported range")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("concurrency is outside the supported range")

    source_path = Path(candidate_manifest_path)
    source_bytes = source_path.read_bytes()
    source = SceneCandidateManifest.model_validate_json(source_bytes)
    if source.provider not in scene_preview_cache.ALLOWED_HOSTS:
        return ""
    _validate_source_manifest(source)

    results = [[None for _ in group.candidates] for group in source.scenes]
    budget = scene_preview_cache.AggregateByteBudget(max_aggregate_bytes)
    remote_started = 0
    cache_hits = 0
    in_task_reuses = 0
    identities = {}

    max_candidates = max((len(group.candidates) for group in source.scenes), default=0)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for position in range(max_candidates):
            misses_by_identity = {}
            for scene_pos, group in enumerate(source.scenes):
                if position >= len(group.candidates):
                    continue
                candidate = group.candidates[position]
                if position >= candidates_per_scene:
                    results[scene_pos][position] = _failure(
                        candidate.candidate_id,
                        "budget_exhausted",
                        "per_scene_candidate_limit",
                    )
                    continue
                if not candidate.preview_url:
                    results[scene_pos][position] = _failure(
                        candidate.candidate_id, "missing_url", "preview_url_missing"
                    )
                    continue
                identity = (
                    candidate.provider,
                    candidate.provider_video_id,
                    candidate.preview_url,
                    scene_preview_cache.NORMALIZATION_VERSION,
                )
                if identity in identities:
                    results[scene_pos][position] = identities[identity].model_copy(
                        deep=True
                    )
                    in_task_reuses += 1
                    continue
                try:
                    cached = scene_preview_cache.find_cached(
                        candidate.provider,
                        candidate.provider_video_id,
                        candidate.preview_url,
                    )
                except scene_preview_cache.PreviewError as exc:
                    outcome = _failure(candidate.candidate_id, exc.status, exc.reason)
                    results[scene_pos][position] = outcome
                    identities[identity] = outcome
                    continue
                except Exception:
                    outcome = _failure(
                        candidate.candidate_id, "download_failed", "network_error"
                    )
                    results[scene_pos][position] = outcome
                    identities[identity] = outcome
                    continue
                if cached:
                    outcome = CandidatePreviewResult(
                        candidate_id=candidate.candidate_id,
                        status="ready",
                        preview=PreviewFile(
                            normalization_version=scene_preview_cache.NORMALIZATION_VERSION,
                            **{
                                key: value
                                for key, value in cached.__dict__.items()
                                if key != "cache_hit"
                            },
                        ),
                    )
                    results[scene_pos][position] = outcome
                    identities[identity] = outcome
                    cache_hits += 1
                else:
                    occurrence = (scene_pos, position, candidate)
                    if identity in misses_by_identity:
                        misses_by_identity[identity][1].append(occurrence)
                        in_task_reuses += 1
                    else:
                        misses_by_identity[identity] = [occurrence, []]

            remaining = max_remote_downloads - remote_started
            misses = list(misses_by_identity.items())
            selected_indexes = set(
                centered_stratified_indices(len(misses), min(remaining, len(misses)))
            )
            futures = {}
            for miss_index, item in enumerate(misses):
                identity, (leader, followers) = item
                scene_pos, candidate_pos, candidate = leader
                if miss_index not in selected_indexes:
                    outcome = _failure(
                        candidate.candidate_id,
                        "budget_exhausted",
                        "remote_download_budget_exhausted",
                    )
                    results[scene_pos][candidate_pos] = outcome
                    for follower_scene, follower_position, follower in followers:
                        results[follower_scene][follower_position] = outcome.model_copy(
                            update={"candidate_id": follower.candidate_id}, deep=True
                        )
                    identities[identity] = outcome
                    continue
                remote_started += 1
                future = executor.submit(
                    scene_preview_cache.prepare_preview,
                    candidate.provider,
                    candidate.provider_video_id,
                    candidate.preview_url,
                    budget,
                )
                futures[future] = (identity, leader, followers)
            for future in as_completed(futures):
                identity, leader, followers = futures[future]
                scene_pos, candidate_pos, candidate = leader
                try:
                    prepared = future.result()
                    outcome = CandidatePreviewResult(
                        candidate_id=candidate.candidate_id,
                        status="ready",
                        preview=PreviewFile(
                            normalization_version=scene_preview_cache.NORMALIZATION_VERSION,
                            **{
                                key: value
                                for key, value in prepared.__dict__.items()
                                if key != "cache_hit"
                            },
                        ),
                    )
                except scene_preview_cache.PreviewError as exc:
                    outcome = _failure(candidate.candidate_id, exc.status, exc.reason)
                except Exception:
                    outcome = _failure(
                        candidate.candidate_id, "download_failed", "network_error"
                    )
                results[scene_pos][candidate_pos] = outcome
                for follower_scene, follower_position, follower in followers:
                    results[follower_scene][follower_position] = outcome.model_copy(
                        update={"candidate_id": follower.candidate_id}, deep=True
                    )
                identities[identity] = outcome

    scene_results = _complete_scene_results(source, results)

    limits = {
        "candidates_per_scene": candidates_per_scene,
        "max_remote_downloads_per_task": max_remote_downloads,
        "max_download_bytes_per_image": scene_preview_cache.MAX_DOWNLOAD_BYTES,
        "max_aggregate_download_bytes": max_aggregate_bytes,
        "max_decoded_pixels": scene_preview_cache.MAX_DECODED_PIXELS,
        "max_normalized_edge": scene_preview_cache.MAX_NORMALIZED_EDGE,
        "download_concurrency": concurrency,
        "max_redirects": scene_preview_cache.MAX_REDIRECTS,
        "connect_timeout_seconds": scene_preview_cache.CONNECT_TIMEOUT_SECONDS,
        "read_timeout_seconds": scene_preview_cache.READ_TIMEOUT_SECONDS,
        "total_deadline_seconds": scene_preview_cache.TOTAL_DEADLINE_SECONDS,
    }
    manifest = ScenePreviewManifest(
        source_candidate_manifest={
            "version": source.version,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        normalization_version=scene_preview_cache.NORMALIZATION_VERSION,
        provider=source.provider,
        video_aspect=source.video_aspect.value,
        limits=limits,
        usage={
            "remote_downloads_started": remote_started,
            "remote_download_bytes": budget.used,
            "cache_hits": cache_hits,
            "in_task_reuses": in_task_reuses,
        },
        scenes=scene_results,
    )
    target = source_path.parent / "scene_previews.json"
    _atomic_manifest(target, manifest)
    return str(target)
