"""Prepare provider metadata for future scene-level visual ranking."""

import json
import os
import re
import tempfile
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from app.models.schema import ProviderVideoCandidate, VideoAspect
from app.services import llm, material, material_cache
from app.services.scene_timeline import NarrationScene

DEFAULT_CANDIDATES_PER_SCENE = 6
MAX_CANDIDATES_PER_SCENE = 12
DEFAULT_PROVIDER_SEARCH_BUDGET = 20
MAX_PROVIDER_SEARCH_BUDGET = 60
SUPPORTED_SOURCES = {"pexels", "pixabay"}


class SceneRetrievalStatus(str, Enum):
    complete = "complete"
    no_results_or_error = "no_results_or_error"
    budget_exhausted = "budget_exhausted"
    partial_budget_exhausted = "partial_budget_exhausted"
    hold_no_search = "hold_no_search"


class SceneMaterialCandidate(BaseModel):
    candidate_id: str
    provider: str
    provider_video_id: str
    provider_page_url: str | None = None
    matched_query: str
    provider_rank: int
    preview_url: str | None = None
    video_url: str
    duration: float
    width: int
    height: int


class SceneCandidateGroup(BaseModel):
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    text: str
    queries: list[str]
    query_source: str
    status: SceneRetrievalStatus
    warning: str | None = None
    reuse_scene_index: int | None = None
    candidates: list[SceneMaterialCandidate]


class SceneCandidateManifest(BaseModel):
    version: int = 1
    provider: str
    video_aspect: VideoAspect
    candidates_per_scene: int
    provider_search_budget: int
    remote_searches_used: int
    query_generation_warning: str | None = None
    scenes: list[SceneCandidateGroup]


def fallback_scene_query(text: str) -> str:
    normalized = " ".join(text.split()).strip(" ,.;:!?，。；：！？")
    if not normalized:
        return ""
    if re.search(r"\s", normalized):
        selected: list[str] = []
        for word in normalized.split()[:12]:
            proposed = " ".join([*selected, word])
            if len(proposed) > 80:
                break
            selected.append(word)
        # A pathological first token longer than the complete query bound has no
        # usable word boundary. Keep the hard storage/provider limit deterministic.
        return " ".join(selected) or normalized.split()[0][:80]
    return normalized[:80]


def _reuse_scene_index(scenes: list[NarrationScene], position: int) -> int | None:
    for scene in reversed(scenes[:position]):
        if scene.text.strip():
            return scene.index
    for scene in scenes[position + 1 :]:
        if scene.text.strip():
            return scene.index
    return None


def _candidate(item: ProviderVideoCandidate, query: str) -> SceneMaterialCandidate:
    return SceneMaterialCandidate(
        candidate_id=f"{item.provider}:{item.provider_video_id}",
        provider=item.provider,
        provider_video_id=item.provider_video_id,
        provider_page_url=item.provider_page_url or None,
        matched_query=query,
        provider_rank=item.provider_rank,
        preview_url=item.preview_url or None,
        video_url=item.url,
        duration=item.duration,
        width=item.width,
        height=item.height,
    )


def retrieve_scene_candidates(
    task_dir: str,
    video_subject: str,
    scenes: list[NarrationScene],
    source: str,
    video_aspect: VideoAspect,
    minimum_duration: int,
    *,
    candidates_per_scene: int = DEFAULT_CANDIDATES_PER_SCENE,
    provider_search_budget: int = DEFAULT_PROVIDER_SEARCH_BUDGET,
) -> str:
    if source not in SUPPORTED_SOURCES:
        return ""
    if not 1 <= candidates_per_scene <= MAX_CANDIDATES_PER_SCENE:
        raise ValueError("candidates_per_scene is outside the supported range")
    if not 0 <= provider_search_budget <= MAX_PROVIDER_SEARCH_BUDGET:
        raise ValueError("provider_search_budget is outside the supported range")

    generated, generation_warning = llm.generate_scene_queries(video_subject, scenes)
    queries: dict[int, list[str]] = {}
    sources: dict[int, str] = {}
    for scene in scenes:
        if not scene.text.strip():
            queries[scene.index] = []
            sources[scene.index] = "none"
            continue
        scene_queries = generated.get(scene.index) or []
        if scene_queries:
            queries[scene.index] = scene_queries[:3]
            sources[scene.index] = "llm"
        else:
            fallback = fallback_scene_query(scene.text)
            queries[scene.index] = [fallback] if fallback else []
            sources[scene.index] = "fallback"

    resolved: dict[str, list[ProviderVideoCandidate]] = {}
    skipped: dict[int, int] = {scene.index: 0 for scene in scenes}
    attempted: dict[int, int] = {scene.index: 0 for scene in scenes}
    found: dict[int, list[tuple[str, ProviderVideoCandidate]]] = {
        scene.index: [] for scene in scenes
    }
    remote_used = 0
    max_query_count = max((len(value) for value in queries.values()), default=0)
    aspect = VideoAspect(video_aspect)

    # Round-robin by query position gives every narration scene an opportunity
    # before additional diversity is fetched for early scenes.
    for query_position in range(max_query_count):
        for scene in scenes:
            scene_queries = queries[scene.index]
            if query_position >= len(scene_queries):
                continue
            query = scene_queries[query_position]
            cache_key = query.casefold()
            if cache_key in resolved:
                items = resolved[cache_key]
            else:
                cache_args = {
                    "provider": source,
                    "search_term": query,
                    "minimum_duration": minimum_duration,
                    "video_aspect": aspect,
                }
                items = material_cache.load_material_candidate_search_cache(
                    **cache_args
                )
                if items is None:
                    if remote_used >= provider_search_budget:
                        skipped[scene.index] += 1
                        continue
                    items, used_remote = material.search_video_candidates_with_cache(
                        source=source,
                        search_term=query,
                        minimum_duration=minimum_duration,
                        video_aspect=aspect,
                    )
                    remote_used += int(used_remote)
                resolved[cache_key] = items
            attempted[scene.index] += 1
            found[scene.index].extend((query, item) for item in items)

    groups = []
    for position, scene in enumerate(scenes):
        if not scene.text.strip():
            groups.append(
                SceneCandidateGroup(
                    scene_index=scene.index,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    duration=scene.duration,
                    text=scene.text,
                    queries=[],
                    query_source="none",
                    status=SceneRetrievalStatus.hold_no_search,
                    reuse_scene_index=_reuse_scene_index(scenes, position),
                    candidates=[],
                )
            )
            continue
        unique = []
        seen = set()
        for query, item in found[scene.index]:
            identity = (item.provider, item.provider_video_id)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(_candidate(item, query))
        unique = unique[:candidates_per_scene]
        warning = None
        if skipped[scene.index]:
            if attempted[scene.index]:
                status = SceneRetrievalStatus.partial_budget_exhausted
            else:
                status = SceneRetrievalStatus.budget_exhausted
            warning = (
                f"Provider search budget exhausted; {skipped[scene.index]} "
                "scene queries were not searched."
            )
        elif unique:
            status = SceneRetrievalStatus.complete
        else:
            status = SceneRetrievalStatus.no_results_or_error
            warning = "Provider returned no candidates or the search failed."
        groups.append(
            SceneCandidateGroup(
                scene_index=scene.index,
                start_time=scene.start_time,
                end_time=scene.end_time,
                duration=scene.duration,
                text=scene.text,
                queries=queries[scene.index],
                query_source=sources[scene.index],
                status=status,
                warning=warning,
                candidates=unique,
            )
        )

    manifest = SceneCandidateManifest(
        provider=source,
        video_aspect=aspect,
        candidates_per_scene=candidates_per_scene,
        provider_search_budget=provider_search_budget,
        remote_searches_used=remote_used,
        query_generation_warning=generation_warning,
        scenes=groups,
    )
    os.makedirs(task_dir, exist_ok=True)
    target = Path(task_dir) / "scene_candidates.json"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=task_dir, suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                manifest.model_dump(mode="json"),
                temp_file,
                ensure_ascii=False,
                indent=2,
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return str(target)
