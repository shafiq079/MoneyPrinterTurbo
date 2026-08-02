"""Prepare provider metadata for future scene-level visual ranking."""

import json
import os
import re
import tempfile
import unicodedata
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.schema import ProviderVideoCandidate, VideoAspect
from app.services import llm, material, material_cache
from app.services.scene_timeline import NarrationScene

DEFAULT_CANDIDATES_PER_SCENE = 6
MAX_CANDIDATES_PER_SCENE = 12
# Covers two distinct uncached queries for a normal 17-scene short-form task,
# with bounded headroom and the existing absolute maximum retained below.
DEFAULT_PROVIDER_SEARCH_BUDGET = 40
MAX_PROVIDER_SEARCH_BUDGET = 60
MAX_PROVIDER_RESULTS_PER_QUERY = 50
SUPPORTED_SOURCES = {"pexels", "pixabay"}
GENERIC_PRIMARY_TERMS = frozenset(
    "person people man woman child hand hands object item thing food bean beans seed seeds farm factory machine machinery liquid material process production worker footage video close up".split()
)


class SceneRetrievalStatus(str, Enum):
    complete = "complete"
    no_results_or_error = "no_results_or_error"
    budget_exhausted = "budget_exhausted"
    partial_budget_exhausted = "partial_budget_exhausted"
    hold_no_search = "hold_no_search"
    no_semantic_candidates = "no_semantic_candidates"


class SemanticTermGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical: str = Field(min_length=2, max_length=48)
    aliases: list[str] = Field(max_length=4)

    @field_validator("canonical", "aliases")
    @classmethod
    def validate_terms(cls, value):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if (
                not isinstance(item, str)
                or item != " ".join(item.split())
                or not 2 <= len(item) <= 48
                or re.search(r"[\x00-\x1f\x7f]", item)
            ):
                raise ValueError("invalid semantic term")
        return value


class SemanticRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_entities: list[SemanticTermGroup] = Field(min_length=1, max_length=4)
    actions: list[SemanticTermGroup] = Field(max_length=3)
    contexts: list[SemanticTermGroup] = Field(max_length=3)

    @model_validator(mode="after")
    def bounded(self):
        total = sum(
            len(value.encode("utf-8"))
            for groups in (self.primary_entities, self.actions, self.contexts)
            for group in groups
            for value in (group.canonical, *group.aliases)
        )
        if total > 512:
            raise ValueError("semantic requirements are too large")
        for group in self.primary_entities:
            if all(
                all(token in GENERIC_PRIMARY_TERMS for token in _tokens(phrase))
                for phrase in (group.canonical, *group.aliases)
            ):
                raise ValueError("primary entity group is overly generic")
        return self


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
    semantic_labels: list[str] = Field(default_factory=list)
    semantic_source: str = "none"


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
    requirements_status: str = "unavailable"
    semantic_requirements: SemanticRequirements | None = None


class SceneCandidateManifest(BaseModel):
    version: int = 2
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
        semantic_labels=list(item.semantic_labels),
        semantic_source=item.semantic_source,
    )


_MATCH_PUNCTUATION = re.compile(r"[^\w]+", re.UNICODE)


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(
        token for token in _MATCH_PUNCTUATION.sub(" ", normalized).split() if token
    )


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _phrase_matches(label: str, phrase: str) -> bool:
    label_tokens = _tokens(label)
    phrase_tokens = _tokens(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(label_tokens):
        return False
    singular_label = tuple(_singular(token) for token in label_tokens)
    singular_phrase = tuple(_singular(token) for token in phrase_tokens)
    width = len(phrase_tokens)
    return any(
        label_tokens[index : index + width] == phrase_tokens
        or singular_label[index : index + width] == singular_phrase
        for index in range(len(label_tokens) - width + 1)
    )


def semantic_support(item: ProviderVideoCandidate, requirements) -> bool:
    labels = tuple(item.semantic_labels)
    if not labels or item.semantic_source == "none" or requirements is None:
        return False
    return all(
        any(
            _phrase_matches(label, phrase)
            for label in labels
            for phrase in (group.canonical, *group.aliases)
        )
        for group in requirements.primary_entities
    )


def _manifest_requirements(requirements) -> SemanticRequirements | None:
    if requirements is None:
        return None
    return SemanticRequirements(
        primary_entities=[
            SemanticTermGroup(canonical=g.canonical, aliases=list(g.aliases))
            for g in requirements.primary_entities
        ],
        actions=[
            SemanticTermGroup(canonical=g.canonical, aliases=list(g.aliases))
            for g in requirements.actions
        ],
        contexts=[
            SemanticTermGroup(canonical=g.canonical, aliases=list(g.aliases))
            for g in requirements.contexts
        ],
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
    semantic_filter_enabled: bool = False,
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
    requirements: dict[int, object | None] = {}
    for scene in scenes:
        if not scene.text.strip():
            queries[scene.index] = []
            sources[scene.index] = "none"
            requirements[scene.index] = None
            continue
        plan = generated.get(scene.index)
        scene_queries = list(getattr(plan, "queries", plan or []))
        if scene_queries:
            queries[scene.index] = scene_queries[:3]
            sources[scene.index] = "llm"
            requirements[scene.index] = getattr(plan, "requirements", None)
        else:
            fallback = fallback_scene_query(scene.text)
            queries[scene.index] = [fallback] if fallback else []
            sources[scene.index] = "fallback"
            requirements[scene.index] = None

    resolved: dict[str, list[ProviderVideoCandidate]] = {}
    skipped: dict[int, int] = {scene.index: 0 for scene in scenes}
    attempted: dict[int, int] = {scene.index: 0 for scene in scenes}
    found: dict[int, list[tuple[str, list[ProviderVideoCandidate]]]] = {
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
            items = items[:MAX_PROVIDER_RESULTS_PER_QUERY]
            eligible = (
                [
                    item
                    for item in items
                    if semantic_support(item, requirements[scene.index])
                ]
                if semantic_filter_enabled
                else items
            )
            found[scene.index].append((query, eligible))

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
                    requirements_status="unavailable",
                )
            )
            continue
        unique = []
        seen = set()
        result_sets = found[scene.index]
        representatives = {}
        for query_position, (query, items) in enumerate(result_sets):
            for item in items:
                identity = (item.provider, item.provider_video_id)
                key = (item.provider_rank, query_position)
                if (
                    identity not in representatives
                    or key < representatives[identity][0]
                ):
                    representatives[identity] = (key, query, item)
        # Interleave provider-ranked result sets so a prolific first query
        # cannot starve later queries.  Each row retains its originating query
        # and rank; identity deduplication is stable on first occurrence.
        max_results = max((len(items) for _, items in result_sets), default=0)
        for rank_position in range(max_results):
            for query, items in result_sets:
                if rank_position >= len(items):
                    continue
                item = items[rank_position]
                identity = (item.provider, item.provider_video_id)
                if (
                    representatives[identity][1] != query
                    or representatives[identity][2] is not item
                ):
                    continue
                if identity in seen:
                    continue
                seen.add(identity)
                unique.append(_candidate(item, query))
                if len(unique) == candidates_per_scene:
                    break
            if len(unique) == candidates_per_scene:
                break
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
        elif semantic_filter_enabled and attempted[scene.index]:
            status = SceneRetrievalStatus.no_semantic_candidates
            warning = "Provider returned no candidates with required semantic evidence."
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
                requirements_status=(
                    "generated"
                    if requirements[scene.index] is not None
                    else "unavailable"
                ),
                semantic_requirements=_manifest_requirements(requirements[scene.index]),
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
