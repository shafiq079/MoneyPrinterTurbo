import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import scene_preview
from app.services.scene_preview_cache import PreparedPreview, PreviewError
from app.services.scene_candidate import SceneCandidateManifest


def candidate(index, rank=1):
    return {
        "candidate_id": f"pexels:{index}",
        "provider": "pexels",
        "provider_video_id": str(index),
        "provider_page_url": None,
        "matched_query": "query",
        "provider_rank": rank,
        "preview_url": f"https://images.pexels.com/{index}.jpg",
        "video_url": f"https://video.pexels.com/{index}.mp4",
        "duration": 8,
        "width": 1080,
        "height": 1920,
    }


def manifest(path, counts, hold=False):
    scenes = []
    for scene_index, count in enumerate(counts, 1):
        is_hold = hold and scene_index == len(counts)
        scenes.append(
            {
                "scene_index": scene_index,
                "start_time": scene_index - 1,
                "end_time": scene_index,
                "duration": 1,
                "text": "" if is_hold else "scene",
                "queries": [] if is_hold else ["query"],
                "query_source": "none" if is_hold else "fallback",
                "status": "hold_no_search" if is_hold else "complete",
                "warning": None,
                "reuse_scene_index": 1 if is_hold else None,
                "candidates": []
                if is_hold
                else [
                    candidate(f"{scene_index}-{position}", position + 1)
                    for position in range(count)
                ],
            }
        )
    payload = {
        "version": 1,
        "provider": "pexels",
        "video_aspect": "9:16",
        "candidates_per_scene": 6,
        "provider_search_budget": 20,
        "remote_searches_used": 1,
        "query_generation_warning": None,
        "scenes": scenes,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def prepared(identifier):
    return PreparedPreview(
        identifier.replace("-", "0").ljust(64, "0")[:64],
        "image/jpeg",
        8,
        6,
        20,
        f"cache_candidate_previews/objects/{identifier}.jpg",
    )


def test_centered_stratified_partial_round_selection():
    assert scene_preview.centered_stratified_indices(10, 0) == []
    assert scene_preview.centered_stratified_indices(10, 1) == [5]
    assert scene_preview.centered_stratified_indices(10, 2) == [2, 7]
    assert scene_preview.centered_stratified_indices(7, 3) == [1, 3, 5]
    selected = scene_preview.centered_stratified_indices(101, 17)
    assert len(selected) == len(set(selected)) == 17
    assert selected == sorted(selected) and selected[0] > 0 and selected[-1] < 100


def test_rounds_are_fair_complete_and_do_not_copy_video_urls(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [2, 2, 2])
    calls = []

    def prepare(provider, provider_video_id, preview_url, budget):
        del provider, preview_url, budget
        calls.append(provider_video_id)
        return prepared(provider_video_id)

    with (
        patch.object(
            scene_preview.scene_preview_cache, "find_cached", return_value=None
        ),
        patch.object(
            scene_preview.scene_preview_cache, "prepare_preview", side_effect=prepare
        ),
    ):
        target = scene_preview.prepare_scene_previews(
            str(source), max_remote_downloads=4, concurrency=1
        )
    data = json.loads(Path(target).read_text())
    assert calls[:3] == ["1-0", "2-0", "3-0"]
    assert calls[3] == "2-1"  # centered one-slot selection, not the first scene
    assert [len(item["candidates"]) for item in data["scenes"]] == [2, 2, 2]
    assert "video_url" not in Path(target).read_text()
    assert data["limits"]["total_deadline_seconds"] == 30
    assert data["source_candidate_manifest"]["sha256"]


def test_cache_hits_and_reuse_do_not_consume_remote_budget(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [1, 1], hold=True)
    cached = prepared("cached")
    with (
        patch.object(
            scene_preview.scene_preview_cache, "find_cached", return_value=cached
        ),
        patch.object(scene_preview.scene_preview_cache, "prepare_preview") as remote,
    ):
        target = scene_preview.prepare_scene_previews(
            str(source), max_remote_downloads=0
        )
    data = json.loads(Path(target).read_text())
    remote.assert_not_called()
    assert data["usage"]["remote_downloads_started"] == 0
    assert data["scenes"][0]["status"] == "complete"
    assert data["scenes"][1]["status"] == "hold_no_search"


def test_failures_and_budget_exhaustion_are_truthful(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [2])
    with (
        patch.object(
            scene_preview.scene_preview_cache, "find_cached", return_value=None
        ),
        patch.object(
            scene_preview.scene_preview_cache,
            "prepare_preview",
            side_effect=PreviewError("invalid_image", "decode_failed"),
        ),
    ):
        target = scene_preview.prepare_scene_previews(
            str(source), max_remote_downloads=1
        )
    candidates = json.loads(Path(target).read_text())["scenes"][0]["candidates"]
    assert {item["status"] for item in candidates} == {
        "invalid_image",
        "budget_exhausted",
    }


def test_invalid_provider_url_is_recorded_without_aborting_manifest(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [1])
    payload = json.loads(source.read_text())
    payload["scenes"][0]["candidates"][0]["preview_url"] = (
        "https://attacker.test/poster.jpg"
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    target = scene_preview.prepare_scene_previews(str(source))

    result = json.loads(Path(target).read_text())["scenes"][0]
    assert result["status"] == "no_previews"
    assert result["candidates"][0]["status"] == "invalid_url"
    assert result["candidates"][0]["failure_reason"] == "provider_host_not_allowed"


def test_same_round_duplicate_identity_uses_one_slot_without_starving_unique(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [1, 1, 1])
    payload = json.loads(source.read_text())
    payload["scenes"][1]["candidates"][0] = payload["scenes"][0]["candidates"][0]
    source.write_text(json.dumps(payload), encoding="utf-8")
    calls = []

    def prepare(provider, provider_video_id, preview_url, budget):
        del provider, preview_url, budget
        calls.append(provider_video_id)
        return prepared(provider_video_id)

    with (
        patch.object(
            scene_preview.scene_preview_cache, "find_cached", return_value=None
        ),
        patch.object(
            scene_preview.scene_preview_cache, "prepare_preview", side_effect=prepare
        ),
    ):
        target = scene_preview.prepare_scene_previews(
            str(source), max_remote_downloads=2, concurrency=1
        )

    data = json.loads(Path(target).read_text())
    assert calls == ["1-0", "3-0"]
    assert data["usage"]["remote_downloads_started"] == 2
    assert data["usage"]["in_task_reuses"] == 1
    assert [scene["candidates"][0]["status"] for scene in data["scenes"]] == [
        "ready",
        "ready",
        "ready",
    ]


def test_incomplete_internal_results_cannot_be_finalized(tmp_path):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [2])
    parsed = SceneCandidateManifest.model_validate_json(source.read_bytes())
    with pytest.raises(ValueError, match="incomplete"):
        scene_preview._complete_scene_results(parsed, [[None, None]])


@pytest.mark.parametrize(
    "mutation",
    [
        "hold_candidates",
        "hold_queries",
        "provider",
        "candidate_id",
        "duplicate_scene",
        "out_of_order",
        "declared_limit",
        "hard_limit",
    ],
)
def test_invalid_source_invariants_do_no_cache_or_network_work(tmp_path, mutation):
    source = tmp_path / "scene_candidates.json"
    manifest(source, [1, 1])
    payload = json.loads(source.read_text())
    if mutation.startswith("hold_"):
        scene = payload["scenes"][0]
        scene["status"] = "hold_no_search"
        scene["text"] = ""
        scene["query_source"] = "none"
        scene["reuse_scene_index"] = 2
        if mutation == "hold_queries":
            scene["candidates"] = []
        else:
            scene["queries"] = []
    elif mutation == "provider":
        payload["scenes"][0]["candidates"][0]["provider"] = "pixabay"
    elif mutation == "candidate_id":
        payload["scenes"][0]["candidates"][0]["candidate_id"] = "pexels:wrong"
    elif mutation == "duplicate_scene":
        payload["scenes"][1]["scene_index"] = 1
    elif mutation == "out_of_order":
        payload["scenes"][0]["scene_index"] = 3
    elif mutation == "declared_limit":
        payload["candidates_per_scene"] = 1
        payload["scenes"][0]["candidates"].append(candidate("1-extra", 2))
    else:
        payload["candidates_per_scene"] = 13
    source.write_text(json.dumps(payload), encoding="utf-8")

    with (
        patch.object(scene_preview.scene_preview_cache, "find_cached") as cached,
        patch.object(scene_preview.scene_preview_cache, "prepare_preview") as remote,
        pytest.raises(ValueError),
    ):
        scene_preview.prepare_scene_previews(str(source))

    cached.assert_not_called()
    remote.assert_not_called()
    assert not (tmp_path / "scene_previews.json").exists()
