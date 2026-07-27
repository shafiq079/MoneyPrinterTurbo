import json
from pathlib import Path
from unittest.mock import patch

from app.services import scene_preview
from app.services.scene_preview_cache import PreparedPreview, PreviewError


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
