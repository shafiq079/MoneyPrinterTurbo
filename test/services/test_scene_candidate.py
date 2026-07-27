import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.models.schema import ProviderVideoCandidate, VideoAspect
from app.services import llm, scene_candidate
from app.services.scene_timeline import NarrationScene


def _scene(index, text):
    return NarrationScene(
        index=index,
        start_time=index - 1,
        end_time=index,
        duration=1,
        text=text,
    )


def _item(candidate_id, rank=1):
    return ProviderVideoCandidate(
        provider="pexels",
        provider_video_id=candidate_id,
        provider_page_url=f"https://pexels.test/{candidate_id}",
        preview_url=f"https://images.test/{candidate_id}.jpg",
        url=f"https://videos.test/{candidate_id}.mp4",
        duration=8,
        width=1080,
        height=1920,
        provider_rank=rank,
    )


def test_fallback_is_deterministic_and_bounded():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    assert scene_candidate.fallback_scene_query(text) == " ".join(text.split()[:12])
    assert (
        scene_candidate.fallback_scene_query("这是可见的城市街道场景")
        == "这是可见的城市街道场景"
    )


def test_fallback_spaced_text_stops_at_complete_word_boundary():
    words = ["cinematic", "pedestrians", "crossing", "a", "bright", "downtown"] * 3
    text = " ".join(words)
    result = scene_candidate.fallback_scene_query(text)

    assert len(result) <= 80
    assert result.split() == words[: len(result.split())]
    assert result.endswith(result.split()[-1])
    assert len(result.split()) <= 12


def test_query_generation_uses_one_batched_call_and_ignores_holds():
    scenes = [_scene(1, "Busy station"), _scene(2, ""), _scene(3, "Card payment")]
    response = json.dumps(
        {
            "scenes": [
                {"scene_index": 1, "queries": ["busy station", "busy station"]},
                {"scene_index": 3, "queries": ["contactless card payment"]},
            ]
        }
    )
    with patch.object(llm, "_generate_response", return_value=response) as generate:
        queries, warning = llm.generate_scene_queries("City life", scenes)

    assert queries == {1: ["busy station"], 3: ["contactless card payment"]}
    assert warning is None
    assert generate.call_count == 1
    assert '"scene_index": 2' not in generate.call_args.args[0]


@pytest.mark.parametrize("response", ["not json", "Error: unavailable"])
def test_failed_or_malformed_llm_output_uses_deterministic_fallback(tmp_path, response):
    scenes = [_scene(1, "People walking through a sunny public park")]
    with (
        patch.object(llm, "_max_retries", 1),
        patch.object(llm, "_generate_response", return_value=response),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=[],
        ),
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path), "Parks", scenes, "pexels", VideoAspect.portrait, 5
        )

    group = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]
    assert group["query_source"] == "fallback"
    assert group["queries"] == [scene_candidate.fallback_scene_query(scenes[0].text)]


def test_partial_duplicate_and_unknown_llm_scenes_fall_back_safely(tmp_path):
    scenes = [_scene(1, "First scene"), _scene(2, "Second scene"), _scene(3, "Third")]
    response = json.dumps(
        {
            "scenes": [
                {"scene_index": 1, "queries": ["valid first footage"]},
                {"scene_index": 2, "queries": ["duplicate one"]},
                {"scene_index": 2, "queries": ["duplicate two"]},
                {"scene_index": 999, "queries": ["unknown"]},
            ]
        }
    )
    with (
        patch.object(llm, "_generate_response", return_value=response),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=[],
        ),
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path), "Subject", scenes, "pexels", VideoAspect.portrait, 5
        )

    groups = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"]
    assert groups[0]["queries"] == ["valid first footage"]
    assert groups[0]["query_source"] == "llm"
    assert groups[1]["query_source"] == "fallback"
    assert groups[2]["query_source"] == "fallback"


def test_manifest_keeps_holds_combines_queries_and_limits_after_dedup(tmp_path):
    scenes = [_scene(1, "Busy station"), _scene(2, "")]
    results = {
        "busy train station": [_item("1"), _item("2", 2)],
        "commuters platform": [_item("1"), _item("3", 2)],
    }

    def search(**kwargs):
        return results[kwargs["search_term"]], True

    with (
        patch.object(
            scene_candidate.llm,
            "generate_scene_queries",
            return_value=(
                {1: ["busy train station", "commuters platform"]},
                None,
            ),
        ),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=None,
        ),
        patch.object(
            scene_candidate.material,
            "search_video_candidates_with_cache",
            side_effect=search,
        ) as provider_search,
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path),
            "Transit",
            scenes,
            "pexels",
            VideoAspect.portrait,
            5,
            candidates_per_scene=2,
        )

    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert [item["provider_video_id"] for item in data["scenes"][0]["candidates"]] == [
        "1",
        "2",
    ]
    assert data["scenes"][1]["status"] == "hold_no_search"
    assert data["scenes"][1]["queries"] == []
    assert data["scenes"][1]["reuse_scene_index"] == 1
    assert provider_search.call_count == 2


def test_default_six_limit_is_applied_after_dedup_and_preserves_utf8(tmp_path):
    scenes = [_scene(1, "城市街道上的行人"), _scene(2, "第二个场景")]
    first = [_item(str(index), index) for index in range(1, 7)]
    second = [_item("1"), _item("7", 2), _item("8", 3)]

    def search(**kwargs):
        return (first if kwargs["search_term"] == "city street" else second), True

    with (
        patch.object(
            scene_candidate.llm,
            "generate_scene_queries",
            return_value=({1: ["city street", "walking people"], 2: ["next"]}, None),
        ),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=None,
        ),
        patch.object(
            scene_candidate.material,
            "search_video_candidates_with_cache",
            side_effect=search,
        ),
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path), "City", scenes, "pexels", VideoAspect.portrait, 5
        )

    manifest = json.loads(Path(target).read_text(encoding="utf-8"))
    assert manifest["candidates_per_scene"] == 6
    assert len(manifest["scenes"]) == len(scenes)
    assert manifest["scenes"][0]["text"] == "城市街道上的行人"
    assert [
        item["provider_video_id"] for item in manifest["scenes"][0]["candidates"]
    ] == ["1", "2", "3", "4", "5", "6"]


@pytest.mark.parametrize("limit", [0, -1, scene_candidate.MAX_CANDIDATES_PER_SCENE + 1])
def test_invalid_candidate_limits_raise(tmp_path, limit):
    with pytest.raises(ValueError):
        scene_candidate.retrieve_scene_candidates(
            str(tmp_path),
            "Subject",
            [_scene(1, "Scene")],
            "pexels",
            VideoAspect.portrait,
            5,
            candidates_per_scene=limit,
        )


def test_budget_round_robin_represents_later_scenes(tmp_path):
    scenes = [_scene(1, "First"), _scene(2, "Second")]
    with (
        patch.object(
            scene_candidate.llm,
            "generate_scene_queries",
            return_value=({1: ["first a", "first b"], 2: ["second a"]}, None),
        ),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=None,
        ),
        patch.object(
            scene_candidate.material,
            "search_video_candidates_with_cache",
            return_value=([], True),
        ) as search,
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path),
            "Subject",
            scenes,
            "pexels",
            VideoAspect.portrait,
            5,
            provider_search_budget=1,
        )

    groups = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"]
    assert search.call_args.kwargs["search_term"] == "first a"
    assert search.call_count == 1
    assert groups[0]["status"] == "partial_budget_exhausted"
    assert groups[1]["status"] == "budget_exhausted"


def test_rich_cache_hit_uses_no_budget_and_shared_query_resolves_once(tmp_path):
    scenes = [_scene(1, "First"), _scene(2, "Second")]
    cached = [_item("cached")]
    with (
        patch.object(
            scene_candidate.llm,
            "generate_scene_queries",
            return_value=({1: ["shared query"], 2: ["shared query"]}, None),
        ),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=cached,
        ) as cache_load,
        patch.object(
            scene_candidate.material, "search_video_candidates_with_cache"
        ) as remote,
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path),
            "Subject",
            scenes,
            "pexels",
            VideoAspect.portrait,
            5,
            provider_search_budget=0,
        )

    manifest = json.loads(Path(target).read_text(encoding="utf-8"))
    assert manifest["remote_searches_used"] == 0
    assert cache_load.call_count == 1
    remote.assert_not_called()
    assert [group["status"] for group in manifest["scenes"]] == ["complete", "complete"]


def test_empty_search_uses_truthful_combined_status(tmp_path):
    with (
        patch.object(
            scene_candidate.llm,
            "generate_scene_queries",
            return_value=({1: ["empty"]}, None),
        ),
        patch.object(
            scene_candidate.material_cache,
            "load_material_candidate_search_cache",
            return_value=None,
        ),
        patch.object(
            scene_candidate.material,
            "search_video_candidates_with_cache",
            return_value=([], True),
        ),
    ):
        target = scene_candidate.retrieve_scene_candidates(
            str(tmp_path),
            "Subject",
            [_scene(1, "Scene")],
            "pexels",
            VideoAspect.portrait,
            5,
        )
    assert (
        json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]["status"]
        == "no_results_or_error"
    )


def test_unsupported_source_has_no_llm_or_artifact(tmp_path):
    with patch.object(scene_candidate.llm, "generate_scene_queries") as generate:
        assert (
            scene_candidate.retrieve_scene_candidates(
                str(tmp_path),
                "Subject",
                [_scene(1, "Scene")],
                "coverr",
                VideoAspect.portrait,
                5,
            )
            == ""
        )
    generate.assert_not_called()
    assert not (tmp_path / "scene_candidates.json").exists()
