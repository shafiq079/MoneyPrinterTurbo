import json
from pathlib import Path
from unittest.mock import patch

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
