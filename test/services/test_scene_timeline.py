import json
import math
from pathlib import Path

import pytest

from app.services.scene_timeline import build_scenes, create_scene_timeline


def write_srt(tmp_path: Path, content: str) -> str:
    target = tmp_path / "subtitle.srt"
    target.write_text(content, encoding="utf-8")
    return str(target)


def assert_valid_timeline(scenes, duration):
    assert [scene.index for scene in scenes] == list(range(1, len(scenes) + 1))
    assert scenes[0].start_time == pytest.approx(0)
    assert scenes[-1].end_time == pytest.approx(duration)
    for previous, scene in zip([None, *scenes], scenes):
        assert 0 <= scene.start_time < scene.end_time <= duration
        assert scene.duration == pytest.approx(scene.end_time - scene.start_time)
        assert all(
            math.isfinite(value)
            for value in (scene.start_time, scene.end_time, scene.duration)
        )
        if previous is not None:
            assert previous.end_time == pytest.approx(scene.start_time)


def test_valid_srt_timestamps_and_multiline_text(tmp_path):
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,250 --> 00:00:02,000\nFirst line\nsecond line\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nLast scene\n",
    )

    scenes = build_scenes("First line second line. Last scene.", 5, subtitle, 10)

    assert [(scene.start_time, scene.end_time, scene.text) for scene in scenes] == [
        (0.0, 0.25, ""),
        (0.25, 2.0, "First line second line"),
        (2.0, 2.5, ""),
        (2.5, 4.0, "Last scene"),
        (4.0, 5.0, ""),
    ]
    assert_valid_timeline(scenes, 5)


def test_srt_without_trailing_blank_line_is_included(tmp_path):
    subtitle = write_srt(
        tmp_path, "1\n00:00:00,000 --> 00:00:01,000\nNo trailing blank"
    )
    scenes = build_scenes("No trailing blank", 2, subtitle)
    assert len(scenes) == 2
    assert scenes[0].text == "No trailing blank"
    assert scenes[1].text == ""


def test_malformed_zero_reversed_negative_and_overlapping_entries_are_safe(tmp_path):
    subtitle = write_srt(
        tmp_path,
        "1\nnot a timestamp\nMalformed\n\n"
        "2\n00:00:01,000 --> 00:00:01,000\nZero\n\n"
        "3\n00:00:03,000 --> 00:00:02,000\nReversed\n\n"
        "4\n-00:00:01,000 --> 00:00:02,000\nNegative\n\n"
        "5\n00:00:01,000 --> 00:00:03,000\nGood\n\n"
        "6\n00:00:02,000 --> 00:00:04,000\nOverlap trimmed\n",
    )
    scenes = build_scenes("Good. Overlap trimmed.", 4, subtitle, 10)
    assert [(scene.text, scene.start_time, scene.end_time) for scene in scenes] == [
        ("", 0.0, 1.0),
        ("Good", 1.0, 3.0),
        ("Overlap trimmed", 3.0, 4.0),
    ]
    assert_valid_timeline(scenes, 4)


def test_no_valid_subtitles_uses_proportional_narration_fallback(tmp_path):
    subtitle = write_srt(tmp_path, "1\ninvalid\nignored\n")
    scenes = build_scenes("Hi. This is longer!", 9, subtitle, 20)
    assert [scene.text for scene in scenes] == ["Hi", "This is longer"]
    assert scenes[0].duration == pytest.approx(9 * 2 / 14)
    assert scenes[-1].end_time == 9
    assert_valid_timeline(scenes, 9)


def test_long_segment_is_split_to_maximum_clip_duration(tmp_path):
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:12,000\n"
        "one two three four five six seven eight nine\n",
    )
    scenes = build_scenes(
        "one two three four five six seven eight nine", 12, subtitle, 5
    )
    assert len(scenes) == 3
    assert " ".join(scene.text for scene in scenes) == (
        "one two three four five six seven eight nine"
    )
    assert max(scene.duration for scene in scenes) <= 5
    assert_valid_timeline(scenes, 12)


def test_uneven_word_lengths_never_exceed_maximum_clip_duration(tmp_path):
    text = "a extraordinarilylongword another tiny word"
    subtitle = write_srt(tmp_path, f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n")

    scenes = build_scenes(text, 10, subtitle, 3)

    assert len(scenes) == 4
    assert all(scene.duration <= 3 for scene in scenes)
    assert " ".join(scene.text for scene in scenes) == text
    assert_valid_timeline(scenes, 10)


def test_indivisible_short_text_still_honors_maximum_clip_duration():
    scenes = build_scenes("好", 10, max_clip_duration=3)

    assert len(scenes) == 4
    assert all(scene.duration <= 3 for scene in scenes)
    assert "".join(scene.text for scene in scenes) == "好"
    assert scenes[-1].end_time == 10


def test_short_spaced_phrase_keeps_words_and_uses_timed_holds():
    narration = "short phrase"

    scenes = build_scenes(narration, 12, max_clip_duration=3)

    nonempty_text = [scene.text for scene in scenes if scene.text]
    assert len(scenes) == 4
    assert nonempty_text == ["short", "phrase"]
    assert all(scene.duration <= 3 for scene in scenes)
    assert " ".join(nonempty_text) == narration
    assert scenes[0].start_time == 0
    assert scenes[-1].end_time == 12
    assert_valid_timeline(scenes, 12)


def test_partially_valid_srt_falls_back_without_losing_narration(tmp_path):
    narration = "Opening scene. Missing middle narration. Closing scene."
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:02,000\nOpening scene\n\n"
        "2\nmalformed timestamp\nMissing middle narration\n",
    )

    scenes = build_scenes(narration, 9, subtitle, 10)

    assert [scene.text for scene in scenes] == [
        "Opening scene",
        "Missing middle narration",
        "Closing scene",
    ]
    assert "".join(
        character for scene in scenes for character in scene.text if character.isalnum()
    ) == ("".join(character for character in narration if character.isalnum()))
    assert scenes[0].start_time == 0
    assert scenes[-1].end_time == 9
    assert_valid_timeline(scenes, 9)


def test_unspaced_multilingual_text_splits_without_losing_order():
    narration = "这是一个没有空格的中文旁白片段"
    scenes = build_scenes(narration, 9, max_clip_duration=3)
    assert len(scenes) == 3
    assert "".join(scene.text for scene in scenes) == narration
    assert scenes[-1].end_time == 9
    assert_valid_timeline(scenes, 9)


def test_create_scene_timeline_writes_utf8_json(tmp_path):
    target = create_scene_timeline(str(tmp_path), "你好世界。", 4)
    assert target == str(tmp_path / "scenes.json")
    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert data[0]["text"] == "你好世界"
    assert set(data[0]) == {"index", "start_time", "end_time", "duration", "text"}


def test_minimum_merges_short_cues_and_preserves_exact_text(tmp_path):
    narration = "Hello, WORLD! Café's #1. Next?"
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:00,600\nHello, WORLD!\n\n"
        "2\n00:00:00,600 --> 00:00:01,200\nCafé's #1.\n\n"
        "3\n00:00:01,200 --> 00:00:02,400\nNext?\n",
    )

    scenes = build_scenes(narration, 2.4, subtitle, 3, min_scene_duration=2)

    assert [(scene.text, scene.start_time, scene.end_time) for scene in scenes] == [
        ("Hello, WORLD! Café's #1. Next?", 0, 2.4)
    ]
    assert_valid_timeline(scenes, 2.4)


def test_minimum_preserves_unspaced_script_without_inserting_spaces(tmp_path):
    narration = "你好，世界！心理健康。"
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:00,700\n你好，\n\n"
        "2\n00:00:00,700 --> 00:00:01,400\n世界！\n\n"
        "3\n00:00:01,400 --> 00:00:02,400\n心理健康。\n",
    )

    scenes = build_scenes(narration, 2.4, subtitle, 3, min_scene_duration=2)

    assert "".join(scene.text for scene in scenes if scene.text) == narration
    assert_valid_timeline(scenes, 2.4)


def test_minimum_absorbs_only_approved_internal_hold(tmp_path):
    narration = "First. Second. Third."
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,200 --> 00:00:00,800\nFirst.\n\n"
        "2\n00:00:01,150 --> 00:00:01,700\nSecond.\n\n"
        "3\n00:00:02,052 --> 00:00:02,600\nThird.\n",
    )

    scenes = build_scenes(narration, 2.8, subtitle, 3, min_scene_duration=2)

    assert [(scene.text, scene.start_time, scene.end_time) for scene in scenes] == [
        ("", 0, 0.2),
        ("First. Second.", 0.2, 1.7),
        ("", 1.7, 2.052),
        ("Third.", 2.052, 2.6),
        ("", 2.6, 2.8),
    ]
    assert_valid_timeline(scenes, 2.8)


def test_minimum_is_best_effort_and_never_exceeds_maximum(tmp_path):
    narration = "Long enough. Tiny."
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:02,700\nLong enough.\n\n"
        "2\n00:00:02,700 --> 00:00:03,100\nTiny.\n",
    )

    scenes = build_scenes(narration, 3.1, subtitle, 3, min_scene_duration=2)

    assert [scene.text for scene in scenes] == ["Long enough.", "Tiny."]
    assert all(scene.duration <= 3 for scene in scenes)
    assert_valid_timeline(scenes, 3.1)


def test_final_short_cue_merges_backward_before_trailing_hold(tmp_path):
    narration = "First part. End."
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:01,500\nFirst part.\n\n"
        "2\n00:00:01,500 --> 00:00:02,000\nEnd.\n",
    )

    scenes = build_scenes(narration, 2.5, subtitle, 3, min_scene_duration=2)

    assert [(scene.text, scene.start_time, scene.end_time) for scene in scenes] == [
        ("First part. End.", 0, 2),
        ("", 2, 2.5),
    ]


def test_none_minimum_keeps_existing_maximum_split_output():
    baseline = build_scenes("one two three four", 8, max_clip_duration=3)
    explicit_none = build_scenes(
        "one two three four", 8, max_clip_duration=3, min_scene_duration=None
    )
    assert explicit_none == baseline


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (0, 3, "finite positive"),
        (math.inf, 3, "finite positive"),
        (2, None, "max_clip_duration"),
        (2, math.nan, "max_clip_duration"),
        (4, 3, "less than or equal"),
    ],
)
def test_active_minimum_rejects_invalid_bounds(minimum, maximum, message):
    with pytest.raises(ValueError, match=message):
        build_scenes(
            "Narration", 2, max_clip_duration=maximum, min_scene_duration=minimum
        )


def test_atomic_publication_failure_preserves_existing_manifest(tmp_path, monkeypatch):
    target = tmp_path / "scenes.json"
    target.write_text('{"existing": true}', encoding="utf-8")

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr("app.services.scene_timeline.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        create_scene_timeline(str(tmp_path), "New narration.", 2)

    assert target.read_text(encoding="utf-8") == '{"existing": true}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_azure_timeline_adds_leading_internal_and_trailing_holds(tmp_path):
    first = "Coffee begins on green farms where coffee plants grow under the warm sun"
    second = "Workers carefully pick the ripe red coffee cherries by hand"
    later = "The beans are washed dried roasted ground brewed and finally served"
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,100 --> 00:00:04,250\n"
        f"{first}\n\n"
        "2\n00:00:05,275 --> 00:00:08,588\n"
        f"{second}\n\n"
        "3\n00:00:10,000 --> 00:00:30,750\n"
        f"{later}\n",
    )
    scenes = build_scenes(
        f"{first}. {second}. {later}.", 31.8, subtitle, max_clip_duration=3
    )

    assert_valid_timeline(scenes, 31.8)
    holds = [scene for scene in scenes if not scene.text]
    assert any(
        scene.start_time == pytest.approx(0) and scene.end_time == pytest.approx(0.1)
        for scene in holds
    )
    assert any(
        scene.start_time == pytest.approx(4.25)
        and scene.end_time == pytest.approx(5.275)
        for scene in holds
    )
    assert any(
        scene.start_time == pytest.approx(30.75)
        and scene.end_time == pytest.approx(31.8)
        for scene in holds
    )
    assert all(scene.duration <= 3 for scene in scenes)
    assert " ".join(scene.text for scene in scenes if scene.text) == (
        f"{first} {second} {later}"
    )


def test_long_silent_interval_is_split_and_tiny_gap_is_snapped(tmp_path):
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n"
        "2\n00:00:07,500 --> 00:00:08,000\nSecond\n",
    )
    scenes = build_scenes("First. Second.", 8, subtitle, max_clip_duration=2)
    holds = [scene for scene in scenes if not scene.text]
    assert len(holds) == 4
    assert all(scene.duration <= 2 for scene in holds)
    assert_valid_timeline(scenes, 8)

    tiny_subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n"
        "2\n00:00:01,0005 --> 00:00:02,000\nSecond\n",
    )
    tiny = build_scenes("First. Second.", 2, tiny_subtitle, max_clip_duration=3)
    assert [scene.text for scene in tiny] == ["First", "Second"]
    assert tiny[1].start_time == tiny[0].end_time
    assert_valid_timeline(tiny, 2)


def test_hold_timeline_passes_strict_render_plan_validation(tmp_path):
    from app.services import scene_render_plan

    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,100 --> 00:00:01,000\nFirst\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSecond\n",
    )
    timeline_path = create_scene_timeline(
        str(tmp_path), "First. Second.", 4, subtitle, max_clip_duration=3
    )
    timeline = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    selections = []
    meaningful = [item["index"] for item in timeline if item["text"]]
    for item in timeline:
        index = item["index"]
        if not item["text"]:
            nearest = min(meaningful, key=lambda value: (abs(value - index), value))
            selections.append(
                {
                    "scene_index": index,
                    "status": "hold_no_search",
                    "reuse_scene_index": nearest,
                }
            )
            continue
        candidate_id = f"pexels:{index}"
        selections.append(
            {
                "scene_index": index,
                "status": "provider_rank_selected",
                "reuse_scene_index": None,
                "selected_candidate_id": candidate_id,
                "selected_candidate": {
                    "candidate_id": candidate_id,
                    "provider": "pexels",
                    "provider_video_id": str(index),
                    "video_url": f"https://cdn.example/{index}.mp4",
                },
                "candidates": [{"candidate_id": candidate_id}],
            }
        )
    selection_path = tmp_path / "scene_selections.json"
    selection_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_candidate_manifest": {"version": 1, "sha256": "a" * 64},
                "source_preview_manifest": {"version": 1, "sha256": "b" * 64},
                "scenes": selections,
            }
        ),
        encoding="utf-8",
    )
    plan_path = scene_render_plan.create_scene_render_plan(
        timeline_path, str(selection_path)
    )
    plan = scene_render_plan.load_scene_render_plan(
        plan_path, timeline_path, str(selection_path)
    )
    assert len(plan["scenes"]) == len(timeline)
    assert [item["scene_index"] for item in plan["scenes"]] == list(
        range(1, len(timeline) + 1)
    )


def _run_material_pipeline(
    match_enabled,
    timeline_mock,
    candidate_mock=None,
    video_source="pexels",
    min_scene_duration=None,
):
    from unittest.mock import Mock, patch

    from app.models.schema import VideoParams
    from app.services import task

    params = VideoParams(
        video_subject="Scenes",
        match_materials_to_script=match_enabled,
        video_clip_duration=6,
        video_scene_min_duration=min_scene_duration,
        video_source=video_source,
    )
    with (
        patch.object(task, "generate_script", return_value="First. Second."),
        patch.object(task, "generate_terms", return_value=["scene"]),
        patch.object(task, "save_script_data"),
        patch.object(task, "generate_audio", return_value=("audio.mp3", 8, object())),
        patch.object(task, "generate_subtitle", return_value="subtitle.srt"),
        patch.object(task, "get_video_materials", return_value=["clip.mp4"]),
        patch.object(task.scene_timeline, "create_scene_timeline", timeline_mock),
        patch.object(
            task.scene_candidate,
            "retrieve_scene_candidates",
            candidate_mock or Mock(return_value=""),
        ),
        patch.object(task.sm.state, "update_task"),
    ):
        return task.start("timeline-pipeline", params, stop_at="materials")


def test_pipeline_creates_timeline_before_materials_when_matching_enabled(tmp_path):
    from unittest.mock import Mock

    timeline_path = str(tmp_path / "scenes.json")
    create_timeline = Mock(return_value=timeline_path)
    candidate_path = str(tmp_path / "scene_candidates.json")
    retrieve_candidates = Mock(return_value=candidate_path)
    Path(timeline_path).write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start_time": 0,
                    "end_time": 8,
                    "duration": 8,
                    "text": "First",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = _run_material_pipeline(True, create_timeline, retrieve_candidates)

    assert result == {
        "materials": ["clip.mp4"],
        "scene_timeline_path": timeline_path,
        "scene_candidates_path": candidate_path,
    }
    retrieve_candidates.assert_called_once()
    create_timeline.assert_called_once()
    assert create_timeline.call_args.kwargs == {
        "task_dir": str(Path(create_timeline.call_args.kwargs["task_dir"])),
        "narration": "First. Second.",
        "audio_duration": 8,
        "subtitle_path": "subtitle.srt",
        "max_clip_duration": 6,
        "min_scene_duration": None,
    }


def test_pipeline_passes_active_minimum_only_to_timeline(tmp_path):
    from unittest.mock import Mock

    timeline_path = tmp_path / "scenes.json"
    timeline_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start_time": 0,
                    "end_time": 8,
                    "duration": 8,
                    "text": "First",
                }
            ]
        ),
        encoding="utf-8",
    )
    create_timeline = Mock(return_value=str(timeline_path))
    retrieve_candidates = Mock(return_value="")

    _run_material_pipeline(
        True,
        create_timeline,
        retrieve_candidates,
        min_scene_duration=2.0,
    )

    assert create_timeline.call_args.kwargs["min_scene_duration"] == 2.0
    assert retrieve_candidates.call_args.kwargs["minimum_duration"] == 6
    assert "min_scene_duration" not in retrieve_candidates.call_args.kwargs


def test_pipeline_does_not_create_or_expose_timeline_when_matching_disabled():
    from unittest.mock import Mock

    create_timeline = Mock(return_value="must-not-be-used.json")
    retrieve_candidates = Mock(return_value="must-not-be-used-candidates.json")
    result = _run_material_pipeline(False, create_timeline, retrieve_candidates)

    assert result == {"materials": ["clip.mp4"]}
    create_timeline.assert_not_called()
    retrieve_candidates.assert_not_called()


def test_unsupported_source_skips_candidate_generation():
    from unittest.mock import Mock

    create_timeline = Mock(return_value="scenes.json")
    retrieve_candidates = Mock(return_value="must-not-be-used.json")
    result = _run_material_pipeline(
        True, create_timeline, retrieve_candidates, video_source="coverr"
    )

    assert result == {"materials": ["clip.mp4"], "scene_timeline_path": "scenes.json"}
    retrieve_candidates.assert_not_called()


def test_candidate_failure_keeps_existing_material_path(tmp_path):
    from unittest.mock import Mock

    timeline_path = tmp_path / "scenes.json"
    timeline_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start_time": 0,
                    "end_time": 2,
                    "duration": 2,
                    "text": "Scene",
                }
            ]
        ),
        encoding="utf-8",
    )
    retrieve_candidates = Mock(side_effect=RuntimeError("candidate failure"))
    result = _run_material_pipeline(
        True, Mock(return_value=str(timeline_path)), retrieve_candidates
    )

    assert result == {
        "materials": ["clip.mp4"],
        "scene_timeline_path": str(timeline_path),
    }
    retrieve_candidates.assert_called_once()
