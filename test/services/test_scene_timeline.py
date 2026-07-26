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
    for previous, scene in zip([None, *scenes], scenes):
        assert 0 <= scene.start_time < scene.end_time <= duration
        assert scene.duration == pytest.approx(scene.end_time - scene.start_time)
        assert all(
            math.isfinite(value)
            for value in (scene.start_time, scene.end_time, scene.duration)
        )
        if previous is not None:
            assert previous.end_time <= scene.start_time


def test_valid_srt_timestamps_and_multiline_text(tmp_path):
    subtitle = write_srt(
        tmp_path,
        "1\n00:00:00,250 --> 00:00:02,000\nFirst line\nsecond line\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nLast scene\n",
    )

    scenes = build_scenes("First line second line. Last scene.", 5, subtitle, 10)

    assert [(scene.start_time, scene.end_time, scene.text) for scene in scenes] == [
        (0.25, 2.0, "First line second line"),
        (2.5, 4.0, "Last scene"),
    ]
    assert_valid_timeline(scenes, 5)


def test_srt_without_trailing_blank_line_is_included(tmp_path):
    subtitle = write_srt(
        tmp_path, "1\n00:00:00,000 --> 00:00:01,000\nNo trailing blank"
    )
    scenes = build_scenes("No trailing blank", 2, subtitle)
    assert len(scenes) == 1
    assert scenes[0].text == "No trailing blank"


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


def _run_material_pipeline(match_enabled, timeline_mock):
    from unittest.mock import patch

    from app.models.schema import VideoParams
    from app.services import task

    params = VideoParams(
        video_subject="Scenes",
        match_materials_to_script=match_enabled,
        video_clip_duration=6,
    )
    with (
        patch.object(task, "generate_script", return_value="First. Second."),
        patch.object(task, "generate_terms", return_value=["scene"]),
        patch.object(task, "save_script_data"),
        patch.object(task, "generate_audio", return_value=("audio.mp3", 8, object())),
        patch.object(task, "generate_subtitle", return_value="subtitle.srt"),
        patch.object(task, "get_video_materials", return_value=["clip.mp4"]),
        patch.object(task.scene_timeline, "create_scene_timeline", timeline_mock),
        patch.object(task.sm.state, "update_task"),
    ):
        return task.start("timeline-pipeline", params, stop_at="materials")


def test_pipeline_creates_timeline_before_materials_when_matching_enabled(tmp_path):
    from unittest.mock import Mock

    timeline_path = str(tmp_path / "scenes.json")
    create_timeline = Mock(return_value=timeline_path)
    result = _run_material_pipeline(True, create_timeline)

    assert result == {
        "materials": ["clip.mp4"],
        "scene_timeline_path": timeline_path,
    }
    create_timeline.assert_called_once()
    assert create_timeline.call_args.kwargs == {
        "task_dir": str(Path(create_timeline.call_args.kwargs["task_dir"])),
        "narration": "First. Second.",
        "audio_duration": 8,
        "subtitle_path": "subtitle.srt",
        "max_clip_duration": 6,
    }


def test_pipeline_does_not_create_or_expose_timeline_when_matching_disabled():
    from unittest.mock import Mock

    create_timeline = Mock(return_value="must-not-be-used.json")
    result = _run_material_pipeline(False, create_timeline)

    assert result == {"materials": ["clip.mp4"]}
    create_timeline.assert_not_called()
