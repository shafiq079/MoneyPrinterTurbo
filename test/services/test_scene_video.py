import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models.schema import VideoAspect, VideoTransitionMode
from app.services import video


class _Audio:
    def __init__(self, duration):
        self.duration = duration
        self.reader = MagicMock()


class _Clip:
    def __init__(self, duration, events, name="source", size=(1080, 1920)):
        self.duration = duration
        self.events = events
        self.name = name
        self.size = size
        self.w, self.h = size
        self.reader = MagicMock()

    def with_speed_scaled(self, factor):
        self.events.append(("speed", self.name, factor))
        return _Clip(self.duration / factor, self.events, "speed", self.size)

    def with_effects(self, effects):
        self.events.append(("loop", self.name, effects[0].duration))
        return _Clip(effects[0].duration, self.events, "loop", self.size)

    def subclipped(self, start, end):
        self.events.append(("trim", self.name, start, end))
        return _Clip(end - start, self.events, "trim", self.size)


def _payload(durations=(0.4, 0.6), paths=("a.mp4", "b.mp4")):
    materials = [
        {"material_id": f"m{i}", "local_path": item}
        for i, item in enumerate(paths, start=1)
    ]
    scenes = []
    cursor = 0.0
    for index, duration in enumerate(durations, start=1):
        scenes.append(
            {
                "scene_index": index,
                "start_time": cursor,
                "end_time": cursor + duration,
                "duration": duration,
                "material_id": f"m{index}",
            }
        )
        cursor += duration
    return {"materials": materials, "scenes": scenes}


class TestSceneVideoRenderer(unittest.TestCase):
    def test_rejects_invalid_narration_durations_and_cleans_audio(self):
        invalid_values = (0, -1, math.nan, math.inf, -math.inf, "1.0", True)
        for value in invalid_values:
            with self.subTest(value=value):
                audio = _Audio(value)
                with (
                    patch.object(video, "AudioFileClip", return_value=audio),
                    patch.object(video, "_open_video_clip_quietly") as opened,
                    self.assertRaises(ValueError),
                ):
                    video.combine_scene_videos("combined.mp4", _payload(), "audio.wav")
                audio.reader.close.assert_called_once()
                opened.assert_not_called()

    def test_rejects_out_of_order_without_opening_materials(self):
        payload = _payload()
        payload["scenes"].reverse()
        with (
            patch.object(video, "AudioFileClip", return_value=_Audio(1.0)),
            patch.object(video, "_open_video_clip_quietly") as opened,
            self.assertRaisesRegex(ValueError, "one-based"),
        ):
            video.combine_scene_videos("combined.mp4", payload, "audio.wav")
        opened.assert_not_called()

    def test_rejects_timeline_gap_and_narration_mismatch(self):
        gap = _payload()
        gap["scenes"][1]["start_time"] = 0.5
        gap["scenes"][1]["duration"] = 0.5
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            video._validate_scene_render_timeline(gap, 1.0)
        with self.assertRaisesRegex(ValueError, "narration"):
            video._validate_scene_render_timeline(_payload(), 1.2)

    def test_exact_order_speed_repeat_trim_and_cleanup(self):
        events = []
        opened = []

        def open_clip(path):
            opened.append(path)
            return _Clip(0.25 if path == "a.mp4" else 2.0, events, path)

        written = []
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "combined.mp4")
            with (
                patch.object(video, "AudioFileClip", return_value=_Audio(1.0)),
                patch.object(video, "_open_video_clip_quietly", side_effect=open_clip),
                patch.object(video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c),
                patch.object(
                    video, "_apply_video_transition", side_effect=lambda c, *_a, **_k: c
                ),
                patch.object(
                    video,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda clip, path, **_kwargs: (
                        Path(path).write_bytes(b"x"),
                        written.append((path, clip.duration)),
                    ),
                ),
                patch.object(video, "concat_video_clips_with_ffmpeg") as concat,
            ):
                result = video.combine_scene_videos(
                    output, _payload(), "audio.wav", clip_speed=2.0, output_index=3
                )
            self.assertEqual(result, output)
            self.assertEqual(opened, ["a.mp4", "b.mp4"])
            self.assertEqual([duration for _, duration in written], [0.4, 0.6])
            self.assertTrue(any(event[0] == "speed" for event in events))
            self.assertTrue(any(event[0] == "loop" for event in events))
            self.assertEqual(concat.call_args.kwargs["target_frames"], 30)
            self.assertEqual(
                concat.call_args.kwargs["clip_files"], [path for path, _ in written]
            )
            self.assertFalse(any(Path(path).exists() for path, _ in written))

    def test_all_transitions_are_skipped_for_sub_two_second_runs(self):
        for mode in VideoTransitionMode:
            with self.subTest(mode=mode):
                durations = []

                def transition(clip, transition_mode, *, duration):
                    durations.append((transition_mode, duration))
                    return clip

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(video, "AudioFileClip", return_value=_Audio(0.4)),
                    patch.object(
                        video, "_open_video_clip_quietly", return_value=_Clip(2.0, [])
                    ),
                    patch.object(
                        video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c
                    ),
                    patch.object(
                        video, "_apply_video_transition", side_effect=transition
                    ),
                    patch.object(video, "_write_videofile_with_codec_fallback"),
                    patch.object(video, "concat_video_clips_with_ffmpeg"),
                ):
                    video.combine_scene_videos(
                        os.path.join(tmp, "combined.mp4"),
                        _payload((0.4,), ("a.mp4",)),
                        "audio.wav",
                        video_transition_mode=mode,
                    )
                self.assertEqual(durations, [])

    def test_long_run_receives_one_adaptive_transition(self):
        for run_duration, expected_transition in ((2.4, 0.6), (8.0, 1.0)):
            with self.subTest(run_duration=run_duration):
                transitions = []

                def transition(clip, transition_mode, *, duration):
                    transitions.append((transition_mode, duration))
                    return clip

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        video, "AudioFileClip", return_value=_Audio(run_duration)
                    ),
                    patch.object(
                        video,
                        "_open_video_clip_quietly",
                        return_value=_Clip(run_duration, []),
                    ),
                    patch.object(
                        video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c
                    ),
                    patch.object(
                        video, "_apply_video_transition", side_effect=transition
                    ),
                    patch.object(video, "_write_videofile_with_codec_fallback"),
                    patch.object(video, "concat_video_clips_with_ffmpeg"),
                ):
                    video.combine_scene_videos(
                        os.path.join(tmp, "combined.mp4"),
                        _payload((run_duration,), ("a.mp4",)),
                        "audio.wav",
                        video_transition_mode=VideoTransitionMode.fade_in,
                    )
                self.assertEqual(
                    transitions,
                    [(VideoTransitionMode.fade_in, expected_transition)],
                )

    def test_consecutive_material_bindings_coalesce_but_later_reuse_does_not(self):
        payload = {
            "materials": [
                {"material_id": "a", "local_path": "a.mp4"},
                {"material_id": "b", "local_path": "b.mp4"},
            ],
            "scenes": [
                {
                    "scene_index": 1,
                    "start_time": 0,
                    "end_time": 0.5,
                    "duration": 0.5,
                    "material_id": "a",
                },
                {
                    "scene_index": 2,
                    "start_time": 0.5,
                    "end_time": 1.0,
                    "duration": 0.5,
                    "material_id": "a",
                },
                {
                    "scene_index": 3,
                    "start_time": 1.0,
                    "end_time": 1.5,
                    "duration": 0.5,
                    "material_id": "b",
                },
                {
                    "scene_index": 4,
                    "start_time": 1.5,
                    "end_time": 2.0,
                    "duration": 0.5,
                    "material_id": "a",
                },
            ],
        }
        scenes, materials = video._validate_scene_render_timeline(payload, 2.0)
        runs = video._build_scene_visual_runs(scenes, materials)
        self.assertEqual(
            [
                (run.material_id, run.first_scene_index, run.last_scene_index)
                for run in runs
            ],
            [("a", 1, 2), ("b", 3, 3), ("a", 4, 4)],
        )

    def test_leading_internal_and_trailing_holds_coalesce_by_material(self):
        payload = {
            "materials": [
                {"material_id": "a", "local_path": "a.mp4"},
                {"material_id": "b", "local_path": "b.mp4"},
            ],
            "scenes": [
                {
                    "scene_index": 1,
                    "start_time": 0,
                    "end_time": 0.2,
                    "duration": 0.2,
                    "material_id": "a",
                },
                {
                    "scene_index": 2,
                    "start_time": 0.2,
                    "end_time": 0.8,
                    "duration": 0.6,
                    "material_id": "a",
                },
                {
                    "scene_index": 3,
                    "start_time": 0.8,
                    "end_time": 1.0,
                    "duration": 0.2,
                    "material_id": "a",
                },
                {
                    "scene_index": 4,
                    "start_time": 1.0,
                    "end_time": 1.8,
                    "duration": 0.8,
                    "material_id": "b",
                },
                {
                    "scene_index": 5,
                    "start_time": 1.8,
                    "end_time": 2.0,
                    "duration": 0.2,
                    "material_id": "b",
                },
            ],
        }
        scenes, materials = video._validate_scene_render_timeline(payload, 2.0)
        runs = video._build_scene_visual_runs(scenes, materials)
        self.assertEqual(len(runs), 2)
        self.assertEqual((runs[0].first_scene_index, runs[0].last_scene_index), (1, 3))
        self.assertEqual((runs[1].first_scene_index, runs[1].last_scene_index), (4, 5))

    def test_adjacent_scenes_open_once_and_loop_for_complete_run(self):
        payload = {
            "materials": [{"material_id": "a", "local_path": "a.mp4"}],
            "scenes": [
                {
                    "scene_index": 1,
                    "start_time": 0,
                    "end_time": 0.6,
                    "duration": 0.6,
                    "material_id": "a",
                },
                {
                    "scene_index": 2,
                    "start_time": 0.6,
                    "end_time": 1.2,
                    "duration": 0.6,
                    "material_id": "a",
                },
            ],
        }
        events = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(video, "AudioFileClip", return_value=_Audio(1.2)),
            patch.object(
                video, "_open_video_clip_quietly", return_value=_Clip(0.5, events)
            ) as opened,
            patch.object(video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c),
            patch.object(video, "_apply_video_transition") as transition,
            patch.object(video, "_write_videofile_with_codec_fallback"),
            patch.object(video, "concat_video_clips_with_ffmpeg"),
        ):
            video.combine_scene_videos(
                os.path.join(tmp, "combined.mp4"), payload, "audio.wav"
            )
        opened.assert_called_once_with("a.mp4")
        transition.assert_not_called()
        loop_events = [event for event in events if event[0] == "loop"]
        self.assertEqual(loop_events, [("loop", "source", 1.2)])

    def test_aspect_helper_preserves_fit_and_black_background(self):
        clip = MagicMock(size=(1920, 1080), w=1920, h=1080, duration=1.0)
        resized = MagicMock()
        resized.with_position.return_value = resized
        clip.resized.return_value = resized
        background = MagicMock()
        background.with_duration.return_value = background
        composite = MagicMock()
        with (
            patch.object(video, "ColorClip", return_value=background) as color,
            patch.object(
                video, "CompositeVideoClip", return_value=composite
            ) as composite_type,
        ):
            result = video._fit_clip_to_aspect(clip, VideoAspect.portrait)
        self.assertIs(result, composite)
        color.assert_called_once_with(size=(1080, 1920), color=(0, 0, 0))
        clip.resized.assert_called_once_with(new_size=(1080, 607))
        composite_type.assert_called_once_with([background, resized])

    def test_encoding_failure_removes_attempt_files_but_not_preexisting_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "combined-2.mp4"
            output.write_bytes(b"partial")

            def fail_write(_clip, path, **_kwargs):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("encode failed")

            with (
                patch.object(video, "AudioFileClip", return_value=_Audio(1.0)),
                patch.object(
                    video, "_open_video_clip_quietly", return_value=_Clip(2.0, [])
                ),
                patch.object(video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c),
                patch.object(
                    video, "_apply_video_transition", side_effect=lambda c, *_a, **_k: c
                ),
                patch.object(
                    video,
                    "_write_videofile_with_codec_fallback",
                    side_effect=fail_write,
                ),
                self.assertRaises(RuntimeError),
            ):
                video.combine_scene_videos(
                    str(output), _payload(), "audio.wav", output_index=2
                )
            self.assertEqual(output.read_bytes(), b"partial")
            self.assertEqual(list(Path(tmp).glob("scene-*.mp4")), [])

    def test_concat_failure_removes_invocation_output_and_scene_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "combined.mp4"

            def fail_concat(**kwargs):
                Path(kwargs["output_file"]).write_bytes(b"partial")
                raise RuntimeError("concat failed")

            with (
                patch.object(video, "AudioFileClip", return_value=_Audio(1.0)),
                patch.object(
                    video, "_open_video_clip_quietly", return_value=_Clip(2.0, [])
                ),
                patch.object(video, "_fit_clip_to_aspect", side_effect=lambda c, _a: c),
                patch.object(
                    video, "_apply_video_transition", side_effect=lambda c, *_a, **_k: c
                ),
                patch.object(
                    video,
                    "_write_videofile_with_codec_fallback",
                    side_effect=lambda _clip, path, **_kwargs: Path(path).write_bytes(
                        b"scene"
                    ),
                ),
                patch.object(
                    video, "concat_video_clips_with_ffmpeg", side_effect=fail_concat
                ),
                self.assertRaises(RuntimeError),
            ):
                video.combine_scene_videos(str(output), _payload(), "audio.wav")
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).glob("scene-*.mp4")), [])


if __name__ == "__main__":
    unittest.main()


class TestSceneVideoRealMedia(unittest.TestCase):
    @staticmethod
    def _write_video(path, duration, frame_function):
        from moviepy import VideoClip

        clip = VideoClip(frame_function=frame_function, duration=duration)
        try:
            clip.write_videofile(
                str(path), codec="libx264", fps=30, audio=False, logger=None
            )
        finally:
            clip.close()

    @staticmethod
    def _write_audio(path, duration):
        import numpy as np
        from moviepy import AudioArrayClip

        sample_rate = 8000
        times = np.arange(int(duration * sample_rate)) / sample_rate
        samples = (0.05 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
        audio = AudioArrayClip(np.column_stack([samples, samples]), fps=sample_rate)
        try:
            audio.write_audiofile(str(path), fps=sample_rate, logger=None)
        finally:
            audio.close()

    def test_real_timeline_loop_trim_reuse_order_and_cleanup(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short = root / "short.mp4"
            long = root / "long.mp4"
            audio = root / "audio.wav"
            output = root / "combined.mp4"
            self._write_video(
                short,
                0.2,
                lambda t: np.full(
                    (48, 48, 3), (240, 0, 0) if t < 0.1 else (0, 240, 0), np.uint8
                ),
            )
            self._write_video(
                long,
                1.0,
                lambda t: np.full(
                    (48, 48, 3), (240, 240, 0) if t < 0.4 else (0, 240, 240), np.uint8
                ),
            )
            self._write_audio(audio, 1.0)
            payload = {
                "materials": [
                    {"material_id": "short", "local_path": str(short)},
                    {"material_id": "long", "local_path": str(long)},
                ],
                "scenes": [
                    {
                        "scene_index": 1,
                        "start_time": 0.0,
                        "end_time": 0.4,
                        "duration": 0.4,
                        "material_id": "short",
                    },
                    {
                        "scene_index": 2,
                        "start_time": 0.4,
                        "end_time": 0.7,
                        "duration": 0.3,
                        "material_id": "long",
                    },
                    {
                        "scene_index": 3,
                        "start_time": 0.7,
                        "end_time": 1.0,
                        "duration": 0.3,
                        "material_id": "short",
                    },
                ],
            }
            opened = []
            real_open = video._open_video_clip_quietly

            def tracked_open(path):
                opened.append(path)
                return real_open(path)

            with (
                patch.object(
                    video, "_open_video_clip_quietly", side_effect=tracked_open
                ),
                patch.object(VideoAspect, "to_resolution", return_value=(48, 48)),
            ):
                video.combine_scene_videos(str(output), payload, str(audio))

            rendered = real_open(str(output))
            try:
                self.assertLessEqual(abs(rendered.duration - 1.0), 1 / 30)
                samples = {
                    "short_start": rendered.get_frame(0.05)[24, 24],
                    "short_loop": rendered.get_frame(0.25)[24, 24],
                    "long_start": rendered.get_frame(0.5)[24, 24],
                    "reuse_start": rendered.get_frame(0.75)[24, 24],
                }
            finally:
                video.close_clip(rendered)
            self.assertGreater(samples["short_start"][0], samples["short_start"][1])
            self.assertGreater(samples["short_loop"][0], samples["short_loop"][1])
            self.assertGreater(samples["long_start"][0], samples["long_start"][2])
            self.assertGreater(samples["reuse_start"][0], samples["reuse_start"][1])
            self.assertEqual(opened.count(str(short)), 2)
            self.assertEqual(list(root.glob("scene-*.mp4")), [])
            self.assertEqual(list(root.glob("ffmpeg-concat-*.txt")), [])

    def test_real_subsecond_fade_preserves_duration(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            audio = root / "audio.wav"
            output = root / "combined.mp4"
            self._write_video(
                source, 0.5, lambda _t: np.full((32, 32, 3), (220, 20, 20), np.uint8)
            )
            self._write_audio(audio, 0.4)
            payload = {
                "materials": [{"material_id": "source", "local_path": str(source)}],
                "scenes": [
                    {
                        "scene_index": 1,
                        "start_time": 0.0,
                        "end_time": 0.4,
                        "duration": 0.4,
                        "material_id": "source",
                    }
                ],
            }
            with patch.object(VideoAspect, "to_resolution", return_value=(32, 32)):
                video.combine_scene_videos(
                    str(output),
                    payload,
                    str(audio),
                    video_transition_mode=VideoTransitionMode.fade_in,
                )
            rendered = video._open_video_clip_quietly(str(output))
            try:
                self.assertLessEqual(abs(rendered.duration - 0.4), 1 / 30)
            finally:
                video.close_clip(rendered)
            self.assertEqual(list(root.glob("scene-*.mp4")), [])
            self.assertEqual(list(root.glob("ffmpeg-concat-*.txt")), [])

    def test_many_fractional_scenes_keep_frames_and_continue_through_hold(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "moving.mp4"
            audio = root / "audio.wav"
            output = root / "combined.mp4"
            self._write_video(
                source,
                10.0,
                lambda t: np.full(
                    (24, 24, 3),
                    (220, 20, 20) if t < 5.0 else (20, 220, 20),
                    np.uint8,
                ),
            )
            self._write_audio(audio, 31.8)
            endpoints = [0.0, 0.1, 2.175, 4.25, 5.275, 8.588]
            while endpoints[-1] < 30.75:
                endpoints.append(min(30.75, endpoints[-1] + 1.037))
            endpoints.append(31.8)
            scenes = [
                {
                    "scene_index": index,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "material_id": "moving",
                }
                for index, (start, end) in enumerate(
                    zip(endpoints, endpoints[1:]), start=1
                )
            ]
            payload = {
                "materials": [{"material_id": "moving", "local_path": str(source)}],
                "scenes": scenes,
            }
            opened = []
            real_open = video._open_video_clip_quietly

            def tracked_open(path):
                opened.append(path)
                return real_open(path)

            with (
                patch.object(
                    video, "_open_video_clip_quietly", side_effect=tracked_open
                ),
                patch.object(VideoAspect, "to_resolution", return_value=(24, 24)),
            ):
                video.combine_scene_videos(
                    str(output),
                    payload,
                    str(audio),
                    video_transition_mode=VideoTransitionMode.fade_in,
                )

            rendered = real_open(str(output))
            try:
                self.assertEqual(math.ceil(31.8 * 30), 954)
                self.assertGreaterEqual(rendered.duration, 31.8)
                self.assertLessEqual(rendered.duration - 31.8, 1 / 30)
                before_hold_end = rendered.get_frame(5.24)[12, 12]
                after_hold_end = rendered.get_frame(5.31)[12, 12]
            finally:
                video.close_clip(rendered)
            self.assertGreater(before_hold_end[1], before_hold_end[0])
            self.assertGreater(after_hold_end[1], after_hold_end[0])
            self.assertGreater(int(before_hold_end.sum()), 100)
            self.assertGreater(int(after_hold_end.sum()), 100)
            self.assertEqual(opened.count(str(source)), 1)
            self.assertEqual(list(root.glob("scene-*.mp4")), [])
            self.assertEqual(list(root.glob("ffmpeg-concat-*.txt")), [])
