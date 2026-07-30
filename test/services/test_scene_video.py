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
        {"material_id": f"m{i}", "local_path": item} for i, item in enumerate(paths)
    ]
    scenes = []
    cursor = 0.0
    for index, duration in enumerate(durations):
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
    def test_rejects_out_of_order_without_opening_materials(self):
        payload = _payload()
        payload["scenes"].reverse()
        with (
            patch.object(video, "AudioFileClip", return_value=_Audio(1.0)),
            patch.object(video, "_open_video_clip_quietly") as opened,
            self.assertRaisesRegex(ValueError, "zero-based"),
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
            self.assertEqual(concat.call_args.kwargs["max_duration"], 1.0)
            self.assertEqual(
                concat.call_args.kwargs["clip_files"], [path for path, _ in written]
            )
            self.assertFalse(any(Path(path).exists() for path, _ in written))

    def test_all_transitions_cap_effect_to_subsecond_scene(self):
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
                self.assertEqual(durations, [(mode, 0.4)])

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

    def test_failure_removes_only_attempt_files_and_partial_output(self):
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
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).glob("scene-*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
