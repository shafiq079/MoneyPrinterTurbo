import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import ValidationError

from app.models.schema import VideoAspect, VideoParams


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")


class TestVideoSceneMinimum(unittest.TestCase):
    def test_api_default_and_explicit_null_are_none(self):
        self.assertIsNone(VideoParams(video_subject="test").video_scene_min_duration)
        self.assertIsNone(
            VideoParams(
                video_subject="test", video_scene_min_duration=None
            ).video_scene_min_duration
        )

    def test_active_valid_range_is_accepted(self):
        params = VideoParams(
            video_subject="test",
            match_materials_to_script=True,
            video_clip_duration=3,
            video_scene_min_duration=2,
        )
        self.assertEqual(params.video_scene_min_duration, 2.0)

    def test_inactive_minimum_does_not_validate_against_maximum(self):
        params = VideoParams(
            video_subject="test",
            match_materials_to_script=False,
            video_clip_duration=None,
            video_scene_min_duration=99,
        )
        self.assertEqual(params.video_scene_min_duration, 99.0)

    def test_active_minimum_rejects_invalid_values_and_maximum(self):
        cases = [
            {"video_scene_min_duration": True, "video_clip_duration": 3},
            {"video_scene_min_duration": 0, "video_clip_duration": 3},
            {"video_scene_min_duration": float("inf"), "video_clip_duration": 3},
            {"video_scene_min_duration": 2, "video_clip_duration": None},
            {"video_scene_min_duration": 4, "video_clip_duration": 3},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                VideoParams(
                    video_subject="test", match_materials_to_script=True, **values
                )


if __name__ == "__main__":
    unittest.main()
