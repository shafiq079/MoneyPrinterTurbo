import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import scene_render_plan


class TestSceneRenderPlan(unittest.TestCase):
    def _timeline(self):
        return [
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 2.5,
                "duration": 2.5,
                "text": "First scene",
            },
            {
                "index": 2,
                "start_time": 2.5,
                "end_time": 4.0,
                "duration": 1.5,
                "text": "",
            },
            {
                "index": 3,
                "start_time": 4.0,
                "end_time": 7.0,
                "duration": 3.0,
                "text": "Last scene",
            },
        ]

    def _selected(self, index, *, url="https://cdn.example/video.mp4"):
        candidate_id = f"pexels:{index}00"
        return {
            "scene_index": index,
            "status": "provider_rank_selected",
            "reuse_scene_index": None,
            "selected_candidate_id": candidate_id,
            "selected_candidate": {
                "candidate_id": candidate_id,
                "provider": "pexels",
                "provider_video_id": f"{index}00",
                "video_url": url,
            },
            "candidates": [{"candidate_id": candidate_id}],
        }

    def _selection(self, scenes):
        digest = "a" * 64
        return {
            "version": 1,
            "source_candidate_manifest": {"version": 1, "sha256": digest},
            "source_preview_manifest": {"version": 1, "sha256": digest},
            "scenes": scenes,
        }

    def _write(self, directory, timeline=None, selection=None):
        timeline_path = Path(directory) / "scenes.json"
        selection_path = Path(directory) / "scene_selections.json"
        timeline_path.write_text(
            json.dumps(timeline if timeline is not None else self._timeline()),
            encoding="utf-8",
        )
        selection_path.write_text(
            json.dumps(selection if selection is not None else self._selection([])),
            encoding="utf-8",
        )
        return timeline_path, selection_path

    def test_selected_reused_and_source_traceability(self):
        timeline = self._timeline()
        selection = self._selection(
            [
                self._selected(1),
                {
                    "scene_index": 2,
                    "status": "hold_no_search",
                    "reuse_scene_index": 1,
                },
                self._selected(3),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            timeline_path, selection_path = self._write(tmp, timeline, selection)
            timeline_bytes = timeline_path.read_bytes()
            selection_bytes = selection_path.read_bytes()
            target = scene_render_plan.create_scene_render_plan(
                str(timeline_path), str(selection_path)
            )
            payload = json.loads(Path(target).read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 1)
        self.assertEqual(
            payload["source_scene_manifest"]["sha256"],
            hashlib.sha256(timeline_bytes).hexdigest(),
        )
        self.assertEqual(
            payload["source_selection_manifest"],
            {"version": 1, "sha256": hashlib.sha256(selection_bytes).hexdigest()},
        )
        first, hold, last = payload["scenes"]
        self.assertEqual(first["binding"], "selected")
        self.assertEqual(first["provider"], "pexels")
        self.assertEqual(first["provider_video_id"], "100")
        self.assertEqual(first["selected_candidate_id"], "pexels:100")
        self.assertEqual(first["visual_source_scene_index"], 1)
        self.assertEqual(hold["binding"], "reused")
        self.assertEqual(hold["visual_source_scene_index"], 1)
        self.assertEqual(hold["selected_candidate_id"], "pexels:100")
        self.assertEqual(hold["text"], "")
        self.assertEqual(last["text"], timeline[2]["text"])
        self.assertEqual(last["start_time"], timeline[2]["start_time"])
        self.assertEqual(last["end_time"], timeline[2]["end_time"])
        self.assertEqual(last["duration"], timeline[2]["duration"])

    def test_all_selected_statuses_and_valid_url_forms(self):
        cases = (
            (
                "provider_rank_selected",
                "https://cdn.example/video.mp4?token=a%2Bb&expires=1",
            ),
            ("provider_rank_fallback", "https://cdn.example/a%20b/video.mp4"),
            ("vlm_selected", "https://cdn.example/video.mp4"),
        )
        for status, url in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                timeline = [self._timeline()[0]]
                selected = self._selected(1, url=url)
                selected["status"] = status
                paths = self._write(tmp, timeline, self._selection([selected]))
                target = scene_render_plan.create_scene_render_plan(
                    *(str(path) for path in paths)
                )
                row = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]
                self.assertEqual(row["binding"], "selected")
                self.assertEqual(row["video_url"], url)
                self.assertIsNone(row["fallback_reason"])

    def test_unicode_narration_is_written_as_unescaped_utf8(self):
        timeline = [self._timeline()[0]]
        timeline[0]["text"] = "咖啡与音乐 — مرحبًا 🎬"
        selection = self._selection([self._selected(1)])
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write(tmp, timeline, selection)
            target = scene_render_plan.create_scene_render_plan(
                *(str(path) for path in paths)
            )
            raw = Path(target).read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["scenes"][0]["text"], timeline[0]["text"])
        self.assertIn(timeline[0]["text"].encode("utf-8"), raw)
        self.assertNotIn(b"\\u", raw)

    def test_initial_hold_reuses_next_selected_visual(self):
        timeline = self._timeline()
        timeline[0]["text"] = ""
        timeline[1]["text"] = "Middle"
        selection = self._selection(
            [
                {"scene_index": 1, "status": "hold_no_search", "reuse_scene_index": 2},
                self._selected(2),
                self._selected(3),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write(tmp, timeline, selection)
            target = scene_render_plan.create_scene_render_plan(
                *(str(p) for p in paths)
            )
            scenes = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"]
        self.assertEqual(scenes[0]["binding"], "reused")
        self.assertEqual(scenes[0]["visual_source_scene_index"], 2)

    def test_hold_skips_unusable_target_and_uses_nearest_selected(self):
        selection = self._selection(
            [
                self._selected(1),
                {"scene_index": 2, "status": "hold_no_search", "reuse_scene_index": 3},
                self._selected(3, url="relative.mp4"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write(tmp, selection=selection)
            target = scene_render_plan.create_scene_render_plan(
                *(str(p) for p in paths)
            )
            scenes = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"]
        self.assertEqual(scenes[1]["binding"], "reused")
        self.assertEqual(scenes[1]["visual_source_scene_index"], 1)
        self.assertEqual(scenes[2]["fallback_reason"], "unavailable_url")

    def test_explicit_fallbacks_preserve_complete_timeline(self):
        selection = self._selection(
            [
                {
                    "scene_index": 1,
                    "status": "no_safe_candidate",
                    "reuse_scene_index": None,
                },
                {"scene_index": 2, "status": "hold_no_search", "reuse_scene_index": 1},
                # Scene 3 is deliberately missing.
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write(tmp, selection=selection)
            target = scene_render_plan.create_scene_render_plan(
                *(str(p) for p in paths)
            )
            scenes = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0]["fallback_reason"], "no_safe_candidate")
        self.assertEqual(scenes[1]["fallback_reason"], "reuse_target_unresolved")
        self.assertEqual(scenes[2]["fallback_reason"], "missing_selection")
        for scene in scenes:
            self.assertEqual(scene["binding"], "fallback_required")
            self.assertIsNone(scene["provider"])
            self.assertIsNone(scene["provider_video_id"])
            self.assertIsNone(scene["selected_candidate_id"])
            self.assertIsNone(scene["video_url"])

    def test_no_candidates_and_unusable_urls_are_scene_fallbacks(self):
        invalid_urls = [
            None,
            "relative.mp4",
            "ftp://cdn.example/video.mp4",
            "https:///x",
            "https://example.com/a b",
            "https://exa mple.com/video.mp4",
            "https://user:password@example.com/video.mp4",
            "https://example.com/video.mp4#fragment",
            "https://example.com:70000/video.mp4",
        ]
        for invalid_url in invalid_urls:
            with self.subTest(url=invalid_url), tempfile.TemporaryDirectory() as tmp:
                timeline = [self._timeline()[0]]
                selected = self._selected(1, url=invalid_url)
                paths = self._write(tmp, timeline, self._selection([selected]))
                target = scene_render_plan.create_scene_render_plan(
                    *(str(path) for path in paths)
                )
                row = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]
                self.assertEqual(row["binding"], "fallback_required")
                self.assertEqual(row["fallback_reason"], "unavailable_url")

        with tempfile.TemporaryDirectory() as tmp:
            timeline = [self._timeline()[0]]
            row = {
                "scene_index": 1,
                "status": "no_candidates",
                "reuse_scene_index": None,
            }
            paths = self._write(tmp, timeline, self._selection([row]))
            target = scene_render_plan.create_scene_render_plan(
                *(str(path) for path in paths)
            )
            result = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]
        self.assertEqual(result["fallback_reason"], "no_candidates")

    def test_structural_ambiguity_is_rejected(self):
        cases = []
        duplicate = self._selected(1)
        cases.append(self._selection([duplicate, duplicate]))
        unknown = self._selected(4)
        cases.append(self._selection([unknown]))
        inconsistent = self._selected(1)
        inconsistent["selected_candidate"]["candidate_id"] = "pexels:different"
        cases.append(self._selection([inconsistent]))
        missing_row = self._selected(1)
        missing_row["candidates"] = [{"candidate_id": "pexels:different"}]
        cases.append(self._selection([missing_row]))
        malformed_binding = self._selection([])
        malformed_binding["source_candidate_manifest"]["sha256"] = "bad"
        cases.append(malformed_binding)
        for selection in cases:
            with (
                self.subTest(selection=selection),
                tempfile.TemporaryDirectory() as tmp,
            ):
                paths = self._write(tmp, selection=selection)
                with self.assertRaises(ValueError):
                    scene_render_plan.create_scene_render_plan(
                        *(str(path) for path in paths)
                    )
                self.assertFalse((Path(tmp) / "scene_render_plan.json").exists())

    def test_invalid_timing_is_rejected(self):
        mutations = (
            lambda scenes: scenes[1].update(index=1),
            lambda scenes: scenes[1].update(start_time=3.0, duration=1.0),
            lambda scenes: scenes[1].update(start_time=2.0, duration=2.0),
            lambda scenes: scenes[0].update(duration=1.0),
            lambda scenes: scenes[0].update(start_time=float("nan")),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                timeline = self._timeline()
                mutate(timeline)
                paths = self._write(tmp, timeline, self._selection([]))
                with self.assertRaises(ValueError):
                    scene_render_plan.create_scene_render_plan(
                        *(str(path) for path in paths)
                    )

    def test_invalid_source_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            timeline, selection = self._write(tmp)
            link = Path(tmp) / "timeline-link.json"
            link.symlink_to(timeline)
            for source in (Path(tmp) / "missing.json", link):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    scene_render_plan.create_scene_render_plan(
                        str(source), str(selection)
                    )

            timeline.write_bytes(b"\xff")
            with self.assertRaises(ValueError):
                scene_render_plan.create_scene_render_plan(
                    str(timeline), str(selection)
                )
            timeline.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                scene_render_plan.create_scene_render_plan(
                    str(timeline), str(selection)
                )
            with timeline.open("wb") as output:
                output.truncate(scene_render_plan.MAX_MANIFEST_BYTES + 1)
            with self.assertRaises(ValueError):
                scene_render_plan.create_scene_render_plan(
                    str(timeline), str(selection)
                )

    def test_atomic_failure_leaves_no_plan_or_temporary_file(self):
        timeline = [self._timeline()[0]]
        selection = self._selection([self._selected(1)])
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write(tmp, timeline, selection)
            with patch.object(
                scene_render_plan.os, "replace", side_effect=OSError("fail")
            ):
                with self.assertRaises(OSError):
                    scene_render_plan.create_scene_render_plan(
                        *(str(path) for path in paths)
                    )
            self.assertFalse((Path(tmp) / "scene_render_plan.json").exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
