import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import scene_render_materials as service
from app.services import scene_render_plan


class TestSceneRenderMaterials(unittest.TestCase):
    def _sources(self, root: Path):
        scenes = [
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 2.0,
                "duration": 2.0,
                "text": "one",
            },
            {
                "index": 2,
                "start_time": 2.0,
                "end_time": 3.0,
                "duration": 1.0,
                "text": "",
            },
            {
                "index": 3,
                "start_time": 3.0,
                "end_time": 5.0,
                "duration": 2.0,
                "text": "three",
            },
        ]

        def selected(index, url):
            candidate_id = f"pexels:{index}"
            return {
                "scene_index": index,
                "status": "provider_rank_selected",
                "reuse_scene_index": None,
                "selected_candidate_id": candidate_id,
                "selected_candidate": {
                    "candidate_id": candidate_id,
                    "provider": "pexels",
                    "provider_video_id": str(index),
                    "video_url": url,
                },
                "candidates": [{"candidate_id": candidate_id}],
            }

        selection = {
            "version": 1,
            "source_candidate_manifest": {"version": 1, "sha256": "a" * 64},
            "source_preview_manifest": {"version": 1, "sha256": "b" * 64},
            "scenes": [
                selected(1, "https://cdn.example/one.mp4?token=secret"),
                {"scene_index": 2, "status": "hold_no_search", "reuse_scene_index": 1},
                selected(3, "https://cdn.example/three.mp4?token=other"),
            ],
        }
        scene_path = root / "scenes.json"
        selection_path = root / "scene_selections.json"
        scene_path.write_text(json.dumps(scenes), encoding="utf-8")
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        plan_path = scene_render_plan.create_scene_render_plan(
            str(scene_path), str(selection_path)
        )
        return scene_path, selection_path, Path(plan_path)

    @staticmethod
    def _metadata(path: Path):
        return {
            "local_path": str(path.resolve()),
            "content_sha256": hashlib.sha256(str(path).encode()).hexdigest(),
            "size_bytes": 10,
            "duration": 5.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
        }

    def test_manifest_deduplicates_reuse_and_does_not_store_signed_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy = legacy_root / "legacy.mp4"
            legacy.write_bytes(b"legacy")

            def download(_url, destination, _root):
                destination.write_bytes(b"selected")
                return self._metadata(destination)

            with (
                patch.object(service, "_download", side_effect=download) as acquire,
                patch.object(
                    service,
                    "_probe",
                    side_effect=lambda path, _root, selected: self._metadata(
                        Path(path)
                    ),
                ),
            ):
                target = service.create_scene_render_materials(
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [str(legacy)],
                    str(legacy_root),
                    str(root),
                )
            payload = json.loads(Path(target).read_text(encoding="utf-8"))
            self.assertEqual(acquire.call_count, 2)
            self.assertNotIn("token=secret", Path(target).read_text(encoding="utf-8"))
            self.assertNotIn("local_path", payload["scenes"][0])
            self.assertEqual(payload["scenes"][1]["resolution"], "reused")
            self.assertEqual(
                payload["scenes"][0]["material_id"],
                payload["scenes"][1]["material_id"],
            )
            self.assertTrue(payload["materials"][0]["source_url_sha256"].startswith(""))

    def test_failed_selected_uses_prior_then_legacy_without_claiming_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy = legacy_root / "legacy.mp4"
            legacy.write_bytes(b"legacy")
            calls = 0

            def download(_url, destination, _root):
                nonlocal calls
                calls += 1
                if calls == 1:
                    destination.write_bytes(b"selected")
                    return self._metadata(destination)
                raise service.AcquisitionError("download_failed")

            with (
                patch.object(service, "_download", side_effect=download),
                patch.object(
                    service,
                    "_probe",
                    side_effect=lambda path, _root, selected: self._metadata(
                        Path(path)
                    ),
                ),
            ):
                target = service.create_scene_render_materials(
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [str(legacy)],
                    str(legacy_root),
                    str(root),
                )
            row = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][2]
            self.assertEqual(row["requested_binding"], "selected")
            self.assertEqual(row["resolution"], "fallback_previous_selected")
            self.assertEqual(row["resolved_visual_source_scene_index"], 1)
            self.assertEqual(row["acquisition_error"], "download_failed")
            self.assertIsNone(row["fallback_reason"])

    def test_limits_abort_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            with (
                patch.object(service, "MAX_UNIQUE_SELECTED_DOWNLOADS", 1),
                self.assertRaises(service.AcquisitionLimitError),
            ):
                service.create_scene_render_materials(
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [],
                    str(legacy_root),
                    str(root),
                )
            self.assertFalse((root / "scene_render_materials.json").exists())

    def test_manual_redirect_validates_before_each_connection(self):
        redirect = MagicMock(
            is_redirect=True,
            is_permanent_redirect=False,
            headers={"Location": "https://other.example/final.mp4"},
        )
        final = MagicMock(is_redirect=False, is_permanent_redirect=False, headers={})
        final.raise_for_status.return_value = None
        with (
            patch.object(service, "_validate_destination") as validate,
            patch.object(service.requests, "get", side_effect=[redirect, final]) as get,
        ):
            self.assertIs(service._response_for("https://cdn.example/start.mp4"), final)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(get.call_count, 2)
        self.assertTrue(
            all(call.kwargs["allow_redirects"] is False for call in get.mock_calls)
        )

    def test_destination_rejects_any_non_global_dns_answer(self):
        answers = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        with patch.object(service.socket, "getaddrinfo", return_value=answers):
            with self.assertRaises(service.AcquisitionError):
                service._validate_destination("https://example.com/video.mp4")

    def test_legacy_path_outside_trusted_root_cannot_resolve_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"outside")
            with (
                patch.object(
                    service,
                    "_download",
                    side_effect=service.AcquisitionError("download_failed"),
                ),
                self.assertRaises(ValueError),
            ):
                service.create_scene_render_materials(
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [str(outside)],
                    str(legacy_root),
                    str(root),
                )
            self.assertFalse((root / "scene_render_materials.json").exists())


if __name__ == "__main__":
    unittest.main()
