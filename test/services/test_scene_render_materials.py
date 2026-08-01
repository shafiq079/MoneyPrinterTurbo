import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

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
            scene_path, selection_path, _plan_path = self._sources(root)
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["scenes"][2]["selected_candidate"].update(
                video_url="https://cdn.example/one.mp4?token=other",
                provider_video_id="1",
            )
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            plan_path = scene_render_plan.create_scene_render_plan(
                str(scene_path), str(selection_path)
            )
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
            serialized = Path(target).read_text(encoding="utf-8")
            self.assertEqual(acquire.call_count, 2)
            self.assertNotIn(
                "https://cdn.example/one.mp4?token=secret", serialized
            )
            self.assertNotIn(
                "https://cdn.example/one.mp4?token=other", serialized
            )
            self.assertNotIn("local_path", payload["scenes"][0])
            self.assertEqual(payload["scenes"][1]["resolution"], "reused")
            self.assertEqual(
                payload["scenes"][0]["material_id"],
                payload["scenes"][1]["material_id"],
            )
            self.assertEqual(
                payload["materials"][0]["source_url_sha256"],
                hashlib.sha256(
                    b"https://cdn.example/one.mp4?token=secret"
                ).hexdigest(),
            )

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

    def test_empty_legacy_root_is_not_required_or_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            missing_legacy_root = root / "never-created-legacy"

            def download(_url, destination, _root):
                destination.write_bytes(b"selected")
                return self._metadata(destination)

            with (
                patch.object(service, "_download", side_effect=download),
                patch.object(
                    service,
                    "_probe",
                    side_effect=lambda path, _root, selected: self._metadata(Path(path)),
                ),
            ):
                target = service.create_scene_render_materials(
                    str(plan_path), str(scene_path), str(selection_path), [],
                    str(missing_legacy_root), str(root),
                )
                payload = service.load_scene_render_materials(
                    target, str(plan_path), str(scene_path), str(selection_path), [],
                    str(missing_legacy_root), str(root),
                )
            self.assertFalse(missing_legacy_root.exists())
            self.assertTrue(payload["materials"])
            self.assertTrue(
                all(row["origin"] == "selected_url" for row in payload["materials"])
            )

            manifest = Path(target)
            tampered = json.loads(manifest.read_text(encoding="utf-8"))
            tampered["materials"][0]["origin"] = "legacy_fallback"
            manifest.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy material was not supplied"):
                service.load_scene_render_materials(
                    target, str(plan_path), str(scene_path), str(selection_path), [],
                    str(missing_legacy_root), str(root),
                )
            self.assertFalse(missing_legacy_root.exists())

    def test_no_selected_coverage_has_narrow_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            missing_legacy_root = root / "missing-legacy"
            with (
                patch.object(
                    service, "_download",
                    side_effect=service.AcquisitionError("download_failed"),
                ),
                self.assertRaises(service.NoSelectedSceneCoverageError),
            ):
                service.create_scene_render_materials(
                    str(plan_path), str(scene_path), str(selection_path), [],
                    str(missing_legacy_root), str(root),
                )
            self.assertFalse(missing_legacy_root.exists())
            self.assertFalse((root / "scene_render_materials.json").exists())

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

    def test_aggregate_counts_cache_hits_and_query_distinct_selected_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            calls = []

            def cached_download(url, destination, _root):
                calls.append(url)
                destination.write_bytes(b"cached")
                return self._metadata(destination)

            with (
                patch.object(service, "_download", side_effect=cached_download),
                patch.object(service, "MAX_TOTAL_SELECTED_VIDEO_BYTES", 15),
                self.assertRaises(service.AcquisitionLimitError),
            ):
                service.create_scene_render_materials(
                    str(plan_path), str(scene_path), str(selection_path), [],
                    str(legacy_root), str(root)
                )
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0].split("?")[1], calls[1].split("?")[1])
            self.assertFalse((root / "scene_render_materials.json").exists())

    def test_same_url_with_inconsistent_provider_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, _plan_path = self._sources(root)
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["scenes"][2]["selected_candidate"]["video_url"] = (
                selection["scenes"][0]["selected_candidate"]["video_url"]
            )
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            plan_path = scene_render_plan.create_scene_render_plan(
                str(scene_path), str(selection_path)
            )
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            with self.assertRaises(ValueError):
                service.create_scene_render_materials(
                    plan_path, str(scene_path), str(selection_path), [],
                    str(legacy_root), str(root)
                )

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

    def test_url_rejects_c0_del_and_unicode_whitespace_but_allows_encoding(self):
        invalid = ("\x00", "\x1f", "\x7f", "\u00a0", "\u2003")
        for character in invalid:
            with self.subTest(character=repr(character)), self.assertRaises(
                service.AcquisitionError
            ):
                service._validate_url(f"https://example.com/a{character}b.mp4")
        self.assertEqual(
            service._validate_url("https://example.com/a%20b.mp4"),
            ("example.com", 443),
        )

    def test_response_boundary_requires_status_and_handles_connection_error(self):
        failed = MagicMock(
            is_redirect=False, is_permanent_redirect=False, headers={}
        )
        failed.raise_for_status.side_effect = requests.HTTPError("bad")
        with (
            patch.object(service, "_validate_destination"),
            patch.object(service.requests, "get", return_value=failed) as get,
            self.assertRaises(service.AcquisitionError),
        ):
            service._response_for("https://example.com/video.mp4")
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        failed.close.assert_called_once()

        with (
            patch.object(service, "_validate_destination"),
            patch.object(
                service.requests,
                "get",
                side_effect=requests.ConnectionError("offline"),
            ),
            self.assertRaises(service.AcquisitionError),
        ):
            service._response_for("https://example.com/video.mp4")

    def test_redirect_missing_location_loop_limit_and_unsafe_target(self):
        missing = MagicMock(
            is_redirect=True, is_permanent_redirect=False, headers={}
        )
        with (
            patch.object(service, "_validate_destination"),
            patch.object(service.requests, "get", return_value=missing),
            self.assertRaises(service.AcquisitionError),
        ):
            service._response_for("https://example.com/start")

        loop = MagicMock(
            is_redirect=True,
            is_permanent_redirect=False,
            headers={"Location": "https://example.com/start"},
        )
        with (
            patch.object(service, "_validate_destination"),
            patch.object(service.requests, "get", return_value=loop),
            self.assertRaises(service.AcquisitionError) as caught,
        ):
            service._response_for("https://example.com/start")
        self.assertEqual(caught.exception.code, "redirect_limit_exceeded")

        redirects = [
            MagicMock(
                is_redirect=True,
                is_permanent_redirect=False,
                headers={"Location": f"https://example.com/{index + 1}"},
            )
            for index in range(service.MAX_REDIRECTS + 1)
        ]
        with (
            patch.object(service, "_validate_destination"),
            patch.object(service.requests, "get", side_effect=redirects),
            self.assertRaises(service.AcquisitionError) as caught,
        ):
            service._response_for("https://example.com/0")
        self.assertEqual(caught.exception.code, "redirect_limit_exceeded")

        unsafe = MagicMock(
            is_redirect=True,
            is_permanent_redirect=False,
            headers={"Location": "http://127.0.0.1/private"},
        )
        with (
            patch.object(
                service,
                "_validate_destination",
                side_effect=[None, service.AcquisitionError("unsafe_destination")],
            ),
            patch.object(service.requests, "get", return_value=unsafe) as get,
            self.assertRaises(service.AcquisitionError),
        ):
            service._response_for("https://example.com/start")
        get.assert_called_once()

    def test_download_streams_chunks_without_content_and_cleans_failures(self):
        class Response:
            headers = {}

            def __init__(self, chunks):
                self.chunks = chunks
                self.closed = False

            @property
            def content(self):
                raise AssertionError("response.content must not be used")

            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                yield from self.chunks

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "selected.mp4"
            response = Response([b"ab", b"", b"cd"])

            def probe(path, _root, selected):
                self.assertTrue(selected)
                return self._metadata(Path(path))

            with (
                patch.object(service, "_response_for", return_value=response),
                patch.object(service, "_probe", side_effect=probe),
            ):
                service._download("https://example.com/a.mp4", destination, root)
            self.assertEqual(destination.read_bytes(), b"abcd")
            self.assertEqual(response.chunk_size, service.DOWNLOAD_CHUNK_BYTES)
            self.assertTrue(response.closed)

            empty = Response([])
            with (
                patch.object(service, "_response_for", return_value=empty),
                self.assertRaises(service.AcquisitionError),
            ):
                service._download("https://example.com/empty.mp4", root / "empty.mp4", root)
            self.assertEqual(list(root.glob("*.part")), [])

            broken = Response([b"one"])
            broken.iter_content = MagicMock(side_effect=requests.ReadTimeout("slow"))
            with (
                patch.object(service, "_response_for", return_value=broken),
                self.assertRaises(service.AcquisitionError) as caught,
            ):
                service._download("https://example.com/broken.mp4", root / "broken.mp4", root)
            self.assertEqual(caught.exception.code, "download_failed")
            self.assertEqual(list(root.glob("*.part")), [])

    def test_download_limits_replace_failure_and_cache_boundaries(self):
        response = MagicMock(headers={"Content-Length": str(service.MAX_SELECTED_VIDEO_BYTES + 1)})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(service, "_response_for", return_value=response),
                self.assertRaises(service.AcquisitionLimitError),
            ):
                service._download("https://example.com/large.mp4", root / "large.mp4", root)
            self.assertEqual(list(root.glob("*.part")), [])

            streamed = MagicMock(headers={})
            streamed.iter_content.return_value = [
                b"x" * 6,
                b"y" * 6,
            ]
            with (
                patch.object(service, "MAX_SELECTED_VIDEO_BYTES", 10),
                patch.object(service, "_response_for", return_value=streamed),
                self.assertRaises(service.AcquisitionLimitError),
            ):
                service._download("https://example.com/large.mp4", root / "stream.mp4", root)
            self.assertEqual(list(root.glob("*.part")), [])

            valid = MagicMock(headers={})
            valid.iter_content.return_value = [b"video"]
            with (
                patch.object(service, "_response_for", return_value=valid),
                patch.object(service, "_probe", side_effect=lambda path, *_a, **_k: self._metadata(Path(path))),
                patch.object(service.os, "replace", side_effect=OSError("disk")),
                self.assertRaises(OSError),
            ):
                service._download("https://example.com/a.mp4", root / "replace.mp4", root)
            self.assertEqual(list(root.glob("*.part")), [])

            target = root / "target.mp4"
            target.write_bytes(b"x")
            link = root / "link.mp4"
            link.symlink_to(target)
            with self.assertRaises(service.AcquisitionError):
                service._download("https://example.com/a.mp4", link, root)

            directory = root / "directory.mp4"
            directory.mkdir()
            with self.assertRaises(service.AcquisitionError):
                service._download("https://example.com/a.mp4", directory, root)

    def test_invalid_regular_cache_is_replaced_and_valid_cache_is_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "cached.mp4"
            destination.write_bytes(b"bad")
            response = MagicMock(headers={})
            response.iter_content.return_value = [b"fresh"]
            probes = 0

            def probe(path, *_args, **_kwargs):
                nonlocal probes
                probes += 1
                if probes == 1:
                    raise ValueError("invalid cache")
                return self._metadata(Path(path))

            with (
                patch.object(service, "_response_for", return_value=response),
                patch.object(service, "_probe", side_effect=probe),
            ):
                service._download("https://example.com/a.mp4", destination, root)
            self.assertEqual(destination.read_bytes(), b"fresh")
            self.assertEqual(probes, 3)

            with patch.object(
                service, "_probe", return_value=self._metadata(destination)
            ) as probe_cache, patch.object(service, "_response_for") as request:
                service._download("https://example.com/a.mp4", destination, root)
            probe_cache.assert_called_once()
            request.assert_not_called()

    def test_atomic_manifest_validates_temporary_before_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "scene_render_materials.json"
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(ValueError):
                service._atomic_write(
                    target,
                    {"version": 1},
                    lambda _path: (_ for _ in ()).throw(ValueError("invalid")),
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_strict_loader_rejects_identity_binding_and_row_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy = legacy_root / "legacy.mp4"
            legacy.write_bytes(b"legacy")

            def download(_url, destination, selected_root):
                destination.write_bytes(destination.name.encode())
                return service._probe(destination, selected_root, selected=True)

            fake_clip = MagicMock(duration=5.0, fps=30.0, size=(1920, 1080))
            with (
                patch.object(service, "VideoFileClip", return_value=fake_clip),
                patch.object(service, "_download", side_effect=download),
            ):
                target = Path(
                    service.create_scene_render_materials(
                        str(plan_path),
                        str(scene_path),
                        str(selection_path),
                        [str(legacy)],
                        str(legacy_root),
                        str(root),
                    )
                )
                original = json.loads(target.read_text(encoding="utf-8"))

                mutations = {
                    "selected material_id": lambda data: data["materials"][0].update(
                        material_id="selected:" + "f" * 64
                    ),
                    "source URL hash": lambda data: data["materials"][0].update(
                        source_url_sha256="f" * 64,
                        material_id="selected:" + "f" * 64,
                    ),
                    "provider": lambda data: data["materials"][0].update(
                        provider="pixabay"
                    ),
                    "provider video ID": lambda data: data["materials"][0].update(
                        provider_video_id="wrong"
                    ),
                    "local path": lambda data: data["materials"][0].update(
                        local_path=data["materials"][1]["local_path"]
                    ),
                    "content hash": lambda data: data["materials"][0].update(
                        content_sha256="0" * 64
                    ),
                    "reused material": lambda data: data["scenes"][1].update(
                        material_id=data["scenes"][2]["material_id"]
                    ),
                    "resolved source": lambda data: data["scenes"][1].update(
                        resolved_visual_source_scene_index=3
                    ),
                    "invalid acquisition error": lambda data: data["scenes"][0].update(
                        acquisition_error="download_failed"
                    ),
                    "missing material": lambda data: data["materials"].pop(),
                    "duplicate material ID": lambda data: data["materials"].append(
                        dict(data["materials"][0])
                    ),
                    "unused material": lambda data: data["materials"].append(
                        {
                            **data["materials"][0],
                            "material_id": "selected:" + "e" * 64,
                            "source_url_sha256": "e" * 64,
                        }
                    ),
                }
                for name, mutate in mutations.items():
                    with self.subTest(name=name):
                        data = json.loads(json.dumps(original))
                        mutate(data)
                        target.write_text(json.dumps(data), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            service.load_scene_render_materials(
                                str(target),
                                str(plan_path),
                                str(scene_path),
                                str(selection_path),
                                [str(legacy)],
                                str(legacy_root),
                                str(root),
                            )
                target.write_text(json.dumps(original), encoding="utf-8")

                unrelated = legacy_root / "unrelated.mp4"
                unrelated.write_bytes(b"unrelated")
                legacy_metadata = service._probe(
                    unrelated, legacy_root, selected=False
                )
                data = json.loads(json.dumps(original))
                data["materials"].append(
                    {
                        **legacy_metadata,
                        "material_id": f"legacy:{legacy_metadata['content_sha256']}",
                        "origin": "legacy_fallback",
                        "source_url_sha256": None,
                        "provider": None,
                        "provider_video_id": None,
                    }
                )
                target.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    service.load_scene_render_materials(
                        str(target),
                        str(plan_path),
                        str(scene_path),
                        str(selection_path),
                        [str(legacy)],
                        str(legacy_root),
                        str(root),
                    )

    def test_selected_url_identity_is_bound_to_its_deterministic_cache_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy = legacy_root / "legacy.mp4"
            legacy.write_bytes(b"legacy")

            def download(_url, destination, selected_root):
                destination.write_bytes(destination.name.encode())
                return service._probe(destination, selected_root, selected=True)

            fake_clip = MagicMock(duration=5.0, fps=30.0, size=(1920, 1080))
            with (
                patch.object(service, "VideoFileClip", return_value=fake_clip),
                patch.object(service, "_download", side_effect=download),
            ):
                target = Path(
                    service.create_scene_render_materials(
                        str(plan_path),
                        str(scene_path),
                        str(selection_path),
                        [str(legacy)],
                        str(legacy_root),
                        str(root),
                    )
                )
                original = json.loads(target.read_text(encoding="utf-8"))
                service.load_scene_render_materials(
                    str(target),
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [str(legacy)],
                    str(legacy_root),
                    str(root),
                )

                file_fields = (
                    "local_path",
                    "content_sha256",
                    "size_bytes",
                    "duration",
                    "fps",
                    "width",
                    "height",
                )
                swapped = json.loads(json.dumps(original))
                for field in file_fields:
                    swapped["materials"][0][field] = swapped["materials"][1][field]
                target.write_text(json.dumps(swapped), encoding="utf-8")
                with self.assertRaises(ValueError):
                    service.load_scene_render_materials(
                        str(target),
                        str(plan_path),
                        str(scene_path),
                        str(selection_path),
                        [str(legacy)],
                        str(legacy_root),
                        str(root),
                    )

                selected_root = root / "scene_materials"
                for filename in ("unrelated.mp4", "unrelated.part"):
                    with self.subTest(filename=filename):
                        unrelated = selected_root / filename
                        unrelated.write_bytes(filename.encode())
                        metadata = service._probe(
                            unrelated, selected_root, selected=True
                        )
                        tampered = json.loads(json.dumps(original))
                        tampered["materials"][0].update(metadata)
                        target.write_text(json.dumps(tampered), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            service.load_scene_render_materials(
                                str(target),
                                str(plan_path),
                                str(scene_path),
                                str(selection_path),
                                [str(legacy)],
                                str(legacy_root),
                                str(root),
                            )

    def test_loader_rejects_changed_plan_symlink_and_outside_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path, selection_path, plan_path = self._sources(root)
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy = legacy_root / "legacy.mp4"
            legacy.write_bytes(b"legacy")

            def download(_url, destination, selected_root):
                destination.write_bytes(destination.name.encode())
                return service._probe(destination, selected_root, selected=True)

            fake_clip = MagicMock(duration=5.0, fps=30.0, size=(1920, 1080))
            with (
                patch.object(service, "VideoFileClip", return_value=fake_clip),
                patch.object(service, "_download", side_effect=download),
            ):
                target = service.create_scene_render_materials(
                    str(plan_path),
                    str(scene_path),
                    str(selection_path),
                    [str(legacy)],
                    str(legacy_root),
                    str(root),
                )
            plan_path.write_text(
                plan_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with patch.object(service, "VideoFileClip", return_value=fake_clip):
                with self.assertRaises(ValueError):
                    service.load_scene_render_materials(
                        target,
                        str(plan_path),
                        str(scene_path),
                        str(selection_path),
                        [str(legacy)],
                        str(legacy_root),
                        str(root),
                    )

            trusted = root / "trusted"
            trusted.mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"outside")
            link = trusted / "link.mp4"
            link.symlink_to(outside)
            for candidate in (outside, link):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    service._contained_file(str(candidate), trusted.resolve(), "material")

    def test_loader_rejects_non_nearest_previous_and_next_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = [
                {"index": index, "start_time": index - 1.0, "end_time": float(index),
                 "duration": 1.0, "text": f"scene {index}"}
                for index in range(1, 5)
            ]

            def selected(index):
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
                        "video_url": f"https://cdn.example/{index}.mp4",
                    },
                    "candidates": [{"candidate_id": candidate_id}],
                }

            selection = {
                "version": 1,
                "source_candidate_manifest": {"version": 1, "sha256": "a" * 64},
                "source_preview_manifest": {"version": 1, "sha256": "b" * 64},
                "scenes": [
                    {"scene_index": 1, "status": "no_candidates", "reuse_scene_index": None},
                    selected(2),
                    selected(3),
                    {"scene_index": 4, "status": "no_candidates", "reuse_scene_index": None},
                ],
            }
            scene_path = root / "scenes.json"
            selection_path = root / "scene_selections.json"
            scene_path.write_text(json.dumps(scenes), encoding="utf-8")
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            plan_path = scene_render_plan.create_scene_render_plan(
                str(scene_path), str(selection_path)
            )
            legacy_root = root / "legacy"
            legacy_root.mkdir()

            def download(_url, destination, selected_root):
                destination.write_bytes(destination.name.encode())
                return service._probe(destination, selected_root, selected=True)

            fake_clip = MagicMock(duration=5.0, fps=30.0, size=(1920, 1080))
            with (
                patch.object(service, "VideoFileClip", return_value=fake_clip),
                patch.object(service, "_download", side_effect=download),
            ):
                target = Path(
                    service.create_scene_render_materials(
                        plan_path,
                        str(scene_path),
                        str(selection_path),
                        [],
                        str(legacy_root),
                        str(root),
                    )
                )
                original = json.loads(target.read_text(encoding="utf-8"))
                cases = (
                    (0, 3, original["scenes"][2]["material_id"]),
                    (3, 2, original["scenes"][1]["material_id"]),
                )
                for row_index, source_index, material_id in cases:
                    with self.subTest(row_index=row_index):
                        tampered = json.loads(json.dumps(original))
                        tampered["scenes"][row_index].update(
                            resolved_visual_source_scene_index=source_index,
                            material_id=material_id,
                        )
                        target.write_text(json.dumps(tampered), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            service.load_scene_render_materials(
                                str(target),
                                plan_path,
                                str(scene_path),
                                str(selection_path),
                                [],
                                str(legacy_root),
                                str(root),
                            )


if __name__ == "__main__":
    unittest.main()
