import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app.services import scene_selection


def _candidate(identifier: str, rank: int) -> dict:
    return {
        "candidate_id": f"pexels:{identifier}",
        "provider": "pexels",
        "provider_video_id": identifier,
        "provider_page_url": f"https://www.pexels.com/video/{identifier}/",
        "matched_query": "city",
        "provider_rank": rank,
        "preview_url": f"https://images.pexels.com/{identifier}.jpg",
        "video_url": f"https://videos.pexels.com/{identifier}.mp4",
        "duration": 8,
        "width": 1080,
        "height": 1920,
    }


def _candidate_manifest(path: Path) -> bytes:
    payload = {
        "version": 1,
        "provider": "pexels",
        "video_aspect": "9:16",
        "candidates_per_scene": 6,
        "provider_search_budget": 20,
        "remote_searches_used": 1,
        "query_generation_warning": "query generation used fallback",
        "scenes": [
            {
                "scene_index": 1,
                "start_time": 0,
                "end_time": 2,
                "duration": 2,
                "text": "A city street",
                "queries": ["city"],
                "query_source": "fallback",
                "status": "complete",
                "warning": None,
                "reuse_scene_index": None,
                "candidates": [_candidate("later", 3), _candidate("winner", 1)],
            },
            {
                "scene_index": 2,
                "start_time": 2,
                "end_time": 3,
                "duration": 1,
                "text": "",
                "queries": [],
                "query_source": "none",
                "status": "hold_no_search",
                "warning": None,
                "reuse_scene_index": 1,
                "candidates": [],
            },
            {
                "scene_index": 3,
                "start_time": 3,
                "end_time": 4,
                "duration": 1,
                "text": "No material",
                "queries": ["city"],
                "query_source": "fallback",
                "status": "no_results_or_error",
                "warning": "provider response contained no results",
                "reuse_scene_index": None,
                "candidates": [],
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode()
    path.write_bytes(data)
    return data


def _jpeg(path: Path, color: tuple[int, int, int]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 12), color)
    image.save(path, "JPEG", quality=85, optimize=False, progressive=False, subsampling=2)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    final_path = path.parent / f"poster-jpeg-v1-{digest}.jpg"
    path.replace(final_path)
    return {
        "normalization_version": "poster-jpeg-v1",
        "image_sha256": digest,
        "media_type": "image/jpeg",
        "width": 16,
        "height": 12,
        "byte_size": len(data),
        "cache_reference": (
            "cache_candidate_previews/objects/"
            f"poster-jpeg-v1-{digest}.jpg"
        ),
    }


def _preview_manifest(path: Path, candidate_bytes: bytes, previews: list[dict]) -> bytes:
    payload = {
        "version": 1,
        "source_candidate_manifest": {
            "version": 1,
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        },
        "normalization_version": "poster-jpeg-v1",
        "provider": "pexels",
        "video_aspect": "9:16",
        "limits": {
            "candidates_per_scene": 6,
            "max_remote_downloads_per_task": 120,
            "max_download_bytes_per_image": 3145728,
            "max_aggregate_download_bytes": 134217728,
            "max_decoded_pixels": 12000000,
            "max_normalized_edge": 1280,
            "download_concurrency": 4,
            "max_redirects": 2,
            "connect_timeout_seconds": 5,
            "read_timeout_seconds": 15,
            "total_deadline_seconds": 30,
        },
        "usage": {
            "remote_downloads_started": 2,
            "remote_download_bytes": 1000,
            "cache_hits": 0,
            "in_task_reuses": 0,
        },
        "scenes": [
            {
                "scene_index": 1,
                "status": "complete",
                "reuse_scene_index": None,
                "candidates": [
                    {
                        "candidate_id": "pexels:later",
                        "status": "ready",
                        "failure_reason": None,
                        "preview": previews[0],
                    },
                    {
                        "candidate_id": "pexels:winner",
                        "status": "ready",
                        "failure_reason": None,
                        "preview": previews[1],
                    },
                ],
            },
            {
                "scene_index": 2,
                "status": "hold_no_search",
                "reuse_scene_index": 1,
                "candidates": [],
            },
            {
                "scene_index": 3,
                "status": "no_previews",
                "reuse_scene_index": None,
                "candidates": [],
            },
        ],
    }
    data = json.dumps(payload).encode()
    path.write_bytes(data)
    return data


@pytest.fixture
def artifacts(tmp_path):
    objects = tmp_path / "objects"
    first = _jpeg(objects / "first.jpg", (255, 0, 0))
    second = _jpeg(objects / "second.jpg", (0, 255, 0))
    candidates = tmp_path / "scene_candidates.json"
    candidate_bytes = _candidate_manifest(candidates)
    previews = tmp_path / "scene_previews.json"
    preview_bytes = _preview_manifest(previews, candidate_bytes, [first, second])
    return candidates, previews, objects, candidate_bytes, preview_bytes


def test_complete_provider_neutral_manifest_and_deterministic_selection(artifacts):
    candidates, previews, objects, candidate_bytes, preview_bytes = artifacts
    with patch.object(scene_selection.scene_preview_cache, "object_dir", return_value=objects):
        target = scene_selection.create_scene_selections(str(candidates), str(previews))

    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["video_aspect"] == "9:16"
    assert data["source_candidate_manifest"]["sha256"] == hashlib.sha256(candidate_bytes).hexdigest()
    assert data["source_preview_manifest"]["sha256"] == hashlib.sha256(preview_bytes).hexdigest()
    assert data["selection_policy_version"] == "provider-rank-v1"
    assert set(data["ranking"].values()) == {None}
    assert data["usage"]["vlm_requests_started"] is None
    assert [scene["status"] for scene in data["scenes"]] == [
        "provider_rank_selected",
        "hold_no_search",
        "no_candidates",
    ]
    selected = data["scenes"][0]
    assert selected["selected_candidate_id"] == "pexels:winner"
    assert selected["selected_candidate"]["provider_video_id"] == "winner"
    assert selected["selected_preview_sha256"]
    assert selected["visual_safety_evaluated"] is False
    assert selected["fallback_reason"] is None
    assert [item["local_order"] for item in selected["candidates"]] == [2, 1]
    assert all(item["assessment"] is None for item in selected["candidates"])
    assert data["scenes"][1]["reuse_scene_index"] == 1
    assert data["scenes"][1]["selected_candidate_id"] is None
    assert data["warnings"] == ["source_query_generation_warning"]


def test_missing_selected_object_is_truthful_but_does_not_block_selection(artifacts):
    candidates, previews, objects, _, _ = artifacts
    payload = json.loads(previews.read_text())
    selected = payload["scenes"][0]["candidates"][1]["preview"]
    (objects / f"poster-jpeg-v1-{selected['image_sha256']}.jpg").unlink()
    with patch.object(scene_selection.scene_preview_cache, "object_dir", return_value=objects):
        target = scene_selection.create_scene_selections(str(candidates), str(previews))
    scene = json.loads(Path(target).read_text())["scenes"][0]
    assert scene["status"] == "provider_rank_selected"
    assert scene["selected_candidate_id"] == "pexels:winner"
    assert scene["selected_preview_sha256"] is None
    assert scene["candidates"][1]["preview_status"] == "object_missing"
    assert "selected_preview_object_missing" in scene["warnings"]


def test_corrupt_nonselected_object_is_invalid_without_repair(artifacts):
    candidates, previews, objects, _, _ = artifacts
    payload = json.loads(previews.read_text())
    metadata = payload["scenes"][0]["candidates"][0]["preview"]
    object_path = objects / f"poster-jpeg-v1-{metadata['image_sha256']}.jpg"
    object_path.write_bytes(b"not a jpeg")
    with patch.object(scene_selection.scene_preview_cache, "object_dir", return_value=objects):
        target = scene_selection.create_scene_selections(str(candidates), str(previews))
    scene = json.loads(Path(target).read_text())["scenes"][0]
    assert scene["candidates"][0]["preview_status"] == "object_invalid"
    assert scene["selected_preview_sha256"]
    assert object_path.read_bytes() == b"not a jpeg"


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_version",
        "binding",
        "provider",
        "aspect",
        "scene_order",
        "candidate_order",
        "preview_status",
        "bool_rank",
        "hold_reference",
        "extra_field",
    ],
)
def test_strict_manifest_validation_rejects_inconsistent_inputs(artifacts, mutation):
    candidates, previews, objects, _, _ = artifacts
    candidate_payload = json.loads(candidates.read_text())
    preview_payload = json.loads(previews.read_text())
    if mutation == "candidate_version":
        candidate_payload["version"] = 2
    elif mutation == "binding":
        preview_payload["source_candidate_manifest"]["sha256"] = "0" * 64
    elif mutation == "provider":
        preview_payload["provider"] = "pixabay"
    elif mutation == "aspect":
        preview_payload["video_aspect"] = "portrait"
    elif mutation == "scene_order":
        preview_payload["scenes"][0]["scene_index"] = 3
    elif mutation == "candidate_order":
        preview_payload["scenes"][0]["candidates"].reverse()
    elif mutation == "preview_status":
        preview_payload["scenes"][0]["status"] = "partial"
    elif mutation == "bool_rank":
        candidate_payload["scenes"][0]["candidates"][0]["provider_rank"] = True
    elif mutation == "hold_reference":
        candidate_payload["scenes"][1]["reuse_scene_index"] = 99
    else:
        preview_payload["unexpected"] = True
    if mutation in {"candidate_version", "bool_rank", "hold_reference"}:
        candidate_bytes = json.dumps(candidate_payload).encode()
        candidates.write_bytes(candidate_bytes)
        preview_payload["source_candidate_manifest"]["sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    previews.write_text(json.dumps(preview_payload))
    with (
        patch.object(scene_selection.scene_preview_cache, "object_dir", return_value=objects),
        pytest.raises(ValueError),
    ):
        scene_selection.create_scene_selections(str(candidates), str(previews))
    assert not (candidates.parent / "scene_selections.json").exists()


@pytest.mark.parametrize("kind", ["empty", "oversized", "symlink"])
def test_bounded_regular_manifest_reads(tmp_path, kind):
    candidate = tmp_path / "scene_candidates.json"
    preview = tmp_path / "scene_previews.json"
    preview.write_text("{}")
    if kind == "empty":
        candidate.write_bytes(b"")
    elif kind == "oversized":
        candidate.write_bytes(b"x" * (scene_selection.MAX_MANIFEST_BYTES + 1))
    else:
        target = tmp_path / "source.json"
        target.write_text("{}")
        candidate.symlink_to(target)
    with pytest.raises(ValueError):
        scene_selection.create_scene_selections(str(candidate), str(preview))


def test_warning_codes_are_stable_deduplicated_and_bounded():
    values = ["candidate_preview_not_ready"] * 3 + [
        "selected_preview_not_ready",
        "source_provider_search_warning",
        "source_provider_budget_warning",
        "candidate_preview_object_missing",
        "candidate_preview_object_invalid",
        "selected_preview_object_missing",
        "selected_preview_object_invalid",
        "vlm_request_failed",
        "raw warning must not escape",
    ]
    result = scene_selection._warnings(values)
    assert result[0] == "candidate_preview_not_ready"
    assert len(result) == 8
    assert result[-1] == "additional_warnings_omitted"
    assert len(result) == len(set(result))
    assert "raw warning must not escape" not in result


def test_atomic_publication_failure_leaves_no_partial_artifact(artifacts):
    candidates, previews, objects, _, _ = artifacts
    with (
        patch.object(scene_selection.scene_preview_cache, "object_dir", return_value=objects),
        patch.object(scene_selection.os, "replace", side_effect=OSError("disk failure")),
        pytest.raises(OSError),
    ):
        scene_selection.create_scene_selections(str(candidates), str(previews))
    assert not (candidates.parent / "scene_selections.json").exists()
    assert not list(candidates.parent.glob("*.tmp"))
