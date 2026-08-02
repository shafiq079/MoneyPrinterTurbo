import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.services import scene_selection


@pytest.fixture(autouse=True)
def _hermetic_scene_ranking_config(monkeypatch):
    """Keep provider-rank tests independent of developer config and secrets."""
    monkeypatch.setattr(scene_selection.config, "scene_ranking", {"enabled": False})
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


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
    image.save(
        path, "JPEG", quality=85, optimize=False, progressive=False, subsampling=2
    )
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
            f"cache_candidate_previews/objects/poster-jpeg-v1-{digest}.jpg"
        ),
    }


def _preview_manifest(
    path: Path, candidate_bytes: bytes, previews: list[dict]
) -> bytes:
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
    with patch.object(
        scene_selection.utils, "storage_dir", return_value=str(objects.parent)
    ):
        target = scene_selection.create_scene_selections(str(candidates), str(previews))

    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["video_aspect"] == "9:16"
    assert (
        data["source_candidate_manifest"]["sha256"]
        == hashlib.sha256(candidate_bytes).hexdigest()
    )
    assert (
        data["source_preview_manifest"]["sha256"]
        == hashlib.sha256(preview_bytes).hexdigest()
    )
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
    with patch.object(
        scene_selection.utils, "storage_dir", return_value=str(objects.parent)
    ):
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
    with patch.object(
        scene_selection.utils, "storage_dir", return_value=str(objects.parent)
    ):
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
        preview_payload["source_candidate_manifest"]["sha256"] = hashlib.sha256(
            candidate_bytes
        ).hexdigest()
    previews.write_text(json.dumps(preview_payload))
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
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


def test_exactly_eight_warning_codes_are_retained_in_order():
    values = [
        "candidate_preview_not_ready",
        "selected_preview_not_ready",
        "source_provider_search_warning",
        "source_provider_budget_warning",
        "candidate_preview_object_missing",
        "candidate_preview_object_invalid",
        "selected_preview_object_missing",
        "selected_preview_object_invalid",
    ]
    assert scene_selection._warnings([values[0], *values, "unknown"]) == values


def test_more_than_eight_warning_codes_are_bounded_and_deduplicated():
    values = [
        "candidate_preview_not_ready",
        "selected_preview_not_ready",
        "source_provider_search_warning",
        "source_provider_budget_warning",
        "candidate_preview_object_missing",
        "candidate_preview_object_invalid",
        "selected_preview_object_missing",
        "selected_preview_object_invalid",
        "vlm_request_failed",
    ]
    result = scene_selection._warnings([values[0], *values, "unknown"])
    assert result == [*values[:7], "additional_warnings_omitted"]


def test_dimension_mismatch_is_rejected_before_pixel_decoding(artifacts):
    _, previews, objects, _, _ = artifacts
    metadata = json.loads(previews.read_text())["scenes"][0]["candidates"][0]["preview"]
    image = MagicMock()
    image.format = "JPEG"
    image.n_frames = 1
    image.size = (scene_selection.scene_preview_cache.MAX_NORMALIZED_EDGE + 1, 12)
    image.mode = "RGB"
    opened = MagicMock()
    opened.__enter__.return_value = image
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(scene_selection.Image, "open", return_value=opened),
    ):
        assert scene_selection._validate_preview_object(metadata) == "object_invalid"
    image.load.assert_not_called()


def test_missing_preview_object_does_not_create_cache_directory(tmp_path):
    cache_root = tmp_path / "storage" / "cache_candidate_previews"
    metadata = {
        "image_sha256": "0" * 64,
        "byte_size": 1,
        "width": 1,
        "height": 1,
    }
    with patch.object(
        scene_selection.utils, "storage_dir", return_value=str(cache_root)
    ):
        assert scene_selection._validate_preview_object(metadata) == "object_missing"
    assert not cache_root.exists()


def test_atomic_publication_failure_leaves_no_partial_artifact(artifacts):
    candidates, previews, objects, _, _ = artifacts
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.os, "replace", side_effect=OSError("disk failure")
        ),
        pytest.raises(OSError),
    ):
        scene_selection.create_scene_selections(str(candidates), str(previews))
    assert not (candidates.parent / "scene_selections.json").exists()
    assert not list(candidates.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("count", "budget", "expected"),
    [
        (5, 0, []),
        (5, 1, [2]),
        (5, 5, [0, 1, 2, 3, 4]),
        (6, 2, [1, 4]),
        (7, 3, [1, 3, 5]),
        (17, 20, list(range(17))),
    ],
)
def test_centered_stratified_allocation(count, budget, expected):
    assert scene_selection._centered_allocation(count, budget) == expected


def _ranking_config(api_key="unit-test-key"):
    return SimpleNamespace(
        enabled=True,
        api_key=api_key,
        max_remote_scene_requests_per_task=12,
        connect_timeout_seconds=10,
        read_timeout_seconds=45,
        total_deadline_seconds=300,
        max_attempts_per_scene=2,
    )


def _ranking_response(unsafe=False):
    return {
        "scene_index": 1,
        "assessments": [
            {
                "label": "C01",
                "relevance": 80,
                "visual_quality": 80,
                "mismatch": 20,
                "unsafe": unsafe,
            },
            {
                "label": "C02",
                "relevance": 90,
                "visual_quality": 90,
                "mismatch": 10,
                "unsafe": unsafe,
            },
        ],
    }


def test_enabled_ranking_selects_safe_candidate_and_counts_usage(artifacts):
    candidates, previews, objects, _, _ = artifacts
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.scene_ranking_cache, "load", return_value=(None, False)
        ),
        patch.object(scene_selection.scene_ranking_cache, "store") as store,
        patch.object(
            scene_selection.scene_ranking,
            "request_remote",
            return_value=(_ranking_response(), 1),
        ) as request,
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), ranking_config=_ranking_config()
        )
    data = json.loads(Path(target).read_text(encoding="utf-8"))
    ranked = data["scenes"][0]
    assert ranked["status"] == "vlm_selected"
    assert ranked["selected_candidate_id"] == "pexels:winner"
    assert ranked["visual_safety_evaluated"] is True
    assert data["usage"]["vlm_requests_started"] == 1
    assert data["usage"]["vlm_attempts_started"] == 1
    assert data["selection_policy_version"] == "nvidia-poster-rank-v1"
    assert "_preview_bytes" not in Path(target).read_text(encoding="utf-8")
    assert '"_source"' not in Path(target).read_text(encoding="utf-8")
    request.assert_called_once()
    store.assert_called_once()


def test_all_unsafe_cache_hit_needs_no_key_or_fallback(artifacts):
    candidates, previews, objects, _, _ = artifacts
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.scene_ranking_cache,
            "load",
            return_value=(_ranking_response(True), False),
        ),
        patch.object(scene_selection.scene_ranking, "request_remote") as request,
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), ranking_config=_ranking_config("")
        )
    ranked = json.loads(Path(target).read_text(encoding="utf-8"))["scenes"][0]
    assert ranked["status"] == "no_safe_candidate"
    assert ranked["selected_candidate_id"] is None
    assert ranked["fallback_reason"] is None
    assert all(item["safety_excluded"] is True for item in ranked["candidates"])
    request.assert_not_called()


@pytest.mark.parametrize(
    ("attempts", "expected_requests", "expected_attempts"),
    [(0, 0, 0), (1, 1, 1), (2, 1, 2)],
)
def test_failed_remote_request_usage_is_exact(
    artifacts, attempts, expected_requests, expected_attempts
):
    candidates, previews, objects, _, _ = artifacts
    error = scene_selection.scene_ranking.RankingError(
        "ranking_deadline_exhausted" if attempts == 0 else "vlm_request_failed",
        attempts=attempts,
    )
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.scene_ranking_cache, "load", return_value=(None, False)
        ),
        patch.object(
            scene_selection.scene_ranking, "request_remote", side_effect=error
        ),
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), ranking_config=_ranking_config()
        )
    usage = json.loads(Path(target).read_text(encoding="utf-8"))["usage"]
    assert usage["vlm_requests_started"] == expected_requests
    assert usage["vlm_attempts_started"] == expected_attempts


@pytest.mark.parametrize(
    "invalid_setting",
    [
        {"provider": "unsupported"},
        {"model": "unsupported"},
        {"api_key": "malformed\nvalue"},
    ],
)
def test_invalid_requested_config_skips_derivative_cache_and_network(
    artifacts, invalid_setting
):
    candidates, previews, objects, _, _ = artifacts
    sleeper = MagicMock()
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.config,
            "scene_ranking",
            {"enabled": True, **invalid_setting},
        ),
        patch.object(scene_selection.scene_ranking, "prepare") as prepare,
        patch.object(scene_selection.scene_ranking_cache, "load") as cache_load,
        patch.object(scene_selection.scene_ranking_cache, "store") as cache_store,
        patch.object(scene_selection.scene_ranking, "request_remote") as request,
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), sleep=sleeper
        )
    raw = Path(target).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["scenes"][0]["status"] == "ranking_unavailable"
    assert data["scenes"][0]["selected_candidate_id"] is None
    assert data["scenes"][0]["fallback_reason"] == "ranking_not_configured"
    assert data["scenes"][1]["status"] == "hold_no_search"
    assert data["scenes"][2]["status"] == "no_candidates"
    assert data["selection_policy_version"] == "nvidia-poster-rank-v1"
    assert "unsupported" not in raw
    assert "malformed" not in raw
    prepare.assert_not_called()
    cache_load.assert_not_called()
    cache_store.assert_not_called()
    request.assert_not_called()
    sleeper.assert_not_called()


def test_incomplete_preview_coverage_skips_cache_and_network(artifacts):
    candidates, previews, objects, _, _ = artifacts
    next(objects.glob("poster-*.jpg")).unlink()
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(scene_selection.scene_ranking_cache, "load") as cache_load,
        patch.object(scene_selection.scene_ranking, "request_remote") as request,
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), ranking_config=_ranking_config()
        )
    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert data["scenes"][0]["fallback_reason"] == "incomplete_preview_coverage"
    assert data["usage"]["vlm_requests_started"] == 0
    assert data["usage"]["vlm_attempts_started"] == 0
    cache_load.assert_not_called()
    request.assert_not_called()


def test_mixed_safety_and_all_tie_breaks():
    def source(identifier, rank):
        return {
            "candidate_id": identifier,
            "provider": "pexels",
            "provider_video_id": identifier,
            "provider_page_url": "https://example.invalid",
            "video_url": "https://example.invalid/video.mp4",
            "provider_rank": rank,
        }

    candidates = [
        {
            "candidate_id": identifier,
            "provider_rank": rank,
            "manifest_position": position,
            "local_order": None,
            "preview_sha256": f"{position:064x}",
            "_source": source(identifier, rank),
        }
        for position, (identifier, rank) in enumerate(
            [("pexels:z", 2), ("pexels:b", 1), ("pexels:a", 1), ("pexels:unsafe", 0)]
        )
    ]
    scene = {"candidates": candidates, "selected_candidate": None}
    response = {
        "assessments": [
            {
                "label": f"C{index:02d}",
                "relevance": 80,
                "visual_quality": 80,
                "mismatch": 20,
                "unsafe": index == 4,
            }
            for index in range(1, 5)
        ]
    }
    scene_selection._apply_ranking(scene, response)
    assert [row["local_order"] for row in candidates] == [3, 1, 2, None]
    assert candidates[3]["safety_excluded"] is True
    assert scene["selected_candidate_id"] == "pexels:b"


def test_low_relevance_and_high_mismatch_fail_closed():
    candidates = [
        {
            "candidate_id": "pexels:low",
            "provider_rank": 1,
            "manifest_position": 0,
            "local_order": None,
            "preview_sha256": "a" * 64,
            "_source": {
                "candidate_id": "pexels:low",
                "provider": "pexels",
                "provider_video_id": "low",
                "provider_page_url": None,
                "video_url": "https://example.invalid/low.mp4",
                "provider_rank": 1,
            },
        },
        {
            "candidate_id": "pexels:mismatch",
            "provider_rank": 2,
            "manifest_position": 1,
            "local_order": None,
            "preview_sha256": "b" * 64,
            "_source": {
                "candidate_id": "pexels:mismatch",
                "provider": "pexels",
                "provider_video_id": "mismatch",
                "provider_page_url": None,
                "video_url": "https://example.invalid/mismatch.mp4",
                "provider_rank": 2,
            },
        },
    ]
    scene = {"candidates": candidates, "selected_candidate": None, "warnings": []}
    scene_selection._apply_ranking(
        scene,
        {
            "assessments": [
                {
                    "label": "C01",
                    "relevance": 59,
                    "visual_quality": 100,
                    "mismatch": 0,
                    "unsafe": False,
                },
                {
                    "label": "C02",
                    "relevance": 100,
                    "visual_quality": 100,
                    "mismatch": 41,
                    "unsafe": False,
                },
            ]
        },
    )
    assert scene["status"] == "no_acceptable_candidate"
    assert scene["selected_candidate_id"] is None


def test_nonconsecutive_duplicate_is_replaced_but_adjacent_reuse_is_allowed():
    def ranked(index, identities, selected=0):
        rows = []
        for order, identity in enumerate(identities, 1):
            source = {
                "candidate_id": f"pexels:{identity}",
                "provider": "pexels",
                "provider_video_id": identity,
                "provider_page_url": None,
                "video_url": f"https://example.invalid/{identity}.mp4",
                "provider_rank": order,
            }
            rows.append(
                {
                    "candidate_id": source["candidate_id"],
                    "provider_rank": order,
                    "manifest_position": order - 1,
                    "local_order": order,
                    "preview_sha256": f"{index + order:064x}",
                    "_source": source,
                    "assessment": {},
                    "score_basis_points": 9000,
                    "safety_excluded": False,
                }
            )
        source = rows[selected]["_source"]
        return {
            "scene_index": index,
            "status": "vlm_selected",
            "warnings": [],
            "visual_safety_evaluated": True,
            "fallback_reason": None,
            "selected_candidate_id": source["candidate_id"],
            "selected_candidate": dict(source),
            "selected_preview_sha256": rows[selected]["preview_sha256"],
            "candidates": rows,
        }

    scenes = [
        ranked(1, ["shared"]),
        ranked(2, ["middle"]),
        ranked(3, ["shared", "alternate"]),
    ]
    scene_selection._suppress_nonconsecutive_duplicates(scenes)
    assert scenes[2]["selected_candidate_id"] == "pexels:alternate"

    adjacent = [ranked(1, ["shared"]), ranked(2, ["shared"])]
    scene_selection._suppress_nonconsecutive_duplicates(adjacent)
    assert adjacent[1]["selected_candidate_id"] == "pexels:shared"


def test_missing_key_cache_miss_falls_back_without_request(artifacts):
    candidates, previews, objects, _, _ = artifacts
    with (
        patch.object(
            scene_selection.utils, "storage_dir", return_value=str(objects.parent)
        ),
        patch.object(
            scene_selection.scene_ranking_cache, "load", return_value=(None, False)
        ),
        patch.object(scene_selection.scene_ranking, "request_remote") as request,
    ):
        target = scene_selection.create_scene_selections(
            str(candidates), str(previews), ranking_config=_ranking_config("")
        )
    data = json.loads(Path(target).read_text(encoding="utf-8"))
    assert data["scenes"][0]["status"] == "ranking_unavailable"
    assert data["scenes"][0]["selected_candidate_id"] is None
    assert data["scenes"][0]["fallback_reason"] == "ranking_not_configured"
    assert data["usage"]["vlm_requests_started"] == 0
    request.assert_not_called()
