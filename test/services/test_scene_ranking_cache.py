from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import scene_ranking, scene_ranking_cache


def _response(unsafe=False):
    return {
        "scene_index": 1,
        "assessments": [
            {
                "label": "C01",
                "relevance": 1,
                "visual_quality": 2,
                "mismatch": 3,
                "unsafe": unsafe,
            }
        ],
    }


def test_cache_hit_miss_corruption_and_all_unsafe(tmp_path):
    with patch.object(
        scene_ranking_cache, "cache_dir", side_effect=lambda create=True: tmp_path
    ):
        key = scene_ranking_cache.identity({"version": 1})

        def validator(value):
            return scene_ranking.validate_response(value, 1, ("C01",))

        assert scene_ranking_cache.load(key, validator) == (None, False)
        scene_ranking_cache.store(key, _response(True))
        assert scene_ranking_cache.load(key, validator) == (_response(True), False)
        path = tmp_path / scene_ranking_cache.object_name(key)
        path.write_text("{broken", encoding="utf-8")
        assert scene_ranking_cache.load(key, validator) == (None, True)
        assert not path.exists()


def test_identity_changes_and_contains_no_secret(tmp_path):
    first = scene_ranking_cache.identity({"model": "a", "scene": "x"})
    second = scene_ranking_cache.identity({"model": "b", "scene": "x"})
    assert first != second
    with patch.object(
        scene_ranking_cache, "cache_dir", side_effect=lambda create=True: tmp_path
    ):
        scene_ranking_cache.store(first, _response())
    assert (
        b"api-key"
        not in (tmp_path / scene_ranking_cache.object_name(first)).read_bytes()
    )


def test_atomic_write_failure_leaves_no_temporary(tmp_path):
    key = "a" * 64
    with (
        patch.object(
            scene_ranking_cache, "cache_dir", side_effect=lambda create=True: tmp_path
        ),
        patch.object(scene_ranking_cache.os, "replace", side_effect=OSError("disk")),
    ):
        try:
            scene_ranking_cache.store(key, _response())
        except OSError:
            pass
    assert list(Path(tmp_path).glob("*.tmp")) == []


@pytest.mark.parametrize("digest", ["", "A" * 64, "g" * 64, "../" + "a" * 64, "a" * 63])
def test_invalid_digests_are_rejected_without_filesystem_access(digest):
    with patch.object(scene_ranking_cache, "cache_dir") as directory:
        assert scene_ranking_cache.load(digest, lambda value: value) == (None, True)
        directory.assert_not_called()
    with pytest.raises(ValueError):
        scene_ranking_cache.object_name(digest)


def test_duplicate_keys_oversize_symlink_and_invalid_response_are_corrupt(tmp_path):
    key = "d" * 64

    def validator(value):
        return scene_ranking.validate_response(value, 1, ("C01",))

    with patch.object(
        scene_ranking_cache, "cache_dir", side_effect=lambda create=True: tmp_path
    ):
        path = tmp_path / scene_ranking_cache.object_name(key)
        path.write_text(
            '{"cache_version":"scene-ranking-cache-v1","identity":"'
            + key
            + '","identity":"'
            + key
            + '","response":{}}',
            encoding="utf-8",
        )
        assert scene_ranking_cache.load(key, validator) == (None, True)
        assert not path.exists()

        path.write_bytes(b"x" * (scene_ranking_cache.MAX_CACHE_BYTES + 1))
        assert scene_ranking_cache.load(key, validator) == (None, True)
        assert not path.exists()

        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.symlink_to(target)
        assert scene_ranking_cache.load(key, validator) == (None, True)
        assert not path.exists()

        scene_ranking_cache.store(key, {"scene_index": 1, "assessments": []})
        assert scene_ranking_cache.load(key, validator) == (None, True)
        assert not path.exists()
