import hashlib
import json
import threading
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from app.services import scene_preview_cache as cache


class Response:
    def __init__(self, chunks, status=200, headers=None):
        self.chunks = chunks
        self.status_code = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self.chunks


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def image_bytes(mode="RGB", color=(20, 40, 60), image_format="PNG"):
    output = BytesIO()
    Image.new(mode, (8, 6), color).save(output, format=image_format)
    return output.getvalue()


def test_alpha_is_composited_on_fixed_neutral_background_deterministically():
    source = image_bytes("RGBA", (255, 0, 0, 0))
    first, width, height = cache.normalize_image(source)
    second, _, _ = cache.normalize_image(source)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert (width, height) == (8, 6)
    with Image.open(BytesIO(first)) as normalized:
        pixel = normalized.getpixel((3, 3))
    assert all(abs(channel - 127) <= 2 for channel in pixel)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://images.pexels.com/a.jpg", "https_required"),
        ("https://images.pexels.com.attacker.test/a.jpg", "provider_host_not_allowed"),
        ("https://user:secret@images.pexels.com/a.jpg", "embedded_credentials_or_port"),
        ("https://cdn.pixabay.com/a.jpg", "provider_host_not_allowed"),
    ],
)
def test_provider_url_policy(url, reason):
    with pytest.raises(cache.PreviewError, match=reason):
        cache.find_cached("pexels", "1", url)


def test_download_sends_no_credentials_and_restricts_redirects():
    session = Session([Response([], 302, {"Location": "https://evil.test/a"})])
    with pytest.raises(cache.PreviewError) as caught:
        cache._download(
            "pexels",
            "https://images.pexels.com/a",
            cache.AggregateByteBudget(100),
            session,
            lambda: 0,
        )
    assert caught.value.reason == "provider_host_not_allowed"
    headers = session.calls[0][1]["headers"]
    assert "Authorization" not in headers and "Cookie" not in headers
    assert session.calls[0][1]["allow_redirects"] is False


def test_total_deadline_stops_a_slow_continuous_stream():
    ticks = iter([0, 0, 1, 10, 31])
    session = Session([Response([b"a", b"b", b"c"])])
    with pytest.raises(cache.PreviewError) as caught:
        cache._download(
            "pexels",
            "https://images.pexels.com/a",
            cache.AggregateByteBudget(100),
            session,
            lambda: next(ticks),
        )
    assert caught.value.status == "download_failed"
    assert caught.value.reason == "total_deadline_exceeded"


def test_concurrent_streams_cannot_exceed_aggregate_limit():
    budget = cache.AggregateByteBudget(10)
    barrier = threading.Barrier(2)

    class ConcurrentResponse(Response):
        def iter_content(self, chunk_size):
            del chunk_size
            barrier.wait()
            yield b"123456"

    outcomes = []

    def worker():
        try:
            cache._download(
                "pexels",
                "https://images.pexels.com/a",
                budget,
                Session([ConcurrentResponse([])]),
                lambda: 0,
            )
            outcomes.append("ready")
        except cache.PreviewError as exc:
            outcomes.append(exc.reason)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert budget.used == 6
    assert sorted(outcomes) == ["aggregate_byte_budget_exhausted", "ready"]


def test_prepare_caches_source_and_normalized_object(tmp_path):
    session = Session([Response([image_bytes()])])
    with patch.object(cache, "_cache_root", return_value=tmp_path):
        first = cache.prepare_preview(
            "pexels",
            "1",
            "https://images.pexels.com/a.jpg",
            cache.AggregateByteBudget(100_000),
            session=session,
        )
        second = cache.find_cached("pexels", "1", "https://images.pexels.com/a.jpg")
    assert first.image_sha256 == second.image_sha256
    assert second.cache_hit
    assert len(session.calls) == 1


def normalized_object_bytes():
    return cache.normalize_image(image_bytes(image_format="PNG"))[0]


def write_source_record(tmp_path, data, **overrides):
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(BytesIO(data)) as image:
        payload = {
            "version": 1,
            "normalization_version": cache.NORMALIZATION_VERSION,
            "image_sha256": digest,
            "media_type": "image/jpeg",
            "width": image.width,
            "height": image.height,
            "byte_size": len(data),
            **overrides,
        }
    key = cache._source_key("pexels", "1", "https://images.pexels.com/a.jpg")
    objects = tmp_path / "objects"
    sources = tmp_path / "sources"
    objects.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    object_path = objects / cache.object_name(digest)
    object_path.write_bytes(data)
    source_path = sources / f"source-v1-{key}.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    return object_path, source_path


@pytest.mark.parametrize("corruption", ["bytes", "metadata", "non_jpeg"])
def test_corrupt_cached_object_or_source_metadata_is_rejected(tmp_path, corruption):
    data = normalized_object_bytes()
    if corruption == "non_jpeg":
        data = image_bytes(image_format="PNG")
    overrides = {"width": 999} if corruption == "metadata" else {}
    object_path, source_path = write_source_record(tmp_path, data, **overrides)
    if corruption == "bytes":
        object_path.write_bytes(b"corrupt")

    with patch.object(cache, "_cache_root", return_value=tmp_path):
        result = cache.find_cached("pexels", "1", "https://images.pexels.com/a.jpg")

    assert result is None
    assert not source_path.exists()
    assert not object_path.exists()


def test_symlink_cached_object_is_rejected_without_touching_target(tmp_path):
    data = normalized_object_bytes()
    object_path, source_path = write_source_record(tmp_path, data)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(data)
    object_path.unlink()
    try:
        object_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with patch.object(cache, "_cache_root", return_value=tmp_path):
        result = cache.find_cached("pexels", "1", "https://images.pexels.com/a.jpg")

    assert result is None
    assert outside.read_bytes() == data
    assert not object_path.exists()
    assert not source_path.exists()


def test_valid_existing_content_object_is_reused(tmp_path):
    normalized = normalized_object_bytes()
    digest = hashlib.sha256(normalized).hexdigest()
    objects = tmp_path / "objects"
    objects.mkdir(parents=True)
    object_path = objects / cache.object_name(digest)
    object_path.write_bytes(normalized)
    original_stat = object_path.stat()
    session = Session([Response([image_bytes()])])

    with patch.object(cache, "_cache_root", return_value=tmp_path):
        result = cache.prepare_preview(
            "pexels",
            "1",
            "https://images.pexels.com/a.jpg",
            cache.AggregateByteBudget(100_000),
            session=session,
        )

    assert result.image_sha256 == digest
    assert object_path.stat().st_ino == original_stat.st_ino


def test_corrupt_existing_destination_is_atomically_repaired(tmp_path):
    normalized = normalized_object_bytes()
    digest = hashlib.sha256(normalized).hexdigest()
    objects = tmp_path / "objects"
    objects.mkdir(parents=True)
    object_path = objects / cache.object_name(digest)
    object_path.write_bytes(b"corrupt")

    with patch.object(cache, "_cache_root", return_value=tmp_path):
        result = cache.prepare_preview(
            "pexels",
            "1",
            "https://images.pexels.com/a.jpg",
            cache.AggregateByteBudget(100_000),
            session=Session([Response([image_bytes()])]),
        )

    assert result.image_sha256 == digest
    assert object_path.read_bytes() == normalized


@pytest.mark.parametrize("fails", [False, True])
def test_owned_session_is_closed_after_success_and_failure(tmp_path, fails):
    response = Response([b"not-an-image"] if fails else [image_bytes()])

    class OwnedSession(Session):
        instances = []

        def __init__(self):
            super().__init__([response])
            self.closed = False
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    with (
        patch.object(cache, "_cache_root", return_value=tmp_path),
        patch.object(cache.requests, "Session", OwnedSession),
    ):
        if fails:
            with pytest.raises(cache.PreviewError):
                cache.prepare_preview(
                    "pexels",
                    "1",
                    "https://images.pexels.com/a.jpg",
                    cache.AggregateByteBudget(100_000),
                )
        else:
            cache.prepare_preview(
                "pexels",
                "1",
                "https://images.pexels.com/a.jpg",
                cache.AggregateByteBudget(100_000),
            )
    assert OwnedSession.instances[0].closed


def test_injected_session_is_not_closed(tmp_path):
    session = Session([Response([image_bytes()])])
    session.closed = False
    session.close = lambda: setattr(session, "closed", True)
    with patch.object(cache, "_cache_root", return_value=tmp_path):
        cache.prepare_preview(
            "pexels",
            "1",
            "https://images.pexels.com/a.jpg",
            cache.AggregateByteBudget(100_000),
            session=session,
        )
    assert not session.closed
