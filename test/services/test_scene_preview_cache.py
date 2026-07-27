import hashlib
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
