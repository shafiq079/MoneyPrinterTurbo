import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services import scene_ranking


def _jpeg(color=(20, 40, 60), size=(80, 60)):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG")
    return output.getvalue()


def _scene(count=2):
    return {
        "scene_index": 1,
        "text": "Narration, not instructions.",
        "queries": ["city skyline"],
        "candidates": [
            {
                "candidate_id": f"pexels:{index}",
                "provider_rank": index,
                "preview_sha256": f"{index:064x}",
            }
            for index in range(1, count + 1)
        ],
    }


def _valid(labels, unsafe=False):
    return {
        "scene_index": 1,
        "assessments": [
            {
                "label": label,
                "relevance": 83,
                "visual_quality": 91,
                "mismatch": 12,
                "unsafe": unsafe,
            }
            for label in labels
        ],
    }


@pytest.mark.parametrize("count", range(1, 13))
def test_grid_and_stable_mapping_cover_every_candidate(count):
    prepared = scene_ranking.prepare(
        _scene(count),
        [_jpeg() for _ in range(count)],
        "nvidia_hosted",
        scene_ranking.MODEL,
        "16:9",
    )
    assert prepared.labels == tuple(f"C{index:02d}" for index in range(1, count + 1))
    prompt = prepared.request["messages"][1]["content"][0]["text"]
    for index in range(1, count + 1):
        assert prompt.count(f'"label":"C{index:02d}"') >= 1
        assert prompt.count(f'"candidate_id":"pexels:{index}"') == 1
    assert len(prepared.jpeg) <= 120_000
    assert len(prepared.request_bytes) <= 190_000


def test_exact_request_and_scoring_example():
    prepared = scene_ranking.prepare(
        _scene(1), [_jpeg()], "nvidia_hosted", scene_ranking.MODEL, "9:16"
    )
    assert set(prepared.request) == {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_tokens",
        "stream",
    }
    assert prepared.request | {"messages": None} == {
        "model": scene_ranking.MODEL,
        "messages": None,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2048,
        "stream": False,
    }
    item = scene_ranking.Assessment("C01", 83, 91, 12, False)
    assert item.score == 8600


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{}\n```",
        '{"scene_index":1,"scene_index":1,"assessments":[]}',
        '{"scene_index":1,"assessments":[{"label":"C01","relevance":NaN,"visual_quality":1,"mismatch":1,"unsafe":false}]}',
    ],
)
def test_strict_content_rejects_fences_duplicates_and_nan(content):
    with pytest.raises(ValueError):
        scene_ranking.parse_content(content, 1, ("C01",))


def test_boolean_integer_and_label_errors_rejected():
    payload = _valid(("C01",))
    payload["assessments"][0]["relevance"] = True
    with pytest.raises(ValueError):
        scene_ranking.validate_response(payload, 1, ("C01",))
    payload = _valid(("C02",))
    with pytest.raises(ValueError):
        scene_ranking.validate_response(payload, 1, ("C01",))


class Response:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self.raw = json.dumps(payload).encode()
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.raw

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.responses)


def test_mocked_request_retry_and_envelope_without_secret_in_body():
    prepared = scene_ranking.prepare(
        _scene(1), [_jpeg()], "nvidia_hosted", scene_ranking.MODEL, "16:9"
    )
    content = json.dumps(_valid(prepared.labels))
    session = Session(
        [
            Response(429, {}, {"Retry-After": "0"}),
            Response(
                200,
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ]
                },
            ),
        ]
    )
    cfg = SimpleNamespace(
        max_attempts_per_scene=2, connect_timeout_seconds=10, read_timeout_seconds=45
    )
    result, attempts = scene_ranking.request_remote(
        prepared,
        1,
        "test-key",
        cfg,
        session=session,
        monotonic=lambda: 1.0,
        sleep=lambda _: None,
        deadline=100,
    )
    assert result == _valid(prepared.labels)
    assert attempts == 2
    assert b"test-key" not in prepared.request_bytes
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer test-key"
    assert all(call[1]["stream"] is True for call in session.calls)


def test_finish_reason_and_oversized_response_rejected():
    prepared = scene_ranking.prepare(
        _scene(1), [_jpeg()], "nvidia_hosted", scene_ranking.MODEL, "16:9"
    )
    cfg = SimpleNamespace(
        max_attempts_per_scene=1, connect_timeout_seconds=10, read_timeout_seconds=45
    )
    for response in (
        Response(
            200,
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        ),
        Response(200, {}),
    ):
        if response.raw == b"{}":
            response.raw = b"x" * (scene_ranking.MAX_RESPONSE_BYTES + 1)
        with pytest.raises(scene_ranking.RankingError):
            scene_ranking.request_remote(
                prepared,
                1,
                "x",
                cfg,
                session=Session([response]),
                monotonic=lambda: 1.0,
                sleep=lambda _: None,
                deadline=100,
            )
