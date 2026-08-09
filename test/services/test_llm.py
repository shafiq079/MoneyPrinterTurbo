import json
import os
import subprocess
import sys
import tempfile
import threading
import tomllib
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    LLM_PROVIDERS,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import VideoScriptRequest, VideoSocialMetadataRequest
from app.services import llm


def _semantic_entry(index=1):
    return {
        "scene_index": index,
        "queries": ["cacao harvest"],
        "requirements": {
            "primary_entities": [{"canonical": "cacao", "aliases": ["cocoa"]}],
            "actions": [],
            "contexts": [],
        },
    }


class _SemanticProvider(BaseHTTPRequestHandler):
    reached = 0

    def do_POST(self):
        type(self).reached += 1
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        prompt = request["messages"][0]["content"]
        scenes = json.loads(prompt.split("Scenes: ", 1)[1])
        plans = [
            {
                "scene_index": scene["scene_index"],
                "queries": ["cacao harvest"],
                "requirements": {
                    "primary_entities": [{"canonical": "cacao", "aliases": ["cocoa"]}],
                    "actions": [],
                    "contexts": [],
                },
            }
            for scene in scenes
        ]
        content = json.dumps({"scenes": plans})
        response = json.dumps(
            {
                "id": "mock",
                "object": "chat.completion",
                "created": 1,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


def _provider_snapshot(server):
    return {
        "provider_id": "openai",
        "adapter": "openai_compatible",
        "model": "mock-model",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "api_key": "semantic-test-secret",
        "api_version": "",
        "extras": {},
    }


def _openai_status_error(error_type, status):
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return error_type("sensitive provider body", response=response, body=None)


def _assert_semantic_failure(callable_, reason, retryable):
    try:
        callable_()
    except llm.SemanticRequestFailure as exc:
        assert exc.code is reason
        assert exc.retryable is retryable
        assert str(exc) == reason.value
        assert exc.args == (reason.value,)
    else:
        raise AssertionError("expected SemanticRequestFailure")


def test_semantic_provider_exception_taxonomy_and_retryability():
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    cases = (
        (
            openai.APITimeoutError(request=request),
            llm.SemanticPlanDiagnostic.provider_timeout,
            True,
        ),
        (
            openai.APIConnectionError(message="secret", request=request),
            llm.SemanticPlanDiagnostic.provider_connection_failed,
            True,
        ),
        (
            _openai_status_error(openai.RateLimitError, 429),
            llm.SemanticPlanDiagnostic.provider_rate_limited,
            True,
        ),
        (
            _openai_status_error(openai.BadRequestError, 400),
            llm.SemanticPlanDiagnostic.provider_http_client_error,
            False,
        ),
        (
            _openai_status_error(openai.InternalServerError, 503),
            llm.SemanticPlanDiagnostic.provider_http_server_error,
            True,
        ),
        (
            _openai_status_error(openai.AuthenticationError, 401),
            llm.SemanticPlanDiagnostic.provider_authentication_failed,
            False,
        ),
        (
            _openai_status_error(openai.PermissionDeniedError, 403),
            llm.SemanticPlanDiagnostic.provider_permission_denied,
            False,
        ),
        (
            RuntimeError("secret raw provider exception"),
            llm.SemanticPlanDiagnostic.provider_failed,
            False,
        ),
    )
    for provider_error, reason, retryable in cases:
        with unittest.TestCase().subTest(reason=reason.value):
            failure = llm._classify_semantic_provider_exception(provider_error)
            assert failure.code is reason
            assert failure.retryable is retryable
            assert failure.args == (reason.value,)
            assert "secret" not in str(failure)


def test_semantic_response_envelope_taxonomy_and_normalization():
    def response(*, choices=None, finish="stop", content="{}", message=True, refusal=None):
        if choices is not None:
            return types.SimpleNamespace(choices=choices)
        value = None
        if message:
            value = types.SimpleNamespace(content=content, refusal=refusal)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason=finish, message=value)]
        )

    failures = (
        (
            response(choices=[]),
            llm.SemanticPlanDiagnostic.provider_invalid_envelope,
            True,
        ),
        (
            response(message=False),
            llm.SemanticPlanDiagnostic.provider_invalid_envelope,
            True,
        ),
        (
            response(content=""),
            llm.SemanticPlanDiagnostic.provider_empty_content,
            True,
        ),
        (
            response(content=[]),
            llm.SemanticPlanDiagnostic.provider_empty_content,
            True,
        ),
        (
            response(finish="content_filter"),
            llm.SemanticPlanDiagnostic.provider_content_filtered,
            False,
        ),
        (
            response(refusal="private refusal text"),
            llm.SemanticPlanDiagnostic.provider_content_filtered,
            False,
        ),
        (
            response(finish="length"),
            llm.SemanticPlanDiagnostic.provider_output_truncated,
            True,
        ),
        (
            response(finish="tool_calls"),
            llm.SemanticPlanDiagnostic.provider_invalid_envelope,
            True,
        ),
    )
    for provider_response, reason, retryable in failures:
        with unittest.TestCase().subTest(reason=reason.value):
            _assert_semantic_failure(
                lambda: llm._semantic_extract_chat_completion_text(
                    provider_response, "openai"
                ),
                reason,
                retryable,
            )

    assert (
        llm._semantic_extract_chat_completion_text(
            response(content='<think>private reasoning</think>\n{"scenes":[]}'),
            "openai",
        )
        == '{"scenes":[]}'
    )
    _assert_semantic_failure(
        lambda: llm._semantic_extract_chat_completion_text(
            response(content="<think>unfinished reasoning"), "openai"
        ),
        llm.SemanticPlanDiagnostic.provider_empty_content,
        True,
    )


def test_semantic_output_allowance_is_bounded_and_provider_neutral():
    assert [llm._semantic_output_token_allowance(count) for count in range(1, 5)] == [
        2048,
        3072,
        4096,
        4096,
    ]
    assert llm._semantic_output_token_allowance(4, 2048) == 2048
    for invalid in (0, 5, True):
        _assert_semantic_failure(
            lambda value=invalid: llm._semantic_output_token_allowance(value),
            llm.SemanticPlanDiagnostic.provider_configuration_invalid,
            False,
        )


def test_supported_semantic_adapters_send_only_declared_capabilities():
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content="{}", refusal=None),
            )
        ]
    )
    for adapter in llm._SEMANTIC_ADAPTER_CAPABILITIES:
        completions = types.SimpleNamespace(create=MagicMock(return_value=response))
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions),
            close=MagicMock(),
        )
        snapshot = {
            "provider_id": "arbitrary-provider",
            "adapter": adapter,
            "model": "arbitrary-model",
            "base_url": "https://provider.invalid/v1",
            "api_key": "secret",
            "api_version": "version",
            "extras": {"account_id": "account", "gateway_id": "gateway"},
        }
        constructor = "AzureOpenAI" if adapter == "azure" else "OpenAI"
        with patch.object(llm, constructor, return_value=client):
            assert llm._semantic_generate_from_snapshot(snapshot, "prompt", 3072) == "{}"
        kwargs = completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 3072
        assert "max_completion_tokens" not in kwargs
        if adapter == "modelscope":
            assert kwargs["extra_body"] == {"enable_thinking": False}
        else:
            assert "extra_body" not in kwargs
        client.close.assert_called_once()


def test_semantic_ipc_v2_is_strict_and_rejects_v1():
    scene = types.SimpleNamespace(index=1, text="scene")
    snapshot = {
        "provider_id": "provider",
        "adapter": "openai_compatible",
        "model": "model",
        "base_url": "https://provider.invalid/v1",
        "api_key": "secret",
        "api_version": "",
        "extras": {},
    }
    framed = llm._worker_request(7, "subject", (scene,), 3, 2048, snapshot)
    length = llm._SEMANTIC_IPC_HEADER.unpack(framed[:4])[0]
    payload = json.loads(framed[4:])
    assert length == len(framed) - 4
    assert payload["version"] == 2
    assert payload["work_id"] == 7
    assert payload["max_output_tokens"] == 2048
    assert "batch_index" not in payload

    version_one = json.dumps(
        {
            "version": 1,
            "work_id": 7,
            "ok": False,
            "retryable": False,
            "diagnostic": "provider_failed",
            "plans": [],
            "issues": [],
        }
    ).encode()
    result, diagnostic, retryable = llm._decode_semantic_ipc(
        version_one, 7, (scene,), 3
    )
    assert result is None
    assert diagnostic is llm.SemanticPlanDiagnostic.ipc_invalid
    assert retryable is False

    old_request = dict(payload)
    old_request["version"] = 1
    worker_result = llm._semantic_worker_payload(old_request)
    assert worker_result == {
        "version": 2,
        "work_id": 7,
        "ok": False,
        "retryable": False,
        "diagnostic": "invalid_input",
        "plans": [],
        "issues": [],
    }


def test_semantic_rate_gate_has_no_burst_and_honors_deadline():
    gate = llm._SemanticRateGate(0.0)
    starts = []
    for now in (0.0, 3.0, 6.0):
        assert llm._semantic_rate_start_delay(gate, now, 10.0, 20) == 0
        starts.append(llm._semantic_record_request_start(gate, now, 20))
    assert starts == [0.0, 3.0, 6.0]
    assert llm._semantic_rate_start_delay(gate, 8.0, 9.0, 20) is None


def test_semantic_scheduler_does_not_start_work_past_rate_deadline():
    scene = types.SimpleNamespace(index=1, text="scene")
    popen = MagicMock()
    settings = config.SemanticPlanningConfig(4096, 20)
    results, unreaped = llm._run_semantic_phase(
        [(0, (scene,))],
        "subject",
        3,
        5.0,
        provider_snapshot={
            "provider_id": "provider",
            "adapter": "openai_compatible",
            "model": "model",
            "base_url": "https://provider.invalid/v1",
            "api_key": "secret",
            "api_version": "",
            "extras": {},
        },
        planning_config=settings,
        rate_gate=llm._SemanticRateGate(5.0),
        popen=popen,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("slept")),
    )
    popen.assert_not_called()
    assert unreaped is False
    assert results[0][1] is llm.SemanticPlanDiagnostic.task_deadline_exhausted


def test_scene_query_plan_has_strict_typed_semantic_requirements():
    scene = types.SimpleNamespace(index=1, text="可可豆在阳光下晾晒")
    response = json.dumps(
        {
            "scenes": [
                {
                    "scene_index": 1,
                    "queries": ["cacao beans drying in sun"],
                    "requirements": {
                        "primary_entities": [
                            {"canonical": "cacao", "aliases": ["cocoa"]}
                        ],
                        "actions": [{"canonical": "drying", "aliases": []}],
                        "contexts": [],
                    },
                }
            ]
        }
    )
    with patch.object(llm, "_generate_response", return_value=response):
        plans, warning = llm.generate_scene_queries("巧克力", [scene])
    assert warning is None
    assert plans[1].queries == ("cacao beans drying in sun",)
    assert plans[1].requirements.primary_entities[0].aliases == ("cocoa",)


def test_semantic_schema_reasons_are_specific_and_scene_bounded():
    scene = types.SimpleNamespace(index=1, text="scene")

    def reason_for(payload):
        result = llm._validate_scene_query_batch(json.dumps(payload), [scene], 3)
        assert not result.plans
        return result.issues[0]

    cases = []

    def case(reason, mutate):
        payload = {"scenes": [_semantic_entry()]}
        value = mutate(payload)
        cases.append(
            (
                reason,
                value
                if reason is llm.SemanticPlanDiagnostic.root_not_object
                else payload,
            )
        )

    case(llm.SemanticPlanDiagnostic.root_not_object, lambda _p: [])
    case(llm.SemanticPlanDiagnostic.root_fields_invalid, lambda p: p.update(extra=[]))
    case(llm.SemanticPlanDiagnostic.scenes_not_array, lambda p: p.update(scenes={}))
    case(
        llm.SemanticPlanDiagnostic.scene_entry_not_object,
        lambda p: p.update(scenes=[1]),
    )
    case(
        llm.SemanticPlanDiagnostic.scene_index_missing,
        lambda p: p["scenes"][0].pop("scene_index"),
    )
    case(
        llm.SemanticPlanDiagnostic.scene_index_invalid,
        lambda p: p["scenes"][0].update(scene_index="1"),
    )
    case(
        llm.SemanticPlanDiagnostic.scene_index_unexpected,
        lambda p: p["scenes"][0].update(scene_index=2),
    )
    case(
        llm.SemanticPlanDiagnostic.entry_fields_invalid,
        lambda p: p["scenes"][0].update(extra=True),
    )
    case(
        llm.SemanticPlanDiagnostic.queries_not_array,
        lambda p: p["scenes"][0].update(queries="query"),
    )
    case(
        llm.SemanticPlanDiagnostic.query_count_invalid,
        lambda p: p["scenes"][0].update(queries=[]),
    )
    case(
        llm.SemanticPlanDiagnostic.query_not_string,
        lambda p: p["scenes"][0].update(queries=[1]),
    )
    case(
        llm.SemanticPlanDiagnostic.query_control_character,
        lambda p: p["scenes"][0].update(queries=["bad\nquery"]),
    )
    case(
        llm.SemanticPlanDiagnostic.query_empty,
        lambda p: p["scenes"][0].update(queries=["  "]),
    )
    case(
        llm.SemanticPlanDiagnostic.query_too_long,
        lambda p: p["scenes"][0].update(queries=["x" * 81]),
    )
    case(
        llm.SemanticPlanDiagnostic.requirements_not_object,
        lambda p: p["scenes"][0].update(requirements=[]),
    )
    case(
        llm.SemanticPlanDiagnostic.requirements_fields_invalid,
        lambda p: p["scenes"][0]["requirements"].pop("actions"),
    )
    case(
        llm.SemanticPlanDiagnostic.primary_entity_count_invalid,
        lambda p: p["scenes"][0]["requirements"].update(primary_entities=[]),
    )
    case(
        llm.SemanticPlanDiagnostic.action_count_invalid,
        lambda p: p["scenes"][0]["requirements"].update(actions=[{}] * 4),
    )
    case(
        llm.SemanticPlanDiagnostic.context_count_invalid,
        lambda p: p["scenes"][0]["requirements"].update(contexts=[{}] * 4),
    )
    case(
        llm.SemanticPlanDiagnostic.term_group_not_object,
        lambda p: p["scenes"][0]["requirements"].update(primary_entities=[1]),
    )
    case(
        llm.SemanticPlanDiagnostic.term_group_fields_invalid,
        lambda p: p["scenes"][0]["requirements"]["primary_entities"][0].pop("aliases"),
    )
    case(
        llm.SemanticPlanDiagnostic.canonical_invalid,
        lambda p: p["scenes"][0]["requirements"]["primary_entities"][0].update(
            canonical=""
        ),
    )
    case(
        llm.SemanticPlanDiagnostic.aliases_not_array,
        lambda p: p["scenes"][0]["requirements"]["primary_entities"][0].update(
            aliases="cocoa"
        ),
    )
    case(
        llm.SemanticPlanDiagnostic.alias_count_invalid,
        lambda p: p["scenes"][0]["requirements"]["primary_entities"][0].update(
            aliases=["a1", "a2", "a3", "a4", "a5"]
        ),
    )
    case(
        llm.SemanticPlanDiagnostic.alias_invalid,
        lambda p: p["scenes"][0]["requirements"]["primary_entities"][0].update(
            aliases=[""]
        ),
    )
    case(
        llm.SemanticPlanDiagnostic.primary_entity_too_generic,
        lambda p: p["scenes"][0]["requirements"].update(
            primary_entities=[{"canonical": "person", "aliases": []}]
        ),
    )

    large_groups = [
        {
            "canonical": f"specific{i}" + "x" * 38,
            "aliases": [f"alias{i}{j}" + "y" * 38 for j in range(4)],
        }
        for i in range(4)
    ]
    case(
        llm.SemanticPlanDiagnostic.requirements_too_large,
        lambda p: p["scenes"][0]["requirements"].update(primary_entities=large_groups),
    )

    for expected, payload in cases:
        with unittest.TestCase().subTest(reason=expected.value):
            index, actual = reason_for(payload)
            assert actual is expected
            assert index in {None, 1, 2}


def test_semantic_batch_preserves_valid_entry_next_to_invalid_entry():
    scenes = [
        types.SimpleNamespace(index=1, text="one"),
        types.SimpleNamespace(index=2, text="two"),
    ]
    invalid = _semantic_entry(2)
    invalid["requirements"]["primary_entities"][0]["aliases"] = "cocoa"
    result = llm._validate_scene_query_batch(
        json.dumps({"scenes": [_semantic_entry(1), invalid]}), scenes, 3
    )
    assert [index for index, _plan in result.plans] == [1]
    assert result.issues == ((2, llm.SemanticPlanDiagnostic.aliases_not_array),)


def test_semantic_coverage_reasons_are_distinct():
    scenes = [
        types.SimpleNamespace(index=1, text="one"),
        types.SimpleNamespace(index=2, text="two"),
    ]
    duplicate = llm._validate_scene_query_batch(
        json.dumps({"scenes": [_semantic_entry(1), _semantic_entry(1)]}), scenes, 3
    )
    assert (1, llm.SemanticPlanDiagnostic.scene_index_duplicate) in duplicate.issues
    missing = llm._validate_scene_query_batch(
        json.dumps({"scenes": [_semantic_entry(1)]}), scenes, 3
    )
    assert (2, llm.SemanticPlanDiagnostic.coverage_invalid) in missing.issues
    reordered = llm._validate_scene_query_batch(
        json.dumps({"scenes": [_semantic_entry(2), _semantic_entry(1)]}), scenes, 3
    )
    assert (
        None,
        llm.SemanticPlanDiagnostic.scene_order_invalid,
    ) in reordered.issues


def test_semantic_corrective_retry_contains_only_unresolved_scenes():
    scenes = [
        types.SimpleNamespace(index=1, text="one"),
        types.SimpleNamespace(index=2, text="two"),
    ]
    requirements = llm.SceneSemanticRequirements(
        primary_entities=(llm.SemanticTermGroup("cacao", ("cocoa",)),)
    )
    plan1 = llm.SceneQueryPlan(("first",), requirements)
    plan2 = llm.SceneQueryPlan(("second",), requirements)
    phases = [
        (
            {
                0: (
                    llm._BatchValidationResult(
                        ((1, plan1),),
                        ((2, llm.SemanticPlanDiagnostic.aliases_not_array),),
                    ),
                    llm.SemanticPlanDiagnostic.aliases_not_array,
                    True,
                    False,
                )
            },
            False,
        ),
        (
            {
                1: (
                    llm._BatchValidationResult(((2, plan2),), ()),
                    llm.SemanticPlanDiagnostic.complete,
                    False,
                    False,
                )
            },
            False,
        ),
    ]
    with patch.object(llm, "_run_semantic_phase", side_effect=phases) as run:
        result = llm.generate_scene_query_plan("subject", scenes)
    retry_works = run.call_args_list[1].args[0]
    assert [scene.index for scene in retry_works[0][1]] == [2]
    assert result.state is llm.SemanticPlanState.complete
    assert [index for index, _plan in result.plans] == [1, 2]
    assert result.issues == ()


def test_semantic_retry_repairs_transient_provider_failure():
    scene = types.SimpleNamespace(index=1, text="one")
    requirements = llm.SceneSemanticRequirements(
        primary_entities=(llm.SemanticTermGroup("cacao", ("cocoa",)),)
    )
    plan = llm.SceneQueryPlan(("first",), requirements)
    phases = [
        (
            {
                0: (
                    None,
                    llm.SemanticPlanDiagnostic.provider_timeout,
                    True,
                    False,
                )
            },
            False,
        ),
        (
            {
                1: (
                    llm._BatchValidationResult(((1, plan),), ()),
                    llm.SemanticPlanDiagnostic.complete,
                    False,
                    False,
                )
            },
            False,
        ),
    ]
    with patch.object(llm, "_run_semantic_phase", side_effect=phases) as run:
        result = llm.generate_scene_query_plan("subject", [scene])
    assert run.call_count == 2
    assert result.state is llm.SemanticPlanState.complete
    assert result.plans == ((1, plan),)
    assert result.issues == ()


def test_truncated_four_scene_batch_retries_as_ordered_pairs():
    scenes = [types.SimpleNamespace(index=index, text=str(index)) for index in range(1, 5)]
    requirements = llm.SceneSemanticRequirements(
        primary_entities=(llm.SemanticTermGroup("cacao", ("cocoa",)),)
    )

    def plan(index):
        return llm.SceneQueryPlan((f"query {index}",), requirements)

    phases = [
        (
            {
                0: (
                    None,
                    llm.SemanticPlanDiagnostic.provider_output_truncated,
                    True,
                    False,
                )
            },
            False,
        ),
        (
            {
                # Deliberately return the later pair first in mapping order.
                2: (
                    llm._BatchValidationResult(((3, plan(3)), (4, plan(4))), ()),
                    llm.SemanticPlanDiagnostic.complete,
                    False,
                    False,
                ),
                1: (
                    llm._BatchValidationResult(((1, plan(1)), (2, plan(2))), ()),
                    llm.SemanticPlanDiagnostic.complete,
                    False,
                    False,
                ),
            },
            False,
        ),
    ]
    with patch.object(llm, "_run_semantic_phase", side_effect=phases) as run:
        result = llm.generate_scene_query_plan("subject", scenes)
    retry_works = run.call_args_list[1].args[0]
    assert [[scene.index for scene in batch] for _work_id, batch in retry_works] == [
        [1, 2],
        [3, 4],
    ]
    assert [index for index, _plan in result.plans] == [1, 2, 3, 4]
    assert result.attempts_started == 3
    assert result.state is llm.SemanticPlanState.complete


def test_persistent_pair_truncation_is_scene_bounded_at_attempt_two():
    scenes = [types.SimpleNamespace(index=index, text=str(index)) for index in range(1, 5)]
    truncated = (
        None,
        llm.SemanticPlanDiagnostic.provider_output_truncated,
        True,
        False,
    )
    with patch.object(
        llm,
        "_run_semantic_phase",
        side_effect=[({0: truncated}, False), ({1: truncated, 2: truncated}, False)],
    ) as run:
        result = llm.generate_scene_query_plan("subject", scenes)
    assert run.call_count == 2
    assert result.attempts_started == 3
    assert result.state is llm.SemanticPlanState.unavailable
    assert result.issues == tuple(
        llm.SemanticPlanIssue(
            0,
            index,
            llm.SemanticPlanDiagnostic.provider_output_truncated,
            2,
        )
        for index in range(1, 5)
    )


def test_forty_scene_truncation_never_exceeds_thirty_requests():
    scenes = [types.SimpleNamespace(index=index, text=str(index)) for index in range(40)]
    truncated = (
        None,
        llm.SemanticPlanDiagnostic.provider_output_truncated,
        True,
        False,
    )
    first = {index: truncated for index in range(10)}
    second = {index: truncated for index in range(10, 30)}
    with patch.object(
        llm, "_run_semantic_phase", side_effect=[(first, False), (second, False)]
    ) as run:
        result = llm.generate_scene_query_plan("subject", scenes)
    first_works = run.call_args_list[0].args[0]
    retry_works = run.call_args_list[1].args[0]
    assert len(first_works) == 10
    assert len(retry_works) == 20
    assert all(len(batch) == 2 for _work_id, batch in retry_works)
    assert result.attempts_started == 30
    retry_indexes = [scene.index for _work_id, batch in retry_works for scene in batch]
    assert retry_indexes == list(range(40))


def test_semantic_final_provider_failure_is_bounded_and_keeps_other_batches():
    scenes = [types.SimpleNamespace(index=index, text=str(index)) for index in range(1, 6)]
    requirements = llm.SceneSemanticRequirements(
        primary_entities=(llm.SemanticTermGroup("cacao", ("cocoa",)),)
    )
    plans = tuple(
        (index, llm.SceneQueryPlan((f"query {index}",), requirements))
        for index in range(1, 5)
    )
    phases = [
        (
            {
                0: (
                    llm._BatchValidationResult(plans, ()),
                    llm.SemanticPlanDiagnostic.complete,
                    False,
                    False,
                ),
                1: (
                    None,
                    llm.SemanticPlanDiagnostic.provider_rate_limited,
                    True,
                    False,
                ),
            },
            False,
        ),
        (
            {
                2: (
                    None,
                    llm.SemanticPlanDiagnostic.provider_rate_limited,
                    True,
                    False,
                )
            },
            False,
        ),
    ]
    with patch.object(llm, "_run_semantic_phase", side_effect=phases):
        result = llm.generate_scene_query_plan("subject", scenes)
    assert result.state is llm.SemanticPlanState.partial
    assert [index for index, _plan in result.plans] == [1, 2, 3, 4]
    assert result.issues == (
        llm.SemanticPlanIssue(
            1,
            None,
            llm.SemanticPlanDiagnostic.provider_rate_limited,
            2,
        ),
    )


def test_semantic_worker_failure_payload_exposes_only_bounded_diagnostic():
    secret = "semantic-super-secret"
    request = {
        "version": 2,
        "work_id": 0,
        "video_subject": "private narration subject",
        "scenes": [{"index": 1, "text": "private narration"}],
        "max_queries_per_scene": 2,
        "max_output_tokens": 2048,
        "provider": {
            "provider_id": "openai",
            "adapter": "openai_compatible",
            "model": "model",
            "base_url": "https://user:password@provider.invalid/v1",
            "api_key": secret,
            "api_version": "",
            "extras": {},
        },
    }
    failure = llm.SemanticRequestFailure(
        llm.SemanticPlanDiagnostic.provider_authentication_failed,
        retryable=False,
    )
    with patch.object(llm, "_semantic_generate_from_snapshot", side_effect=failure):
        payload = llm._semantic_worker_payload(request)
    serialized = json.dumps(payload)
    assert payload == {
        "version": 2,
        "work_id": 0,
        "ok": False,
        "retryable": False,
        "diagnostic": "provider_authentication_failed",
        "plans": [],
        "issues": [],
    }
    for forbidden in (secret, "password", "private narration", "provider.invalid"):
        assert forbidden not in serialized


def test_semantic_partial_result_keeps_valid_scenes_and_bounded_issue():
    scenes = [
        types.SimpleNamespace(index=1, text="one"),
        types.SimpleNamespace(index=2, text="two"),
    ]
    requirements = llm.SceneSemanticRequirements(
        primary_entities=(llm.SemanticTermGroup("cacao", ("cocoa",)),)
    )
    valid = llm.SceneQueryPlan(("first",), requirements)
    unresolved = llm._BatchValidationResult(
        (), ((2, llm.SemanticPlanDiagnostic.aliases_not_array),)
    )
    with patch.object(
        llm,
        "_run_semantic_phase",
        side_effect=[
            (
                {
                    0: (
                        llm._BatchValidationResult(
                            ((1, valid),),
                            ((2, llm.SemanticPlanDiagnostic.aliases_not_array),),
                        ),
                        llm.SemanticPlanDiagnostic.aliases_not_array,
                        True,
                        False,
                    )
                },
                False,
            ),
            (
                {
                    1: (
                        unresolved,
                        llm.SemanticPlanDiagnostic.aliases_not_array,
                        True,
                        False,
                    )
                },
                False,
            ),
        ],
    ):
        result = llm.generate_scene_query_plan("subject", scenes)
    assert result.state is llm.SemanticPlanState.partial
    assert [index for index, _plan in result.plans] == [1]
    assert result.issues == (
        llm.SemanticPlanIssue(0, 2, llm.SemanticPlanDiagnostic.aliases_not_array, 2),
    )


def test_dedicated_worker_reaches_mock_provider_and_returns_complete_plan():
    _SemanticProvider.reached = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SemanticProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scenes = [
            types.SimpleNamespace(index=index, text="scene") for index in range(1, 9)
        ]
        result = llm.generate_scene_query_plan(
            "subject", scenes, provider_snapshot=_provider_snapshot(server)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.state is llm.SemanticPlanState.complete
    assert [index for index, _ in result.plans] == list(range(1, 9))
    assert _SemanticProvider.reached == 2


def test_dedicated_worker_never_reimports_hostile_parent_main(tmp_path):
    _SemanticProvider.reached = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SemanticProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    marker = tmp_path / "imports.txt"
    script = tmp_path / "hostile_parent.py"
    script.write_text(
        """
import json
from pathlib import Path
from types import SimpleNamespace
from app.services import llm
marker = Path(__file__).with_name("imports.txt")
marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
snapshot = {snapshot}
result = llm.generate_scene_query_plan("subject", [SimpleNamespace(index=1, text="scene")], provider_snapshot=snapshot)
print(json.dumps({{"state": result.state.value, "count": len(result.plans)}}))
""".format(snapshot=repr(_provider_snapshot(server))),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=Path(__file__).parents[2],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert marker.read_text() == "x"
    assert json.loads(completed.stdout.strip()) == {"state": "complete", "count": 1}
    assert "semantic-test-secret" not in completed.stderr
    assert _SemanticProvider.reached == 1


def test_semantic_ipc_rejects_message_above_bound_without_deadlock():
    payload = b"x" * (llm.MAX_SEMANTIC_IPC_BYTES + 1)
    scene = types.SimpleNamespace(index=1, text="scene")

    def decode():
        return llm._decode_semantic_ipc(payload, 0, (scene,), 3)

    plans, diagnostic, retryable = decode()
    assert plans is None
    assert diagnostic is llm.SemanticPlanDiagnostic.ipc_invalid
    assert not retryable


def test_semantic_worker_environment_excludes_parent_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy-user:proxy-secret@example.test")
    monkeypatch.setenv("AUTHORIZATION", "Bearer environment-secret")
    environment = llm._minimal_worker_environment()
    serialized = json.dumps(environment)
    assert "environment-secret" not in serialized
    assert "proxy-secret" not in serialized
    assert "OPENAI_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "AUTHORIZATION" not in environment


RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestScriptPromptOptions(unittest.TestCase):
    def test_normalize_text_response_removes_think_blocks(self):
        """
        reasoning 模型可能返回 `<think>...</think>`。脚本生成链路必须只保留
        最终正文，避免思考过程进入字幕和配音。
        """
        result = llm._normalize_text_response(
            "<think>\nI should reason here.\n</think>\n测试成功",
            "minimax",
        )

        self.assertEqual(result, "测试成功")

    def test_normalize_text_response_rejects_think_only_response(self):
        """
        如果模型只返回思考块而没有最终答案，应视为空内容，触发重试或明确错误。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning</think>", "minimax")

    def test_normalize_text_response_removes_unclosed_think_block(self):
        """
        某些网关可能因为截断只返回未闭合的 `<think>`。这种内容同样不能
        进入最终脚本；如果清理后没有正文，就应该按空响应处理。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning", "minimax")

    def test_build_script_prompt_appends_advanced_requirements(self):
        """
        高级文案要求只作为附加约束，不替换默认系统提示词。
        这样普通用户不配置时仍然走稳定默认规则，高级用户也能细化风格。
        """
        prompt = llm.build_script_prompt(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=3,
            video_script_prompt="语气轻松，面向程序员",
        )

        self.assertIn("# Role: Video Script Generator", prompt)
        self.assertIn("- video subject: 咖啡", prompt)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("- language: zh-CN", prompt)
        self.assertIn("# Additional User Requirements:", prompt)
        self.assertIn("语气轻松，面向程序员", prompt)

    def test_custom_system_prompt_keeps_runtime_context(self):
        """
        自定义 system prompt 会替换默认脚本规则，但视频主题、语言、段落数
        仍由服务层统一追加，避免高级用户漏写必要上下文。
        """
        prompt = llm.build_script_prompt(
            video_subject="露营",
            language="en",
            paragraph_number=2,
            custom_system_prompt="Only write cinematic narration.",
        )

        self.assertNotIn("# Role: Video Script Generator", prompt)
        self.assertIn("Only write cinematic narration.", prompt)
        self.assertIn("- video subject: 露营", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("- language: en", prompt)

    def test_generate_script_sends_custom_prompt_to_llm(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return "第一段。\n\n第二段。"

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_script(
                video_subject="咖啡",
                language="zh-CN",
                paragraph_number=2,
                video_script_prompt="开头更有悬念",
            )

        self.assertEqual(result, "第一段。\n\n第二段。")
        self.assertIn("- number of paragraphs: 2", captured["prompt"])
        self.assertIn("开头更有悬念", captured["prompt"])

    def test_generate_terms_can_request_script_ordered_keywords(self):
        """
        按文案顺序匹配素材依赖 LLM 返回有序关键词。这里不调用真实模型，
        只验证服务层会把“按脚本叙事顺序输出”的约束写入 prompt，避免
        后续素材下载虽然顺序化，但关键词仍然是全局无序主题词。
        """
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return '["opening city", "middle office", "final sunset"]'

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="First city. Then office. Finally sunset.",
                amount=3,
                match_script_order=True,
            )

        self.assertEqual(result, ["opening city", "middle office", "final sunset"])
        self.assertIn("chronological stock-video search terms", captured["prompt"])
        self.assertIn("same order as the script narration", captured["prompt"])

    def test_generate_terms_returns_empty_list_on_provider_error(self):
        """
        Provider 错误必须保持 generate_terms 的 List[str] 返回契约。

        非空的 ``Error: ...`` 字符串在 Python 中是真值；如果直接返回，任务层
        会把它当成有效关键词，素材下载层随后还可能逐字符发起搜索请求。
        """
        with patch.object(
            llm,
            "_generate_response",
            return_value="Error: invalid API key",
        ):
            result = llm.generate_terms(
                video_subject="startup story",
                video_script="A short startup story.",
            )

        self.assertEqual(result, [])
        self.assertIsInstance(result, list)

    def test_video_script_request_rejects_invalid_advanced_options(self):
        """
        API 请求模型需要限制高级 prompt 参数，避免外部调用绕过 WebUI
        传入异常段落数或超长提示词，导致模型成本和结果不可控。
        """
        with self.assertRaises(ValidationError):
            VideoScriptRequest(video_subject="咖啡", paragraph_number=0)

        with self.assertRaises(ValidationError):
            VideoScriptRequest(
                video_subject="咖啡",
                video_script_prompt="x" * (llm.MAX_SCRIPT_PROMPT_LENGTH + 1),
            )


class TestLLMConnection(unittest.TestCase):
    def test_connection_sends_one_minimal_request(self):
        """连接测试只发送一次固定最小请求，不触发脚本生成重试。"""
        with (
            patch.object(llm, "_generate_response", return_value="OK") as generate,
            patch.object(llm, "perf_counter", side_effect=[10.0, 10.25]),
        ):
            result = llm.test_connection()

        generate.assert_called_once_with(prompt="Reply with exactly: OK")
        self.assertEqual(result, (True, "", 0.25))

    def test_connection_returns_provider_error(self):
        """Provider 返回错误时应保留可诊断信息，并报告本次请求耗时。"""
        with (
            patch.object(
                llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(llm, "perf_counter", side_effect=[20.0, 20.5]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "invalid API key", 0.5))

    def test_connection_rejects_empty_response(self):
        """极端情况下的空响应应显示明确错误，而不是误报连接成功。"""
        with (
            patch.object(llm, "_generate_response", return_value=""),
            patch.object(llm, "perf_counter", side_effect=[30.0, 31.0]),
        ):
            result = llm.test_connection()

        self.assertEqual(result, (False, "LLM returned an empty response", 1.0))


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_current_default_model_names(self):
        """WebUI 与服务层必须共享同一组默认模型，避免展示值和请求值漂移。"""
        self.assertEqual(get_llm_provider("openai").default_model, "gpt-5.5")
        self.assertEqual(get_llm_provider("aimlapi").default_model, "openai/gpt-5-5")
        self.assertEqual(get_llm_provider("deepseek").default_model, "deepseek-v4-pro")
        self.assertEqual(
            get_llm_provider("modelscope").default_model, "ZhipuAI/GLM-5.2"
        )
        self.assertEqual(
            get_llm_provider("gemini").default_model, "gemini-3.1-pro-preview"
        )
        pollinations = get_llm_provider("pollinations")
        self.assertEqual(pollinations.default_model, "openai-fast")
        self.assertEqual(
            pollinations.default_base_url,
            "https://gen.pollinations.ai/v1",
        )
        self.assertTrue(pollinations.requires_api_key)
        self.assertEqual(pollinations.adapter, "openai_compatible")

    def test_provider_defaults_are_not_persisted_as_user_overrides(self):
        """默认值只用于运行和展示，只有不同值才应写入用户配置。"""
        self.assertEqual(
            normalize_provider_override("gpt-5.5", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("  gpt-5.5  ", "gpt-5.5"),
            "",
        )
        self.assertEqual(
            normalize_provider_override("gpt-5.6-custom", "gpt-5.5"),
            "gpt-5.6-custom",
        )

    def test_provider_registry_has_unique_stable_ids(self):
        """Registry 是 Provider 列表的唯一数据源，ID 必须唯一且默认项存在。"""
        provider_ids = [provider.provider_id for provider in LLM_PROVIDER_REGISTRY]

        self.assertEqual(len(provider_ids), len(set(provider_ids)))
        self.assertEqual(len(provider_ids), len(LLM_PROVIDERS))
        self.assertIn(DEFAULT_LLM_PROVIDER_ID, LLM_PROVIDERS)

    def test_provider_registry_preserves_product_group_order(self):
        """下拉顺序按推荐、原厂、聚合平台、本地部署和其它服务排列。"""
        self.assertEqual(
            [provider.provider_id for provider in LLM_PROVIDER_REGISTRY],
            [
                "moonshot",
                "openai",
                "gemini",
                "deepseek",
                "qwen",
                "azure",
                "volcengine",
                "grok",
                "minimax",
                "mimo",
                "cloudflare",
                "modelscope",
                "aihubmix",
                "aimlapi",
                "evolink",
                "ollama",
                "oneapi",
                "litellm",
                "groq",
                "pollinations",
            ],
        )
        self.assertEqual(
            get_llm_provider("gemini").default_label,
            "Google Gemini",
        )
        self.assertEqual(
            get_llm_provider("azure").default_label,
            "Microsoft Azure OpenAI",
        )

    def test_provider_registry_uses_conventional_locale_and_config_keys(self):
        """统一命名规则可避免 WebUI 为每个 Provider 增加硬编码映射。"""
        for provider in LLM_PROVIDER_REGISTRY:
            self.assertEqual(
                provider.label_key,
                f"llm_provider_label.{provider.provider_id}",
            )
            self.assertEqual(
                provider.tips_key,
                f"llm_provider_tips.{provider.provider_id}",
            )
            self.assertEqual(
                provider.config_key("api_key"),
                f"{provider.provider_id}_api_key",
            )

    def test_registry_replaces_deprecated_provider_models(self):
        """历史默认模型应自动迁移，避免升级后继续使用已移除的接入语义。"""
        cloudflare = get_llm_provider("cloudflare")
        gemini = get_llm_provider("gemini")

        self.assertEqual(
            cloudflare.resolve_model_name("@cf/meta/llama-3.1-8b-instruct"),
            "openai/gpt-4.1-mini",
        )
        self.assertEqual(
            gemini.resolve_model_name("gemini-pro"),
            "gemini-3.1-pro-preview",
        )
        self.assertEqual(
            cloudflare.resolve_model_name("anthropic/claude-sonnet-4-5"),
            "anthropic/claude-sonnet-4-5",
        )

        pollinations = get_llm_provider("pollinations")
        self.assertEqual(
            pollinations.resolve_model_name("default"),
            "openai-fast",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://text.pollinations.ai/openai"),
            "https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            pollinations.resolve_base_url("https://example.com/v1"),
            "https://example.com/v1",
        )

    def test_provider_tip_templates_accept_registry_defaults(self):
        """所有语言的 Provider 提示模板都必须能安全注入 Registry 默认值。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for locale_file in i18n_dir.glob("*.json"):
            translations = json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations.get(provider.tips_key, "")
                if not tips:
                    continue
                rendered = tips.format(
                    api_key_url=provider.api_key_url,
                    default_model=provider.default_model,
                    default_base_url=provider.default_base_url,
                    docker_hint="",
                    **{
                        f"default_{field.config_suffix}": field.default_value
                        for field in provider.extra_fields
                    },
                )
                self.assertNotIn("{default_model}", rendered)
                self.assertNotIn("{default_base_url}", rendered)

    def test_primary_provider_tips_use_consistent_structure(self):
        """中英文配置说明统一展示 API Key、Base URL 和模型名称。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        for language in ("zh", "en"):
            translations = json.loads(
                (i18n_dir / f"{language}.json").read_text(encoding="utf-8")
            )["Translation"]
            for provider in LLM_PROVIDER_REGISTRY:
                tips = translations[provider.tips_key]
                self.assertTrue(tips.startswith("##### "), provider.provider_id)
                self.assertIn("**API Key**", tips, provider.provider_id)
                self.assertIn("**Base Url**", tips, provider.provider_id)
                self.assertIn("**Model Name**", tips, provider.provider_id)

        zh_kimi_tips = json.loads((i18n_dir / "zh.json").read_text(encoding="utf-8"))[
            "Translation"
        ]["llm_provider_tips.moonshot"]
        self.assertIn("推荐理由：", zh_kimi_tips)
        self.assertIn("视频创作链路匹配", zh_kimi_tips)

    def test_required_api_key_providers_have_clickable_entry_points(self):
        """需要密钥的 Provider 必须提供统一申请入口，避免 WebUI 只给出文字。"""
        i18n_dir = Path(__file__).parent.parent.parent / "webui" / "i18n"
        locale_translations = {
            locale_file.stem: json.loads(locale_file.read_text(encoding="utf-8"))[
                "Translation"
            ]
            for locale_file in i18n_dir.glob("*.json")
        }

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.requires_api_key:
                self.assertTrue(provider.api_key_url, provider.provider_id)
                self.assertTrue(
                    provider.api_key_url.startswith("https://"),
                    provider.provider_id,
                )
                for language, translations in locale_translations.items():
                    tips_template = translations.get(provider.tips_key, "")
                    if not tips_template:
                        continue
                    tips = tips_template.format(
                        api_key_url=provider.api_key_url,
                        default_model=provider.default_model,
                        default_base_url=provider.default_base_url,
                        docker_hint="",
                        **{
                            f"default_{field.config_suffix}": field.default_value
                            for field in provider.extra_fields
                        },
                    )
                    api_key_line = next(
                        line for line in tips.splitlines() if "**API Key**" in line
                    )
                    self.assertIn("](", api_key_line, provider.provider_id)
                    self.assertIn(
                        f"]({provider.api_key_url})",
                        api_key_line,
                        f"{language}: {provider.provider_id}",
                    )

    def test_example_config_does_not_duplicate_registry_defaults(self):
        """示例配置只保存用户覆盖值，默认模型和地址由 Registry 唯一维护。"""
        config_path = Path(__file__).parent.parent.parent / "config.example.toml"
        app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.default_model:
                self.assertEqual(
                    app_config.get(provider.config_key("model_name"), ""),
                    "",
                    provider.provider_id,
                )
            if provider.default_base_url:
                self.assertEqual(
                    app_config.get(provider.config_key("base_url"), ""),
                    "",
                    provider.provider_id,
                )
            for field in provider.extra_fields:
                if field.default_value:
                    self.assertEqual(
                        app_config.get(provider.config_key(field.config_suffix), ""),
                        "",
                        provider.provider_id,
                    )

    def test_removed_ernie_provider_is_unsupported(self):
        """移除 ERNIE 后，遗留配置应返回明确错误，不再发起旧 OAuth 请求。"""
        config.app["llm_provider"] = "ernie"

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("unsupported llm provider", result)

    def test_pollinations_requires_api_key_before_request(self):
        """新统一 API 要求鉴权，缺少 Key 时不得发送匿名生成请求。"""
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "",
                "pollinations_base_url": "",
                "pollinations_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("api_key is not set", result)

    def test_pollinations_uses_unified_openai_compatible_api(self):
        """历史地址和模型名应自动迁移，并通过统一 Chat Completions API 调用。"""
        config.app.update(
            {
                "llm_provider": "pollinations",
                "pollinations_api_key": "pollinations-test-key",
                "pollinations_base_url": "https://text.pollinations.ai/openai/",
                "pollinations_model_name": "default",
            }
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\npollinations")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai-fast",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hellopollinations")

    def test_gemini_uses_google_genai_client(self):
        """Gemini 适配器应通过新版 SDK 的统一 Client 发起内容生成请求。"""
        config.app.update(
            {
                "llm_provider": "gemini",
                "gemini_api_key": "gemini-test-key",
                "gemini_base_url": "",
                "gemini_model_name": "gemini-test-model",
            }
        )
        captured = {}

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text="hello\ngemini")

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        with patch("google.genai.Client", FakeClient):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "hellogemini")
        self.assertEqual(
            captured["client_kwargs"],
            {"api_key": "gemini-test-key", "http_options": None},
        )
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(captured["contents"], "Say hello")
        self.assertEqual(captured["config"].max_output_tokens, 2048)
        self.assertTrue(captured["closed"])

    def test_cloudflare_requires_account_id_before_request(self):
        """Cloudflare 缺少 Account ID 时应在本地失败，不发送无效请求。"""
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "test-token",
                "cloudflare_account_id": "",
                "cloudflare_model_name": "",
            }
        )

        with patch.object(llm, "OpenAI") as openai_client:
            result = llm._generate_response("test")

        openai_client.assert_not_called()
        self.assertIn("account_id is not set", result)

    def test_cloudflare_uses_ai_gateway_openai_endpoint(self):
        """Cloudflare Provider 必须走 AI Gateway，不再调用 Workers AI 接口。"""
        config.app.update(
            {
                "llm_provider": "cloudflare",
                "cloudflare_api_key": "cloudflare-token",
                "cloudflare_account_id": "account-123",
                "cloudflare_gateway_id": "",
                "cloudflare_model_name": "",
            }
        )

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="gateway\nresponse")
                )
            ]
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return fake_response

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="cloudflare-token",
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
            ),
            default_headers={"cf-aig-gateway-id": "default"},
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "gatewayresponse")

    def _use_litellm_provider(self, model_name="openai/gpt-4o-mini"):
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = model_name

    def test_litellm_provider_returns_normalized_text(self):
        """
        验证 LiteLLM provider 的主路径不依赖真实网络和私有 API key。

        这里用 fake module 注入 `sys.modules`，直接覆盖动态 import 的
        `litellm.completion()`，确保测试稳定覆盖 `_generate_response()` 里的
        litellm 分支。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                kwargs["messages"], [{"role": "user", "content": "Say hello"}]
            )
            self.assertTrue(kwargs["drop_params"])
            message = types.SimpleNamespace(content="hello\nworld")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "helloworld")

    def test_litellm_provider_uses_registry_default_model(self):
        self._use_litellm_provider(model_name="")

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            message = types.SimpleNamespace(content="default model")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertEqual(result, "default model")

    def test_litellm_provider_handles_empty_response(self):
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[])
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty response", result)

    def test_litellm_provider_handles_empty_message(self):
        """
        某些 OpenAI-compatible 网关在内容过滤或安全拦截时会返回
        HTTP 200，但 `choices[0].message` 为 None。这里必须返回
        可诊断的错误，而不是抛出 AttributeError。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=None)]
            )
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty message", result)

    def test_sanitize_error_message_redacts_url_credentials_and_query_tokens(self):
        message = (
            "request failed for "
            "https://myuser:mypassword@proxy.example.com/v1/chat"
            "?api_key=secret-key&token=secret-token&safe=value"
        )

        result = llm._sanitize_error_message(message)

        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("api_key=***", result)
        self.assertIn("token=***", result)
        self.assertIn("safe=value", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-key", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_error_redacts_embedded_base_url_credentials(self):
        """
        自定义 OpenAI-compatible base_url 可能包含代理网关的 user:pass。
        SDK 抛错时常会把 URL 带回异常信息，这里验证最终返回给 WebUI/API 的
        `Error:` 文案不会泄露这些凭据。
        """
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-key"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"
        config.app["groq_base_url"] = (
            "https://myuser:mypassword@proxy.example.com/openai/v1"
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise RuntimeError(
                    "connection failed: "
                    "https://myuser:mypassword@proxy.example.com/openai/v1"
                    "?access_token=secret-token"
                )

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        with patch.object(llm, "OpenAI", return_value=fake_client):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("https://***:***@proxy.example.com", result)
        self.assertIn("access_token=***", result)
        self.assertNotIn("myuser", result)
        self.assertNotIn("mypassword", result)
        self.assertNotIn("secret-token", result)

    def test_openai_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = ""
        config.app["openai_base_url"] = "https://api.openai.com/v1"
        config.app["openai_model_name"] = "gpt-4o-mini"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def _use_qwen_provider(self):
        config.app["llm_provider"] = "qwen"
        config.app["qwen_api_key"] = "qwen-key"
        config.app["qwen_model_name"] = "qwen-max"

    def _patch_dashscope_generation(self, response):
        class FakeGenerationResponse(dict):
            pass

        fake_response = FakeGenerationResponse(response)
        fake_response.status_code = response.get("status_code", 200)
        fake_dashscope = types.SimpleNamespace(
            api_key="",
            Generation=types.SimpleNamespace(call=lambda **kwargs: fake_response),
        )
        fake_dashscope_response = types.SimpleNamespace(
            GenerationResponse=FakeGenerationResponse
        )

        return patch.dict(
            sys.modules,
            {
                "dashscope": fake_dashscope,
                "dashscope.api_entities": types.SimpleNamespace(),
                "dashscope.api_entities.dashscope_response": fake_dashscope_response,
            },
        )

    def test_qwen_provider_reads_chat_choices_content(self):
        """
        DashScope chat 模式会把文本放在 `output.choices[0].message.content`。
        这里覆盖 issue #966 报告的 `output.text is None` 场景，避免再次触发
        `'NoneType' object has no attribute 'replace'`。
        """
        self._use_qwen_provider()
        response = {
            "output": {
                "text": None,
                "choices": [{"message": {"content": "你好\n世界"}}],
            }
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "你好世界")

    def test_qwen_provider_falls_back_to_output_text(self):
        """保留旧 DashScope completion 响应结构的兼容路径。"""
        self._use_qwen_provider()
        response = {"output": {"text": "旧格式\n响应"}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "旧格式响应")

    def test_qwen_provider_reports_empty_text(self):
        """Qwen 空响应应返回可诊断错误，而不是底层 AttributeError。"""
        self._use_qwen_provider()
        response = {
            "output": {"text": None, "choices": [{"message": {"content": None}}]}
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty text content", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_provider_reports_empty_choices(self):
        """Qwen chat 响应 choices 为空时应返回明确错误。"""
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": []}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty choices", result)
        self.assertNotIn("NoneType", result)

    def test_aihubmix_provider_uses_openai_compatible_client(self):
        """
        AIHubMix 是 OpenAI-compatible 网关。这里用 fake OpenAI client
        验证独立 Provider 会使用 Registry 中的默认地址和模型，避免真实网络
        或私有 API Key 影响测试稳定性。
        """
        config.app["llm_provider"] = "aihubmix"
        config.app["aihubmix_api_key"] = "aihubmix-key"
        config.app["aihubmix_base_url"] = ""
        config.app["aihubmix_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naihubmix")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aihubmix-key",
            base_url="https://aihubmix.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloaihubmix")

    def test_aimlapi_provider_uses_openai_compatible_client(self):
        config.app["llm_provider"] = "aimlapi"
        config.app["aimlapi_api_key"] = "aimlapi-key"
        config.app["aimlapi_base_url"] = ""
        config.app["aimlapi_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naimlapi")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aimlapi-key",
            base_url="https://api.aimlapi.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "openai/gpt-5-5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloaimlapi")

    def test_evolink_provider_uses_openai_compatible_client(self):
        """
        EvoLink exposes OpenAI-compatible Chat Completions at direct.evolink.ai.
        The provider should keep its own default endpoint and model instead of
        requiring users to overload the generic OpenAI settings.
        """
        config.app["llm_provider"] = "evolink"
        config.app["evolink_api_key"] = "evolink-key"
        config.app["evolink_base_url"] = ""
        config.app["evolink_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nevolink")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="evolink-key",
            base_url="https://direct.evolink.ai/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloevolink")

    def test_volcengine_provider_uses_openai_compatible_client(self):
        """
        VolcEngine Ark 暴露 OpenAI-compatible Chat Completions。
        这里用 fake OpenAI client 覆盖 provider 默认地址和默认模型，
        避免真实网络或私有 API key 影响测试稳定性。
        """
        config.app["llm_provider"] = "volcengine"
        config.app["volcengine_api_key"] = "volcengine-key"
        config.app["volcengine_base_url"] = ""
        config.app["volcengine_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nvolcengine")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="volcengine-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "doubao-seed-2-1-turbo-260628",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hellovolcengine")

    def test_grok_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "grok"
        config.app["grok_api_key"] = ""
        config.app["grok_base_url"] = "https://api.x.ai/v1"
        config.app["grok_model_name"] = "grok-4.3"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_requires_api_key(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = ""
        config.app["groq_base_url"] = "https://api.groq.com/openai/v1"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_uses_default_base_url(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-test-key"
        config.app["groq_base_url"] = ""
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\ngroq")
                )
            ]
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(result, "hellogroq")

    def _use_ollama_provider(self, base_url=""):
        config.app["llm_provider"] = "ollama"
        config.app["ollama_api_key"] = ""
        config.app["ollama_base_url"] = base_url
        config.app["ollama_model_name"] = "llama3"

    def _assert_ollama_base_url(self, expected_base_url: str):
        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nollama")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="ollama",
            base_url=expected_base_url,
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloollama")

    def test_ollama_default_base_url_uses_localhost_outside_container(self):
        """
        普通本机运行时，Ollama 默认仍然使用 localhost，避免影响已有用户。
        """
        self._use_ollama_provider()

        with patch.object(config, "is_running_in_container", return_value=False):
            self._assert_ollama_base_url("http://localhost:11434/v1")

    def test_ollama_default_base_url_uses_host_gateway_inside_container(self):
        """
        容器内运行时，localhost 指向容器自身；默认改为 host.docker.internal，
        方便 Docker Desktop 用户访问宿主机上的 Ollama。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=True),
        ):
            self._assert_ollama_base_url("http://host.docker.internal:11434/v1")

    def test_ollama_default_base_url_falls_back_to_container_gateway(self):
        """
        原生 Linux Docker 里不一定能解析 host.docker.internal。此时使用容器
        默认网关作为兜底地址，比直接返回不可解析的 hostname 更稳。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=False),
            patch.object(
                config, "get_container_default_gateway_ip", return_value="172.17.0.1"
            ),
        ):
            self._assert_ollama_base_url("http://172.17.0.1:11434/v1")

    def test_ollama_explicit_base_url_takes_precedence(self):
        """
        用户手动配置的 ollama_base_url 优先级最高，不受容器检测影响。
        """
        self._use_ollama_provider(base_url="http://ollama:11434/v1")

        with patch.object(config, "is_running_in_container", return_value=True):
            self._assert_ollama_base_url("http://ollama:11434/v1")

    def test_mimo_provider_uses_openai_compatible_client(self):
        """
        MiMo 官方接口兼容 OpenAI Chat Completions 协议。这里用 fake OpenAI
        client 验证 provider 会使用 MiMo 独立配置和默认 base_url，不依赖
        真实网络或私有 API Key。
        """
        config.app["llm_provider"] = "mimo"
        config.app["mimo_api_key"] = "mimo-key"
        config.app["mimo_base_url"] = ""
        config.app["mimo_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nmimo")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hellomimo")

    def test_azure_provider_uses_azure_client_directly(self):
        """
        Azure OpenAI 的鉴权、endpoint 和 api-version 都由 AzureOpenAI 客户端处理。
        这个测试覆盖 issue #892：azure 分支必须直接调用 AzureOpenAI 创建的客户端，
        不能继续落入普通 OpenAI-compatible 分支，否则会丢失 Azure 专用请求配置。
        """
        config.app["llm_provider"] = "azure"
        config.app["azure_api_key"] = "azure-key"
        config.app["azure_base_url"] = "https://example.openai.azure.com"
        config.app["azure_model_name"] = "gpt-4o-mini"
        config.app["azure_api_version"] = "2024-02-15-preview"

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nazure")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "AzureOpenAI", return_value=fake_client) as azure_client,
            patch.object(llm, "OpenAI") as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        azure_client.assert_called_once_with(
            api_key="azure-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://example.openai.azure.com",
        )
        openai_client.assert_not_called()
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloazure")

    def test_unsupported_provider_returns_clear_error(self):
        config.app["llm_provider"] = "g" + "4f"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("unsupported llm provider", result)


class TestRuntimeEnvironmentDetection(unittest.TestCase):
    def test_container_detection_ignores_plain_linux_cgroup_file(self):
        """
        普通 Linux 也有 /proc/1/cgroup，不能因为文件存在就判定为容器。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text("0::/init.scope\n", encoding="utf-8")

            self.assertFalse(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_detection_accepts_dockerenv_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerenv_path = Path(tmp_dir) / ".dockerenv"
            dockerenv_path.write_text("", encoding="utf-8")

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(dockerenv_path),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(Path(tmp_dir) / "missing-cgroup"),
                )
            )

    def test_container_detection_accepts_cgroup_container_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text(
                "0::/system.slice/docker-abcdef.scope\n",
                encoding="utf-8",
            )

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_gateway_ip_decodes_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)),
                "172.17.0.1",
            )

    def test_container_gateway_ip_ignores_missing_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t0011AC0A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)), ""
            )


class TestSocialMetadata(unittest.TestCase):
    """通用短视频发布文案元数据生成。"""

    def test_build_prompt_auto_language_uses_source_language(self):
        """
        language 默认 auto 时，不应该固定成某个国家或语种，而是让模型
        跟随视频主题和脚本的语言，扩大 API 适用范围。
        """
        prompt = llm.build_social_metadata_prompt(
            video_subject="上海一日游",
            video_script="今天带你快速看完上海经典路线。",
            language="auto",
            platform="tiktok",
        )

        self.assertIn("TikTok", prompt)
        self.assertIn("Use the same language as the video subject and script", prompt)
        self.assertIn("上海一日游", prompt)
        self.assertIn("array of exactly 5 strings", prompt)

    def test_build_prompt_accepts_explicit_language(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="Coffee tips",
            language="en-US",
            platform="youtube_shorts",
        )

        self.assertIn("YouTube Shorts", prompt)
        self.assertIn('Write "title" and "caption" in this language: en-US', prompt)
        self.assertIn("array of exactly 3 strings", prompt)

    def test_unknown_platform_falls_back_to_tiktok(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x",
            platform="unsupported-platform",
        )

        self.assertIn("TikTok", prompt)

    def test_normalize_hashtags_from_string_dedupes_and_clamps(self):
        tags = llm._normalize_hashtags("#fyp fyp, trending #Trending viral", count=2)

        self.assertEqual(tags, ["#fyp", "#trending"])

    def test_normalize_hashtags_from_list_keeps_unicode_letters(self):
        tags = llm._normalize_hashtags(
            ["上海 旅行", "#việt nam", "  ", "@bad!chars"], count=5
        )

        self.assertEqual(tags, ["#上海旅行", "#việtnam", "#badchars"])

    def test_parse_social_metadata_recovers_embedded_json(self):
        raw = 'Sure: {"title":"T","caption":"C","hashtags":["#x"]} thanks'
        result = llm._parse_social_metadata(raw, "tiktok")

        self.assertEqual(result["title"], "T")
        self.assertEqual(result["caption"], "C")
        self.assertEqual(result["hashtags"], ["#x"])

    def test_parse_social_metadata_requires_title_or_caption(self):
        with self.assertRaises(ValueError):
            llm._parse_social_metadata('{"hashtags":["#x"]}', "tiktok")

    def test_generate_social_metadata_uses_llm_response(self):
        payload = (
            '{"title":"上海一日游","caption":"收藏这条路线，下次直接出发！",'
            '"hashtags":["#上海","#旅行","#shorts"]}'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_social_metadata(
                video_subject="上海一日游",
                video_script="今天带你快速看完上海经典路线。",
                language="zh-CN",
                platform="tiktok",
            )

        self.assertEqual(result["title"], "上海一日游")
        self.assertEqual(result["caption"], "收藏这条路线，下次直接出发！")
        self.assertEqual(result["hashtags"], ["#上海", "#旅行", "#shorts"])

    def test_generate_social_metadata_falls_back_to_generic_hashtags(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            result = llm.generate_social_metadata(
                video_subject="Coffee tips",
                video_script="Save these three coffee tips.",
                platform="instagram_reels",
            )

        self.assertEqual(result["title"], "Coffee tips")
        self.assertEqual(result["caption"], "Save these three coffee tips.")
        self.assertEqual(len(result["hashtags"]), 8)
        self.assertEqual(result["hashtags"][0], "#shorts")

    def test_request_model_defaults_to_auto_language_tiktok(self):
        body = VideoSocialMetadataRequest(video_subject="Test")

        self.assertEqual(body.language, "auto")
        self.assertEqual(body.platform, "tiktok")

    def test_request_model_rejects_oversized_social_metadata_fields(self):
        """
        外部 API 不能接受无限长的脚本和语言参数，否则会直接放大 LLM
        token 成本。schema 层先拦截，服务层再做内部调用兜底。
        """
        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x" * 501)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", video_script="x" * 8001)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", language="x" * 65)

    def test_build_prompt_clamps_direct_service_inputs(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x" * 600,
            video_script="y" * 9000,
            language="en",
        )

        self.assertIn("x" * llm.MAX_SOCIAL_SUBJECT_LENGTH, prompt)
        self.assertNotIn("x" * (llm.MAX_SOCIAL_SUBJECT_LENGTH + 1), prompt)
        self.assertIn("y" * llm.MAX_SOCIAL_SCRIPT_LENGTH, prompt)
        self.assertNotIn("y" * (llm.MAX_SOCIAL_SCRIPT_LENGTH + 1), prompt)

    def test_social_metadata_endpoint_response_shape(self):
        from fastapi.testclient import TestClient

        from app.asgi import app

        request_body = {
            "video_subject": "Tokyo coffee shops",
            "video_script": "Three quiet coffee shops for your next Tokyo morning.",
            "language": "en",
            "platform": "youtube_shorts",
        }
        llm_response = (
            '{"title":"3 Quiet Tokyo Coffee Shops",'
            '"caption":"Save these spots for your next Tokyo morning.",'
            '"hashtags":["#Tokyo","#Coffee","#Shorts"]}'
        )

        with patch.object(llm, "_generate_response", return_value=llm_response):
            response = TestClient(app).post(
                "/api/v1/social-metadata",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "3 Quiet Tokyo Coffee Shops",
                    "caption": "Save these spots for your next Tokyo morning.",
                    "hashtags": ["#Tokyo", "#Coffee", "#Shorts"],
                },
            },
        )


FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"
FOUNDRY_MODEL = "azure_ai/claude-sonnet-4-6"


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and FOUNDRY_KEY,
    "MPT_RUN_INTEGRATION_TESTS and ANTHROPIC_FOUNDRY_API_KEY not set",
)
class TestLiteLLMLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = FOUNDRY_MODEL
        os.environ["AZURE_AI_API_KEY"] = FOUNDRY_KEY
        os.environ["AZURE_AI_API_BASE"] = FOUNDRY_BASE

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_live_litellm_completion(self):
        result = llm._generate_response("What is 2+2? Reply with just the number.")

        self.assertNotIn("Error:", result)
        self.assertIn("4", result)


if __name__ == "__main__":
    unittest.main()
