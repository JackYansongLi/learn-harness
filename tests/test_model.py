"""使用真的 OpenAI SDK + 假 HTTP 响应，检查线上请求格式，无需密钥。"""

import json

import httpx
import pytest
from openai import OpenAI

from agent import ModelOutputError, OpenAIModel, Tool, string_args
from tests.helpers import answer, call


def make_client(payload, requests):
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    return OpenAI(
        api_key="test-placeholder",
        base_url="https://example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def payload(message, reason="stop"):
    return {
        "id": "test-response",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
    }


def test_sdk_tool_call_round_trip(tmp_path):
    requests = []
    expected = answer(None, call("inspect_environment", id="sdk_call"))
    with make_client(payload(expected, "tool_calls"), requests) as client:
        model = OpenAIModel(client)
        messages = [{"role": "user", "content": "inspect environment"}]
        schemas = [Tool("inspect_environment", "inspect", string_args(), lambda: "ok").schema()]
        assert model.complete(messages, schemas) == expected
    body = json.loads(requests[0].content)
    assert body["model"] == "deepseek-flash"
    assert body["messages"] == messages
    assert body["tools"] == schemas
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 2400
    assert requests[0].url.path == "/v1/chat/completions"


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_truncated_response_is_not_success(reason):
    with make_client(payload(answer("unfinished"), reason), []) as client:
        with pytest.raises(ModelOutputError):
            OpenAIModel(client).complete([], [])


def test_no_tools_omits_tools_parameter():
    requests = []
    with make_client(payload(answer("ok")), requests) as client:
        assert OpenAIModel(client).complete([], []) == answer("ok")
    assert "tools" not in json.loads(requests[0].content)


def test_missing_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        OpenAIModel.from_env()
