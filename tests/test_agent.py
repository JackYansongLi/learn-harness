import pytest

from agent import (
    AgentError,
    ModelOutputError,
    StepLimitExceeded,
    Tool,
    string_args,
)
from tests.helpers import ScriptedModel, answer, call


def tools():
    return {"echo": Tool("echo", "return value", string_args("value"), lambda value: value)}


def test_plain_answer(agent_class):
    model = ScriptedModel(answer("done"))
    assert agent_class(model, "system", {}).run("question") == "done"
    assert model.requests[0] == {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "tools": [],
    }


def test_dispatch(agent_class):
    assert (
        agent_class(None, "s", tools()).dispatch(call("echo", '{"value":"evidence"}')) == "evidence"
    )


@pytest.mark.parametrize(
    "args",
    [
        "{",
        "[]",
        "null",
        '"text"',
        "{}",
        '{"value":4}',
        '{"value":null}',
        '{"value":false}',
        '{"value":"x","extra":"y"}',
    ],
)
def test_invalid_arguments_do_not_invoke_handler(agent_class, args):
    invoked = []
    tool = Tool("echo", "", string_args("value"), lambda **kw: invoked.append(kw))
    assert agent_class(None, "s", {"echo": tool}).dispatch(call("echo", args)).startswith("Error:")
    assert invoked == []


def test_unknown_tool(agent_class):
    result = agent_class(None, "s", {}).dispatch(call("task", '{"description":"recurse"}'))
    assert result.startswith("Error:") and "task" in result


@pytest.mark.parametrize("error", [ValueError, TypeError, OSError, AgentError])
def test_expected_tool_errors_are_returned(agent_class, error):
    def fail():
        raise error("bad input")

    agent = agent_class(None, "s", {"fail": Tool("fail", "", string_args(), fail)})
    assert agent.dispatch(call("fail")).startswith("Error:")


def test_multiple_calls_round_trip(agent_class):
    first = answer(
        "checking", call("echo", '{"value":"A"}', "a"), call("echo", '{"value":"B"}', "b")
    )
    model = ScriptedModel(first, answer("A and B"))
    assert agent_class(model, "s", tools()).run("read both") == "A and B"
    assert model.requests[1]["messages"][2:] == [
        first,
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ]
    assert model.requests[0]["tools"] == [t.schema() for t in tools().values()]


def test_unknown_tool_does_not_abort_remaining_calls(agent_class):
    model = ScriptedModel(
        answer(None, call("bad", id="bad"), call("echo", '{"value":"ok"}', "ok")),
        answer("recovered"),
    )
    assert agent_class(model, "s", tools()).run("go") == "recovered"
    results = model.requests[-1]["messages"][-2:]
    assert results[0]["tool_call_id"] == "bad"
    assert results[0]["content"].startswith("Error:")
    assert results[1] == {"role": "tool", "tool_call_id": "ok", "content": "ok"}


@pytest.mark.parametrize("content", [None, "", "  "])
def test_empty_final_is_failure(agent_class, content):
    with pytest.raises(ModelOutputError):
        agent_class(ScriptedModel(answer(content)), "s", {}).run("go")


def test_turn_budget_never_returns_intermediate_text(agent_class):
    model = ScriptedModel(
        *[answer("still working", call("echo", '{"value":"x"}')) for _ in range(2)]
    )
    with pytest.raises(StepLimitExceeded):
        agent_class(model, "s", tools(), max_turns=2).run("go")
    assert len(model.requests) == 2


def test_last_allowed_turn_may_finish(agent_class):
    model = ScriptedModel(answer("done"))
    assert agent_class(model, "s", {}, max_turns=1).run("go") == "done"


def test_separate_runs_have_fresh_context(agent_class):
    model = ScriptedModel(answer("first"), answer("second"))
    agent = agent_class(model, "s", {})
    agent.run("old request")
    agent.run("new request")
    assert model.requests[1]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "new request"},
    ]


def test_programming_errors_are_not_hidden(agent_class):
    def broken():
        raise NotImplementedError("student forgot implementation")

    agent = agent_class(None, "s", {"broken": Tool("broken", "", string_args(), broken)})
    with pytest.raises(NotImplementedError):
        agent.dispatch(call("broken"))
