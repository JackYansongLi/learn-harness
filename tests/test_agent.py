import json

import pytest

from agent import (
    MAX_SUMMARY_CHARS,
    AgentError,
    ModelOutputError,
    Specialist,
    StepLimitExceeded,
    Tool,
    string_args,
)
from tests.helpers import ScriptedModel, answer, call


def tools():
    return {"echo": Tool("echo", "return value", string_args("value"), lambda value: value)}


def parent(agent_class, model, **kwargs):
    specs = {
        "paper": Specialist("paper knowledge", tools()),
        "environment": Specialist(
            "environment knowledge",
            {
                "inspect": Tool("inspect", "", string_args(), lambda: "measured environment"),
            },
        ),
    }
    agent = agent_class(model, "parent system", {}, specialists=specs, **kwargs)
    agent.tools["task"] = Tool(
        "task", "", string_args("agent_type", "description"), agent.spawn_subagent
    )
    return agent


def task(kind, description, id="parent_task"):
    return call("task", json.dumps({"agent_type": kind, "description": description}), id)


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


@pytest.mark.exercise
def test_child_isolation_and_summary_only(agent_class):
    model = ScriptedModel(
        answer(None, task("paper", "read this paper")),
        answer(None, call("echo", '{"value":"CHILD_RAW_EVIDENCE"}', "child_read")),
        answer("brief conclusion"),
        answer("parent final"),
    )
    agent = parent(agent_class, model)
    assert agent.run("PARENT_PRIVATE_CONTEXT") == "parent final"
    assert model.requests[1]["messages"] == [
        {"role": "system", "content": "paper knowledge"},
        {"role": "user", "content": "read this paper"},
    ]
    assert {t["function"]["name"] for t in model.requests[1]["tools"]} == {"echo"}
    history = model.requests[3]["messages"]
    assert history[-1] == {
        "role": "tool",
        "tool_call_id": "parent_task",
        "content": "brief conclusion",
    }
    assert "CHILD_RAW_EVIDENCE" not in json.dumps(history)
    assert "child_read" not in json.dumps(history)
    assert "task" in agent.tools
    assert "CHILD_RAW_EVIDENCE" in json.dumps(model.requests[2])


@pytest.mark.exercise
def test_experts_use_their_own_knowledge_and_tools(agent_class):
    model = ScriptedModel(answer("paper result"), answer("environment result"))
    agent = parent(agent_class, model)
    agent.spawn_subagent("paper", "check method")
    agent.spawn_subagent("environment", "check computer")
    second = model.requests[1]
    assert second["messages"] == [
        {"role": "system", "content": "environment knowledge"},
        {"role": "user", "content": "check computer"},
    ]
    assert {t["function"]["name"] for t in second["tools"]} == {"inspect"}
    assert "paper result" not in json.dumps(second)


@pytest.mark.exercise
def test_repeated_expert_calls_start_fresh(agent_class):
    model = ScriptedModel(answer("first result"), answer("second result"))
    agent = parent(agent_class, model)
    agent.spawn_subagent("paper", "first task")
    agent.spawn_subagent("paper", "second task")
    assert len(model.requests[1]["messages"]) == 2
    assert model.requests[1]["messages"][1]["content"] == "second task"
    assert "first result" not in json.dumps(model.requests[1])


@pytest.mark.exercise
def test_child_rejects_fabricated_task_even_if_registry_has_task(agent_class):
    model = ScriptedModel(answer(None, task("paper", "again")), answer("stop"))
    agent = parent(agent_class, model)
    agent.specialists["paper"].tools["task"] = agent.tools["task"]
    assert agent.spawn_subagent("paper", "one task") == "stop"
    assert model.requests[1]["messages"][-1]["content"].startswith("Error:")
    assert "task" in agent.specialists["paper"].tools  # 不能修改注册表原件
    assert all(t["function"]["name"] != "task" for t in model.requests[0]["tools"])


@pytest.mark.exercise
def test_child_cannot_use_another_experts_tool(agent_class):
    model = ScriptedModel(answer(None, call("inspect")), answer("cannot inspect"))
    agent = parent(agent_class, model)
    agent.spawn_subagent("paper", "inspect environment")
    assert model.requests[1]["messages"][-1]["content"].startswith("Error:")


def test_unknown_expert(agent_class):
    agent = parent(agent_class, ScriptedModel())
    with pytest.raises(ValueError):
        agent.spawn_subagent("unknown", "go")
    assert agent.dispatch(task("unknown", "go")).startswith("Error:")


@pytest.mark.exercise
def test_child_budget_failure_is_reported_to_parent(agent_class):
    model = ScriptedModel(
        answer(None, task("paper", "loop")),
        answer(None, call("echo", '{"value":"loop"}')),
        answer(None, call("echo", '{"value":"loop"}')),
        answer("child could not finish"),
    )
    assert parent(agent_class, model, max_turns=2).run("go") == "child could not finish"
    assert model.requests[-1]["messages"][-1]["content"].startswith("Error:")
    assert len(model.requests) == 4


@pytest.mark.exercise
def test_summary_is_bounded(agent_class):
    agent = parent(agent_class, ScriptedModel(answer("x" * 3000)))
    assert agent.spawn_subagent("paper", "go") == "x" * MAX_SUMMARY_CHARS + "\n[summary truncated]"


def test_programming_errors_are_not_hidden(agent_class):
    def broken():
        raise NotImplementedError("student forgot implementation")

    agent = agent_class(None, "s", {"broken": Tool("broken", "", string_args(), broken)})
    with pytest.raises(NotImplementedError):
        agent.dispatch(call("broken"))
