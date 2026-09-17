"""两题分别验收。用预设模型回复检查 Python 调用，不依赖模型碰巧答对。"""

import json
from uuid import uuid4

import pytest

from agent import Agent, StepLimitExceeded, Tool, string_args
from main import MainAgent, build_parent
from tests.helpers import ScriptedModel, answer, call, task
from tools import ALEXNET_PDF


def test_two_provided_subagents_work_before_assignment_is_completed(monkeypatch):
    """第一题还没写，也能单独运行已提供的论文、环境助手。"""
    import main

    probes = []
    monkeypatch.setattr(main, "run_probe", lambda action, device: probes.append(action) or "CPU")
    model = ScriptedModel(
        answer(None, call("read_paper", '{"pages":"1"}')),
        answer("paper conclusion"),
        answer(None, call("inspect_environment")),
        answer("environment conclusion"),
    )
    parent = build_parent(MainAgent, model, ALEXNET_PDF)
    assert parent.subagents["paper"].run("read") == "paper conclusion"
    assert "Krizhevsky" in model.requests[1]["messages"][-1]["content"]
    assert parent.subagents["environment"].run("inspect") == "environment conclusion"
    assert probes == ["environment"]
    assert model.requests[2]["messages"][1]["content"] == "inspect"
    assert parent.difficulty_tools is None


@pytest.mark.exercise1
@pytest.mark.parametrize("separate_rounds", [False, True])
@pytest.mark.parametrize("order", [("paper", "environment"), ("environment", "paper")])
def test_main_loop_runs_both_subagents_and_returns_their_actual_answers(
    main_agent_class, separate_rounds, order, monkeypatch
):
    import main

    token = uuid4().hex
    tool_outputs = {"paper": "RAW_PAPER_" + token, "environment": "RAW_ENV_" + token}
    conclusions = {name: name + "_CONCLUSION_" + token for name in order}
    jobs = {name: "TASK_FOR_" + name + token for name in order}
    executed = []

    def read(pdf, pages):
        executed.append(("paper", pdf, pages))
        return tool_outputs["paper"]

    def inspect(action, device):
        executed.append(("environment", action, device))
        return tool_outputs["environment"]

    monkeypatch.setattr(main, "read_paper", read)
    monkeypatch.setattr(main, "run_probe", inspect)
    replies = []
    if not separate_rounds:
        replies.append(answer(None, *[task(name, jobs[name], name) for name in order]))
    for name in order:
        if separate_rounds:
            replies.append(answer(None, task(name, jobs[name], name)))
        request = (
            call("read_paper", '{"pages":"2"}', "child_p")
            if name == "paper"
            else call("inspect_environment", id="child_e")
        )
        replies.extend([answer(None, request), answer(conclusions[name])])
    replies.append(answer("FINAL_" + token))
    model = ScriptedModel(*replies)
    parent = build_parent(main_agent_class, model, ALEXNET_PDF, device="cpu", exercise=1)
    parent.run_difficulty = lambda _: pytest.fail("第一题不应调用未完成的难度助手")
    assert parent.run("PARENT_PRIVATE_" + token) == "FINAL_" + token
    assert [row[0] for row in executed] == list(order)
    assert ("paper", ALEXNET_PDF, "2") in executed
    assert ("environment", "environment", "cpu") in executed
    history = model.requests[-1]["messages"]
    results = [message for message in history if message["role"] == "tool"]
    assert results == [
        {"role": "tool", "tool_call_id": name, "content": conclusions[name]} for name in order
    ]
    assert all(raw not in json.dumps(history) for raw in tool_outputs.values())
    children = [r for r in model.requests if r["messages"][0]["content"] != parent.system]
    assert len(children) == 4
    for index, name in enumerate(order):
        first, second = children[2 * index : 2 * index + 2]
        assert len(first["messages"]) == 2
        assert first["messages"][1]["content"] == jobs[name]
        assert "PARENT_PRIVATE_" not in json.dumps(first)
        assert second["messages"][-1]["content"] == tool_outputs[name]
        assert "task" not in {t["function"]["name"] for t in first["tools"]}


@pytest.mark.exercise1
def test_main_loop_keeps_error_result_and_executes_next_request(main_agent_class):
    model = ScriptedModel(
        answer(None, task("missing", "unknown", "bad"), task("paper", "read", "good")),
        answer("paper result"),
        answer("report with error"),
    )
    parent = build_parent(main_agent_class, model, ALEXNET_PDF)
    assert parent.run("investigate") == "report with error"
    results = model.requests[-1]["messages"][-2:]
    assert results[0]["tool_call_id"] == "bad"
    assert results[0]["content"].startswith("Error:")
    assert results[1] == {"role": "tool", "tool_call_id": "good", "content": "paper result"}


@pytest.mark.exercise1
def test_main_loop_repeated_tasks_start_fresh(main_agent_class):
    model = ScriptedModel(
        answer(None, task("paper", "first", "p1")),
        answer("first conclusion"),
        answer(None, task("paper", "second", "p2")),
        answer("second conclusion"),
        answer("final"),
    )
    parent = build_parent(main_agent_class, model, ALEXNET_PDF)
    assert parent.run("read twice") == "final"
    assert len(model.requests[3]["messages"]) == 2
    assert model.requests[3]["messages"][1]["content"] == "second"
    assert "first conclusion" not in json.dumps(model.requests[3])


@pytest.mark.exercise1
def test_main_loop_obeys_turn_limit(main_agent_class):
    model = ScriptedModel(answer(None, task("paper", "read")), answer("paper done"))
    parent = build_parent(main_agent_class, model, ALEXNET_PDF, max_turns=1)
    with pytest.raises(StepLimitExceeded):
        parent.run("investigate")
    assert len(model.requests) == 2  # 主助手一轮、子助手一轮，不能把两者混算。


def difficulty_tools(events, token):
    def dataset():
        events.append(("dataset",))
        return "DATASET_" + token

    def workload(samples, epochs, batch_size):
        events.append(("workload", samples, epochs, batch_size))
        return "WORKLOAD_" + token

    return {
        "inspect_dataset": Tool("inspect_dataset", "inspect", string_args(), dataset),
        "training_workload": Tool(
            "training_workload",
            "calculate",
            string_args("samples", "epochs", "batch_size"),
            workload,
        ),
    }


@pytest.mark.exercise2
@pytest.mark.parametrize("samples,epochs,batch", [("42", "3", "8"), ("17", "2", "4")])
def test_new_subagent_calls_its_tools_and_returns_model_answer(
    main_agent_class, samples, epochs, batch
):
    token = uuid4().hex
    events = []
    tools = difficulty_tools(events, token)
    handoff = f"论文使用 {samples} 个样本，{epochs} 轮，batch={batch}。本机 CPU 单步通过。"
    params = json.dumps({"samples": samples, "epochs": epochs, "batch_size": batch})
    model = ScriptedModel(
        answer(None, call("inspect_dataset", id="dataset")),
        answer(None, call("training_workload", params, "work")),
        answer("ASSESSMENT_" + token),
    )
    parent = main_agent_class(model, {}, difficulty_tools=tools)
    assert parent.run_difficulty(handoff) == "ASSESSMENT_" + token
    assert events == [("dataset",), ("workload", samples, epochs, batch)]
    first = model.requests[0]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"].splitlines()[0] == "复现难度助手"
    assert len(first["messages"][0]["content"].splitlines()) > 1
    assert first["messages"][1] == {"role": "user", "content": handoff}
    assert len(first["messages"]) == 2
    assert {t["function"]["name"] for t in first["tools"]} == set(tools)
    assert model.requests[1]["messages"][-1]["content"] == "DATASET_" + token
    assert model.requests[2]["messages"][-1]["content"] == "WORKLOAD_" + token
    assert set(parent.difficulty_tools) == set(tools)


@pytest.mark.exercise2
def test_new_subagent_does_not_share_messages_between_tasks(main_agent_class):
    model = ScriptedModel(answer("old result"), answer("new result"))
    parent = main_agent_class(model, {}, difficulty_tools=difficulty_tools([], "token"))
    assert parent.run_difficulty("old task") == "old result"
    assert parent.run_difficulty("new task") == "new result"
    assert len(model.requests[1]["messages"]) == 2
    assert model.requests[1]["messages"][1]["content"] == "new task"
    assert "old task" not in json.dumps(model.requests[1])
    assert "old result" not in json.dumps(model.requests[1])


@pytest.mark.exercise2
def test_new_subagent_uses_requested_turn_limit(main_agent_class):
    model = ScriptedModel(answer(None, call("inspect_dataset")), answer("too late"))
    parent = main_agent_class(
        model, {}, difficulty_tools=difficulty_tools([], "token"), max_turns=1
    )
    with pytest.raises(StepLimitExceeded):
        parent.run_difficulty("inspect")
    assert len(model.requests) == 1


@pytest.mark.exercise2
def test_new_subagent_cannot_call_parent_or_paper_tools(main_agent_class):
    model = ScriptedModel(
        answer(None, call("task", id="nested"), call("read_paper", '{"pages":"1"}', "read")),
        answer("need more evidence"),
    )
    parent = main_agent_class(model, {}, difficulty_tools=difficulty_tools([], "token"))
    assert parent.run_difficulty("assess") == "need more evidence"
    assert all(m["content"].startswith("Error:") for m in model.requests[1]["messages"][-2:])


def test_second_exercise_registers_difficulty_without_running_it():
    parent = build_parent(MainAgent, None, ALEXNET_PDF, exercise=2)
    allowed = parent.tools["task"].parameters["properties"]["agent_type"]["enum"]
    assert allowed == ["paper", "environment", "difficulty"]
    assert set(parent.difficulty_tools) == {"inspect_dataset", "training_workload"}
    assert all(isinstance(child, Agent) for child in parent.subagents.values())
