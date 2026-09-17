from io import StringIO

import pytest
from rich.console import Console

from main import STAGES, ProgressModel
from tests.helpers import ScriptedModel, answer, call


def messages(who):
    return [
        {"role": "system", "content": who + "\nknowledge"},
        {"role": "user", "content": "task"},
    ]


def display(model, disabled=False):
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=110)
    return ProgressModel(model, console=console, disabled=disabled), stream


def test_progress_counts_completed_stages_once():
    ui, stream = display(ScriptedModel(*[answer("done") for _ in range(5)]))
    with ui:
        ui.complete(messages("论文 Subagent"), [])
        ui.complete(messages("论文 Subagent"), [])  # 补查不会重复增加进度
        assert ui.progress.tasks[0].completed == 1
        for who in ("环境 Subagent", "复现难度 Subagent", "Main Agent"):
            ui.complete(messages(who), [])
    assert ui.finished == STAGES
    assert ui.progress.tasks[0].completed == 4
    assert ui.calls == 5
    assert "调查结束" in stream.getvalue()


def test_pending_tools_do_not_complete_a_stage():
    response = answer(None, call("run_model_check"))
    ui, _ = display(ScriptedModel(response, answer("measured")))
    with ui:
        assert ui.complete(messages("环境 Subagent"), []) == response
        assert ui.progress.tasks[0].completed == 0
        assert "执行 run_model_check" in ui.progress.tasks[0].description
        ui.complete(messages("环境 Subagent"), [])
        assert ui.progress.tasks[0].completed == 1


def test_waiting_status_is_set_before_network_call():
    class Model:
        def complete(self, messages, tools):
            assert "等待模型" in ui.progress.tasks[0].description
            assert "第 1 次请求" in ui.progress.tasks[0].description
            return answer("done")

    ui, _ = display(Model())
    with ui:
        ui.complete(messages("论文 Subagent"), [])


def test_failure_stops_progress_and_propagates():
    class Model:
        def complete(self, messages, tools):
            raise RuntimeError("network failed")

    ui, stream = display(Model())
    with pytest.raises(RuntimeError, match="network failed"):
        with ui:
            ui.complete(messages("论文 Subagent"), [])
    assert not ui.finished
    assert "运行中断" in stream.getvalue()
    assert not ui.progress.live.is_started


def test_disabled_progress_is_silent():
    ui, stream = display(ScriptedModel(answer("done")), disabled=True)
    with ui:
        assert ui.complete(messages("Main Agent"), []) == answer("done")
    assert stream.getvalue() == ""


def test_empty_answer_does_not_complete_a_stage():
    ui, _ = display(ScriptedModel(answer(None)))
    with ui:
        ui.complete(messages("论文 Subagent"), [])
    assert not ui.finished


def test_first_exercise_has_three_stages():
    ui = ProgressModel(
        ScriptedModel(*[answer("done") for _ in range(3)]), disabled=True, exercise=1
    )
    with ui:
        for who in ("论文 Subagent", "环境 Subagent", "Main Agent"):
            ui.complete(messages(who), [])
    assert ui.progress.tasks[0].total == 3
    assert ui.progress.tasks[0].completed == 3
    assert ui.finished == ui.stages
    assert ui.progress.tasks[0].description == "调查结束"
