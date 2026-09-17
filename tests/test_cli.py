"""入口重组后，确认默认论文、实验与学生实现的选择没有串用。"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import main
from tools import ALEXNET_PDF


@pytest.mark.parametrize("exercise", [1, 2])
@pytest.mark.parametrize("custom_pdf", [False, True])
def test_cli_selects_paper_and_experiment(tmp_path, monkeypatch, custom_pdf, exercise):
    chosen = {}
    api = SimpleNamespace(client=nullcontext(), receipts=[])
    monkeypatch.setattr(main.OpenAIModel, "from_env", lambda: api)

    def build(cls, model, pdf, **kwargs):
        chosen.update(cls=cls, pdf=pdf, **kwargs)
        return SimpleNamespace(run=lambda prompt: "test report")

    monkeypatch.setattr(main, "build_parent", build)
    output = tmp_path / "report.md"
    args = [
        "--output",
        str(output),
        "--device",
        "cpu",
        "--no-progress",
        "--exercise",
        str(exercise),
    ]
    if custom_pdf:
        pdf = tmp_path / "different.pdf"
        pdf.write_bytes(b"%PDF-test")
        args += ["--pdf", str(pdf)]
    main.main(args)
    assert chosen["cls"] is main.MainAgent
    assert chosen["exercise"] == exercise
    assert chosen["device"] == "cpu"
    assert chosen["pdf"] == (pdf if custom_pdf else ALEXNET_PDF)
    assert chosen["experiment"] == (None if custom_pdf else "alexnet")
    assert output.read_text() == "test report\n"


def test_trace_prints_all_tool_results_without_changing_messages():
    from copy import deepcopy
    from io import StringIO

    from rich.console import Console

    from tests.helpers import ScriptedModel, answer

    messages = [
        {"role": "system", "content": "论文 Subagent"},
        {"role": "user", "content": "read"},
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "a", "content": "first evidence"},
        {"role": "tool", "tool_call_id": "b", "content": "second evidence"},
    ]
    original = deepcopy(messages)
    stream = StringIO()
    with main.ProgressModel(
        ScriptedModel(answer("done")), trace=True, console=Console(file=stream)
    ) as progress:
        assert progress.complete(messages, []) == answer("done")
    assert messages == original
    assert "first evidence" in stream.getvalue()
    assert "second evidence" in stream.getvalue()
