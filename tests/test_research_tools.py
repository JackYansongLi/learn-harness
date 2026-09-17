import json
import subprocess
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import tools as rt
from agent import AgentError
from main import build_parent
from tests.helpers import ScriptedModel, answer, call


def make_pdf(path, texts):
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                ),
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(
        tmp_path / "new-paper.pdf", ["A different research paper", "Training uses 42 samples"]
    )


def test_pdf_is_user_input_not_alexnet(pdf, tmp_path):
    other = make_pdf(tmp_path / "second.pdf", ["Another architecture"])
    first = json.loads(rt.read_paper(pdf, "1,2"))
    second = json.loads(rt.read_paper(other, "1"))
    assert first["total_pages"] == 2
    assert "42 samples" in first["pages"][1]["text"]
    assert "Another architecture" in second["pages"][0]["text"]
    assert first["sha256"] != second["sha256"]
    assert json.loads(rt.search_paper(pdf, "TRAINING"))["hits"][0]["page"] == 2
    assert json.loads(rt.search_paper(other, "training"))["hits"] == []


@pytest.mark.parametrize("pages", ["0", "-1", "3", "1,2,1,2", "", "abc"])
def test_page_validation(pdf, pages):
    with pytest.raises(ValueError):
        rt.read_paper(pdf, pages)


def test_empty_search(pdf):
    with pytest.raises(ValueError):
        rt.search_paper(pdf, " ")


def test_invalid_and_scanned_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(AgentError, match="expected a PDF"):
        rt.paper_pages(bad)
    blank = make_pdf(tmp_path / "scan.pdf", [""])
    with pytest.raises(AgentError, match="OCR"):
        rt.paper_pages(blank)


def test_pdf_limit(tmp_path):
    pdf = tmp_path / "huge.pdf"
    with pdf.open("wb") as f:
        f.truncate(20_000_001)
    with pytest.raises(AgentError, match="20 MB"):
        rt.paper_pages(pdf)


def test_code_only_reads_designated_file(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("raise RuntimeError('must not execute')\n" + "x" * 17000)
    result = json.loads(rt.read_code(code))
    assert result["truncated"] and len(result["text"]) == 16000


def test_dataset_not_configured_is_not_missing():
    assert json.loads(rt.inspect_dataset(None))["status"] == "not_configured"


def test_dataset_layout(tmp_path):
    assert json.loads(rt.inspect_dataset(tmp_path / "absent"))["status"] == "missing_directory"
    for name in ["train/cat", "train/dog", "val/cat"]:
        (tmp_path / name).mkdir(parents=True)
    result = json.loads(rt.inspect_dataset(tmp_path))
    assert result["class_directories"] == {"train": 2, "val": 1}
    assert result["status"] == "layout_checked"


def test_workload_rounds_up_per_epoch():
    result = json.loads(rt.training_workload("5", "3", "2"))
    assert result["steps_per_epoch"] == 3
    assert result["total_steps"] == 9
    assert result["sample_visits"] == 15
    assert result["drop_last"] is False


@pytest.mark.parametrize(
    "values",
    [
        ("0", "1", "2"),
        ("3", "-1", "2"),
        ("3", "1", "NaN"),
        ("3.5", "1", "2"),
        ("1000000001", "1", "2"),
    ],
)
def test_workload_invalid(values):
    with pytest.raises(ValueError):
        rt.training_workload(*values)


def test_subprocess_uses_current_python_no_cpu_fallback(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout='{"status":"unavailable"}', returncode=1)

    monkeypatch.setattr(rt.subprocess, "run", run)
    result = json.loads(rt.run_probe("step", "mps"))
    assert result == {"status": "unavailable", "process_exit_code": 1}
    args, kwargs = calls[0]
    assert args == [rt.sys.executable, "-m", "tools", "step", "--device", "mps"]
    assert kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert kwargs["timeout"] == 180
    assert kwargs["cwd"] == rt.PROJECT


@pytest.mark.parametrize("action,device", [("shell", "mps"), ("step", "gpu"), ("step", "mps;ls")])
def test_probe_allowlist(action, device):
    with pytest.raises(ValueError):
        rt.run_probe(action, device)


def test_probe_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 180)

    monkeypatch.setattr(rt.subprocess, "run", timeout)
    with pytest.raises(AgentError, match="timed out"):
        rt.run_probe("step")


def test_probe_bad_output(monkeypatch):
    monkeypatch.setattr(
        rt.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=1)
    )
    with pytest.raises(AgentError, match="exit 1"):
        rt.run_probe("step")


def test_generic_registry_has_no_alexnet_experiment(main_agent_class, pdf):
    parent = build_parent(main_agent_class, None, pdf)
    assert set(parent.tools) == {"task"}
    assert set(parent.subagents["paper"].tools) == {"read_paper", "search_paper"}
    assert set(parent.subagents["environment"].tools) == {"inspect_environment"}
    assert set(parent.subagents) == {"paper", "environment"}
    assert parent.difficulty_tools is None
    read = parent.subagents["paper"].tools["read_paper"].handler(pages="1")
    assert "different research" in read


def test_explicit_experiment_and_code(main_agent_class, pdf, tmp_path):
    code = tmp_path / "custom.py"
    code.write_text("my implementation")
    parent = build_parent(main_agent_class, None, pdf, experiment="alexnet", code=code)
    assert set(parent.subagents["paper"].tools) == {
        "read_paper",
        "search_paper",
        "read_code",
        "inspect_model",
    }
    assert "run_model_check" in parent.subagents["environment"].tools
    assert "my implementation" in parent.subagents["paper"].tools["read_code"].handler()


@pytest.mark.exercise2
def test_three_experts_complete_generic_pdf_workflow(main_agent_class, pdf, monkeypatch):
    import main as app
    from tests.helpers import task

    monkeypatch.setattr(
        app, "run_probe", lambda action, device: '{"status":"ok","mps_available":true}'
    )
    handoff = "Paper uses 42 samples. Environment reports MPS available, no model experiment."
    model = ScriptedModel(
        answer(
            None,
            task("paper", "read the supplied PDF", "p"),
            task("environment", "inspect the machine", "e"),
        ),
        answer(None, call("read_paper", '{"pages":"1,2"}', "read")),
        answer("Paper uses 42 samples."),
        answer(None, call("inspect_environment", id="env")),
        answer("Environment reports MPS available, no model experiment."),
        answer(None, task("difficulty", handoff, "d")),
        answer(None, call("inspect_dataset", id="dataset")),
        answer("Need implementation and dataset."),
        answer("A paper investigation with known limitations."),
    )
    result = build_parent(main_agent_class, model, pdf, exercise=2).run("investigate my PDF")
    assert result == "A paper investigation with known limitations."
    assert len(model.requests) == 9
    assert model.requests[6]["messages"][1]["content"] == handoff
    assert "42 samples" in model.requests[2]["messages"][-1]["content"]
    assert json.loads(model.requests[7]["messages"][-1]["content"])["status"] == "not_configured"
    final = model.requests[-1]["messages"]
    assert [m["tool_call_id"] for m in final if m["role"] == "tool"] == ["p", "e", "d"]
    assert "different research paper" not in json.dumps(final)


def test_bundled_alexnet_case():
    from tools import ALEXNET_PDF

    assert ALEXNET_PDF.is_file(), "仓库缺少 examples/alexnet/paper.pdf"
    pages, _ = rt.paper_pages(ALEXNET_PDF)
    assert len(pages) == 9 and "Krizhevsky" in pages[0]
    assert "ImageNet" in pages[0]
    assert json.loads(rt.search_paper(ALEXNET_PDF, "GTX"))["hits"]


@pytest.mark.parametrize("device", ["auto", "cuda", "mps", "cpu"])
def test_device_bound_to_tools(main_agent_class, pdf, monkeypatch, device):
    import main as app

    calls = []

    def probe(action, requested):
        calls.append((action, requested))
        return "measured"

    monkeypatch.setattr(app, "run_probe", probe)
    parent = build_parent(main_agent_class, None, pdf, experiment="alexnet", device=device)
    env = parent.subagents["environment"].tools
    assert env["run_model_check"].parameters["properties"] == {}
    env["inspect_environment"].handler()
    env["run_model_check"].handler()
    assert calls == [("environment", device), ("step", device)]
