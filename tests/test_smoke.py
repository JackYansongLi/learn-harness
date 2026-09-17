import json
from copy import deepcopy

import pytest

import main as app
from tests.helpers import ScriptedModel, answer, call
from tests.smoke import RecordingModel, verify
from tests.test_agent import task
from tests.test_research_tools import make_pdf
from tools import select_device


@pytest.fixture(
    params=[
        ("auto", True, False),
        ("auto", False, True),
        ("auto", False, False),
        ("cuda", True, False),
        ("mps", False, True),
        ("cpu", False, True),
    ]
)
def recording(agent_class, tmp_path, monkeypatch, request):
    requested, cuda, mps = request.param
    selected = select_device(requested, cuda, mps)
    pdf = make_pdf(tmp_path / "test.pdf", ["Test paper for workflow validation"])

    def fake_probe(action, device="auto"):
        if action == "step":
            return json.dumps(
                {
                    "status": "passed",
                    "requested_device": requested,
                    "selected_device": selected,
                    "device": selected,
                    "parameter_device": selected,
                    "input_device": selected,
                    "target_device": selected,
                    "gradients_finite": True,
                    "weights_updated": True,
                    "cpu_fallback": False,
                    "process_exit_code": 0,
                }
            )
        return json.dumps(
            {
                "status": "ok",
                "cuda_available": cuda,
                "mps_available": mps,
                "requested_device": requested,
                "selected_device": selected,
            }
        )

    monkeypatch.setattr(app, "run_probe", fake_probe)
    model = RecordingModel(
        ScriptedModel(
            answer(
                None,
                task("paper", "read paper and code", "p"),
                task("environment", "inspect environment and run step", "e"),
            ),
            answer(
                None, call("read_paper", '{"pages":"1"}', "read"), call("inspect_model", id="code")
            ),
            answer("paper summary"),
            answer(
                None,
                call("inspect_environment", id="env"),
                call("run_model_check", id="step"),
            ),
            answer("environment summary"),
            answer(None, task("difficulty", "paper summary and environment summary", "d")),
            answer(
                None,
                call("inspect_dataset", id="dataset"),
                call("training_workload", '{"samples":"5","epochs":"3","batch_size":"2"}', "work"),
            ),
            answer("difficulty summary"),
            answer("final report"),
        )
    )
    result = app.build_parent(agent_class, model, pdf, experiment="alexnet", device=requested).run(
        "test investigation"
    )
    return model.requests, result, requested


def test_online_verifier_accepts_complete_evidence(recording):
    verify(*recording)


@pytest.mark.parametrize(
    "mutation",
    [
        "skipped_tool",
        "wrong_device",
        "failed",
        "no_update",
        "early_difficulty",
        "wrong_parameter",
        "wrong_request",
    ],
)
def test_online_verifier_rejects_false_success(recording, mutation):
    requests, result, requested = deepcopy(recording)
    for request in requests:
        for message in request["messages"]:
            if mutation == "skipped_tool":
                for c in message.get("tool_calls", []):
                    if c["function"]["name"] == "run_model_check":
                        c["function"]["name"] = "pretend_check"
            elif message.get("tool_call_id") == "step":
                data = json.loads(message["content"])
                if mutation == "wrong_device":
                    data["device"] = "mps" if data["device"] == "cpu" else "cpu"
                elif mutation == "wrong_parameter":
                    data["parameter_device"] = "mps" if data["parameter_device"] == "cpu" else "cpu"
                elif mutation == "wrong_request":
                    data["requested_device"] = "auto" if requested != "auto" else "cpu"
                elif mutation == "failed":
                    data["status"] = "failed"
                elif mutation == "no_update":
                    data["weights_updated"] = False
                message["content"] = json.dumps(data)
    if mutation == "early_difficulty":
        history = requests[-1]["messages"]
        # 一开始就生成难度任务，description 不可能包含尚未返回的工具结果。
        history[2]["tool_calls"].append(history[5]["tool_calls"].pop())
    with pytest.raises(AssertionError):
        verify(requests, result, requested)
