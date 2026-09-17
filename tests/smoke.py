"""AlexNet 在线验收：真实 API、真实 PDF、真实本机实验。运行后在本地保留证据。"""

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError

from agent import AgentError, OpenAIModel
from main import MAIN_SYSTEM, QUESTION, MainAgent, ProgressModel, build_parent
from tools import ALEXNET_PDF, DEVICE_CHOICES, select_device


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(deepcopy({"messages": messages, "tools": tools}))
        return self.model.complete(messages, tools)


def verify(requests, result, requested_device="auto", exercise=2):
    parent = [r for r in requests if r["messages"][0]["content"].startswith(MAIN_SYSTEM)]
    children = [r for r in requests if not r["messages"][0]["content"].startswith(MAIN_SYSTEM)]
    assert parent and children and result.strip(), "没有完整的 Main Agent、Subagent 调用"
    assert all({t["function"]["name"] for t in r["tools"]} == {"task"} for r in parent)
    assert all("task" not in {t["function"]["name"] for t in r["tools"]} for r in children)
    history = parent[-1]["messages"]
    tasks = [c for m in history for c in m.get("tool_calls", [])]
    args = [json.loads(c["function"]["arguments"]) for c in tasks]
    expected_agents = {"paper", "environment"} | ({"difficulty"} if exercise == 2 else set())
    assert {a["agent_type"] for a in args} == expected_agents
    starts = [r["messages"][1]["content"] for r in children if len(r["messages"]) == 2]
    assert starts == [a["description"] for a in args], "Subagent没有从独立任务开始"
    results = {m["tool_call_id"]: m["content"] for m in history if m["role"] == "tool"}
    assert set(results) == {c["id"] for c in tasks}, "task 返回结果不匹配"
    assert not any(v.startswith("Error:") for v in results.values()), "有子任务失败"
    completed = set()
    for message in history:
        for c in message.get("tool_calls", []):
            a = json.loads(c["function"]["arguments"])
            if a["agent_type"] == "difficulty":
                assert {"paper", "environment"} <= completed, "难度判断早于证据收集"
        if message["role"] == "tool":
            a = next(a for c, a in zip(tasks, args) if c["id"] == message["tool_call_id"])
            completed.add(a["agent_type"])
    calls = {}
    evidence = {}
    for request in children:
        for message in request["messages"]:
            for c in message.get("tool_calls", []):
                calls[c["id"]] = c
            if message["role"] == "tool":
                evidence[message["tool_call_id"]] = message["content"]
    assert not set(calls) & set(results), "Subagent的工具记录进入了Main Agent消息"
    names = {c["function"]["name"] for c in calls.values()}
    required_tools = {
        "read_paper",
        "inspect_model",
        "inspect_environment",
        "run_model_check",
    }
    if exercise == 2:
        required_tools |= {"inspect_dataset", "training_workload"}
    assert required_tools <= names, "有实际检查被跳过"

    def outputs(name):
        try:
            return [
                json.loads(evidence[id]) for id, c in calls.items() if c["function"]["name"] == name
            ]
        except (KeyError, ValueError) as exc:
            raise AssertionError(f"{name} 没有返回有效 JSON 结果") from exc

    environments = outputs("inspect_environment")
    assert environments and all(e["status"] == "ok" for e in environments), "环境检测未通过"
    env = environments[-1]
    expected = select_device(requested_device, env["cuda_available"], env["mps_available"])
    assert env["requested_device"] == requested_device and env["selected_device"] == expected
    steps = outputs("run_model_check")
    assert steps and all(s["status"] == "passed" for s in steps), "本机模型实验未通过"
    for step in steps:
        assert step["requested_device"] == requested_device and step["selected_device"] == expected
        assert all(
            step[field].split(":")[0] == expected
            for field in ("device", "parameter_device", "input_device", "target_device")
        ), "实际设备不符"
        assert step["weights_updated"] and step["gradients_finite"]
        assert not step["cpu_fallback"] and step["process_exit_code"] == 0


def main():
    parser = argparse.ArgumentParser(description="AlexNet 在线验收，会消耗 DeepSeek 额度")
    parser.add_argument("--exercise", type=int, choices=[1, 2], default=1)
    parser.add_argument("--impl", choices=["exercise", "solution"], default="exercise")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args()
    if args.impl == "solution":
        from solution.agent import MainAgent as implementation
    else:
        implementation = MainAgent
    load_dotenv(override=False)
    try:
        api = OpenAIModel.from_env()
        with api.client, ProgressModel(api, trace=True, exercise=args.exercise) as progress:
            recorder = RecordingModel(progress)
            result = build_parent(
                implementation,
                recorder,
                ALEXNET_PDF,
                experiment="alexnet",
                device=args.device,
                exercise=args.exercise,
            ).run(
                QUESTION + "\n本次为课堂验收，请在任务说明中明确以下实际调用要求："
                "paper 用 read_paper 和 inspect_model 核对论文及实现，查出数据量、epoch、batch。"
                "environment 用 inspect_environment 和 run_model_check 实测，均无需参数。"
                + (
                    "收到两份回答后，把证据传给 difficulty，并要求它调用 inspect_dataset 和 "
                    "training_workload。工具失败就如实报告。"
                    if args.exercise == 2
                    else ""
                )
            )
    except APIError as exc:
        parser.exit(1, f"API 失败：{type(exc).__name__}，检查密钥、余额、网络和模型名。\n")
    except (
        AgentError,
        ValueError,
        OSError,
        NotImplementedError,
        AssertionError,
    ) as exc:
        parser.exit(1, f"验收未通过：{exc}\n")
    check_error = None
    try:
        verify(recorder.requests, result, args.device, args.exercise)
    except AssertionError as exc:
        check_error = str(exc) or "验收条件未满足"
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "implementation": args.impl,
        "exercise": args.exercise,
        "requested_model": api.model,
        "requested_device": args.device,
        "checks": "failed" if check_error else "passed",
        "check_error": check_error,
        "result": result,
        "receipts": api.receipts,
        "requests": recorder.requests,
    }
    output = Path("output")
    output.mkdir(exist_ok=True)
    (output / "smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(result + "\n", encoding="utf-8")
    if check_error:
        parser.exit(1, f"验收未通过：{check_error}；本次记录在 output/smoke.json\n")
    print(f"验收通过：{len(api.receipts)} 次 API 调用；记录在 output/smoke.json")


if __name__ == "__main__":
    main()
