"""作业：在 spawn_subagent 中创建子助手，并调用它的 run。其余代码已提供。"""

import json
import os
from dataclasses import dataclass
from typing import Callable, cast

from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

MAX_SUMMARY_CHARS = 2400


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]  # 保存执行函数；调用 handler(...) 时才真正执行工具

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class Specialist:
    system: str
    tools: dict[str, Tool]


def string_args(*names: str) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": list(names),
        "additionalProperties": False,
    }


class Agent:
    def __init__(
        self,
        model,
        system: str,
        tools: dict[str, Tool],
        *,
        specialists: dict[str, Specialist] | None = None,
        max_turns: int = 8,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.model = model
        self.system = system
        self.tools = dict(tools)
        self.max_turns = max_turns
        self.specialists = dict(specialists or {})

    def dispatch(self, call: dict) -> str:
        """只执行当前实例注册的工具；参数错误作为 tool 消息返回。"""
        name = call["function"]["name"]
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool: {name}"
        try:
            args = json.loads(call["function"]["arguments"])
            # 本课所有参数均为必填字符串，拒绝多参、缺参和非字符串。
            if not isinstance(args, dict) or set(args) != set(tool.parameters["properties"]):
                raise ValueError("arguments must match tool parameters")
            if any(not isinstance(value, str) for value in args.values()):
                raise ValueError("arguments must be strings")
            return tool.handler(**args)
        except (ValueError, TypeError, OSError, AgentError) as exc:
            return f"Error: {exc}"

    def run(self, prompt: str) -> str:
        """一次 run 是一次新任务；messages 只属于这次函数调用。"""
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        for _ in range(self.max_turns):
            answer = self.model.complete(messages, [tool.schema() for tool in self.tools.values()])
            messages.append(answer)
            calls = answer.get("tool_calls") or []
            if not calls:
                text = answer.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ModelOutputError("empty final answer")
                return text
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self.dispatch(call),
                    }
                )
        raise StepLimitExceeded(f"stopped after {self.max_turns} model calls without final answer")

    def spawn_subagent(self, agent_type: str, description: str) -> str:
        """作业：创建子助手，让它完成 description，再把回答交回主助手。"""
        if agent_type not in self.specialists:
            raise ValueError(f"unknown specialist: {agent_type}")
        spec = self.specialists[agent_type]
        # 子助手只使用自己的工具；不给 task，避免它继续创建下一层助手。
        child_tools = {  # noqa: F841 — TODO 1 创建 child 时使用
            name: tool for name, tool in spec.tools.items() if name != "task"
        }

        # TODO 1：用 Agent(...) 创建 child。共用 self.model，使用 spec.system、
        # child_tools 和 self.max_turns；不传主助手的 specialists。
        child = None
        if child is None:
            raise NotImplementedError("TODO 1: 创建子助手")

        # TODO 2：调用 child 的 run，只传 description，把返回的回答存进 summary。
        summary = None
        if summary is None:
            raise NotImplementedError("TODO 2: 运行子助手")

        # 回答长度限制已提供，不属于作业。
        if len(summary) > MAX_SUMMARY_CHARS:
            return summary[:MAX_SUMMARY_CHARS] + "\n[summary truncated]"
        return summary


# 以下是已提供的运行检查，无需修改。
class AgentError(RuntimeError):
    """运行错误的共同类型，便于执行工具时统一处理。"""


class StepLimitExceeded(AgentError):
    """请求模型的次数已达上限，仍未得到最终回答。"""


class ModelOutputError(AgentError):
    """模型回答为空、被截断或被拒绝。"""


class OpenAIModel:
    def __init__(self, client: OpenAI, model: str = "deepseek-flash"):
        self.client = client
        self.model = model
        self.receipts: list[dict] = []

    @classmethod
    def from_env(cls):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("请在 .env 或环境变量中设置 DEEPSEEK_API_KEY")
        return cls(
            OpenAI(
                api_key=key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout=45.0,
                max_retries=0,
            ),
            os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        # 教学代码使用普通字典；在 SDK 入口标明它们遵循的消息、工具格式。
        response = self.client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools) if tools else omit,
            tool_choice="auto" if tools else omit,
            max_tokens=2400,
            extra_body={"thinking": {"type": "disabled"}},
        )
        choice = response.choices[0]
        self.receipts.append(
            {
                "id": response.id,
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise ModelOutputError(f"model stopped with {choice.finish_reason}")
        message = choice.message
        if message.refusal:
            raise ModelOutputError("model refused the request")
        result: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                call.model_dump(exclude_none=True) for call in message.tool_calls
            ]
        return result
