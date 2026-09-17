"""作业：补全 Agent 的三个 TODO；API 和工具数据结构已提供。"""

import os
from dataclasses import dataclass
from typing import Callable, cast

from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

MAX_SUMMARY_CHARS = 2400


# 这三个类用来区分错误类型，错误消息的保存和显示沿用 RuntimeError。
# pass 表示不添加其他行为，这里已经写完，不是需要你补全的作业。
class AgentError(RuntimeError):
    """本课运行错误的共同父类。

    dispatch 执行工具时捕获 AgentError，也就能捕获下面两种子类错误，
    将它们转成 Error: 开头的工具结果，交给调用工具的 Agent 处理。
    """

    pass


class StepLimitExceeded(AgentError):
    """请求模型达到 max_turns 次，仍没有最终回答时，由 run 抛出。

    用它结束本次任务，避免 Agent 一直请求模型、调用工具而停不下来。
    """

    pass


class ModelOutputError(AgentError):
    """模型回复不能作为有效结果使用时抛出。

    run 检查没有工具请求时的回答是否为空；API 适配器检查回复是否
    被截断或被拒绝。检查失败就抛出这个错误，不把无效回复当作完成。
    """

    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]

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
        """TODO 1：检查工具名和 JSON 参数，调用 handler；约定的错误返回 Error: 文本。"""
        raise NotImplementedError("TODO 1: dispatch")

    def run(self, prompt: str) -> str:
        """TODO 2：补全 Agent loop；每轮请求模型一次，再判断继续还是结束。"""
        # 先在循环外新建本次任务的 messages，再补全下面每一轮的处理。
        for _ in range(self.max_turns):
            # 有工具请求：执行并保存结果，继续下一轮。
            # 没有工具请求：返回有效回答；空回答则报错。
            raise NotImplementedError("TODO 2: run")
        # 轮数用完仍没返回时，抛出 StepLimitExceeded。

    def spawn_subagent(self, agent_type: str, description: str) -> str:
        """TODO 3：取专家配置，创建新 Agent，只传任务，返回有长度上限的回答。"""
        raise NotImplementedError("TODO 3: spawn_subagent")


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
