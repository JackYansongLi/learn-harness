"""已提供的普通 Agent：请求模型、执行工具，再把结果交回模型。无需修改。"""

import json
import os
from dataclasses import dataclass
from typing import Callable, cast

from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]  # 保存执行函数；调用 handler(...) 时才真正执行工具

    def schema(self) -> dict:
        """生成发给模型的工具说明，返回符合接口格式的字典。

        说明中包含工具名称、用途和参数格式。Agent 的 run() 每轮请求
        模型前都会调用这个方法，让模型知道能使用哪些工具、怎样填写参数。

        handler 保存的是本地 Python 函数，不会放进这份说明里。
        模型返回工具请求后，才由 dispatch() 找到并执行那个函数；
        调用 schema() 本身不会执行工具。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def string_args(*names: str) -> dict:
    """为一组必填的字符串参数生成格式说明，供 Tool 使用。

    names 是参数名，例如 string_args("agent_type", "description")。
    返回的字典规定：这些参数都必须提供，每个值都应是字符串，
    不能增加其他参数。不传名字时，生成的是一个不接受参数的工具说明。

    这里只生成说明，交给模型阅读；模型实际传回的参数仍要由
    Agent.dispatch() 检查，不能因为提供了说明就假定它一定填写正确。
    """
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
        max_turns: int = 8,
    ):
        """保存一个 Agent 的模型、工作说明和工具，准备接收任务。

        model 需要提供 complete(messages, tools) 方法，不同 Agent 可以
        共用它。system 是工作说明，run() 会把它放在消息列表的开头。
        tools 将工具名对应到 Tool 对象；这里复制这张表，各项仍指向
        原来的 Tool 对象。此 Agent 执行请求时只从自己的表中查找工具。

        max_turns 限制每次 run() 最多请求模型多少轮，必须至少为 1，
        否则抛出 ValueError。它不限制一次模型回复中包含多少个工具请求。
        创建对象时不会请求模型；调用 run(prompt) 后才开始执行任务。
        """
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.model = model
        self.system = system
        self.tools = dict(tools)
        self.max_turns = max_turns

    def dispatch(self, call: dict) -> str:
        """执行模型提出的一条工具请求，把工具结果作为字符串返回。

        call 是回复中 tool_calls 列表的一项。先用 function.name 在
        self.tools 中找工具，再把 function.arguments 中的 JSON 字符串
        读成参数字典。本课要求参数名称全部匹配，且每个参数值都是字符串；
        检查通过后，才调用工具保存的 handler 函数。

        Main Agent 继承这个方法时，self 仍指 Main Agent，因此它的
        task 请求会调用 call_subagent()。普通 Subagent 则执行自己的
        文件读取、环境检查等工具。

        未知工具、参数错误，以及下面 except 列出的运行错误，会变成
        Error: 开头的字符串；其他异常继续向外抛出。这个方法不修改
        消息列表，调用它的 run() 负责把返回值写成一条 tool 消息。
        """
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
        """执行 prompt 中的一项任务，返回模型最后给出的文字回答。

        本课的论文、环境和复现难度 Subagent 使用这个循环。每次调用
        都用 system 和 prompt 新建 messages，不自动继承 Main Agent
        的对话，也不保留上一次 run() 的消息。

        每轮把当前消息和工具说明发给模型。如果回复带有工具请求，
        就按顺序调用 dispatch()，把各条结果连同请求 id 写回消息列表，
        再进入下一轮。没有工具请求且文字回答非空时，才返回这段文字。

        返回值不包含内部消息列表。Main Agent 等待 Subagent 运行结束后，
        接收到的就是这份回答。空回答会抛出 ModelOutputError；请求模型
        达到 max_turns 轮仍未结束时，抛出 StepLimitExceeded。
        """
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


# 以下是已提供的运行检查，无需修改。
class AgentError(RuntimeError):
    """运行错误的共同类型，便于执行工具时统一处理。"""


class StepLimitExceeded(AgentError):
    """请求模型的次数已达上限，仍未得到最终回答。"""


class ModelOutputError(AgentError):
    """模型回答为空、被截断或被拒绝。"""


class OpenAIModel:
    def __init__(self, client: OpenAI, model: str = "deepseek-flash"):
        """保存 API 客户端和模型名称，供 complete() 发送请求。

        client 使用 OpenAI SDK 的接口格式，具体服务地址由客户端配置；
        本课默认连接 DeepSeek。model 是请求中使用的模型名称。
        不同 Agent 可以共用这个对象，各自的消息列表由各自的 run() 管理。

        receipts 用来记录请求 id、结束原因和 token 用量，供运行和验收
        查看。这里仅初始化这些字段，不发送请求；客户端由 main() 关闭。
        """
        self.client = client
        self.model = model
        self.receipts: list[dict] = []

    @classmethod
    def from_env(cls):
        """读取环境变量，创建并返回本课使用的模型调用对象。

        DEEPSEEK_API_KEY 必须提供，缺少时抛出 ValueError。
        DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL 可分别覆盖接口地址和模型名；
        未设置时使用代码中给出的 DeepSeek 默认值。

        这个方法不读取 .env 文件，入口 main() 会先调用 load_dotenv()。
        创建的客户端把请求超时设为 45 秒，并关闭 SDK 自动重试。
        此时还没有 API 请求，真正的网络调用发生在 complete() 中。
        """
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
        """请求一次模型，把 SDK 响应整理成 Agent 循环使用的字典。

        messages 是当前任务的消息记录；tools 是 Tool.schema() 生成的
        工具说明列表，为空时不向接口传工具相关参数。这里不执行工具，
        也不向 messages 追加内容，这些工作由调用它的 run() 完成。

        返回值包含 role="assistant" 和 content；模型请求工具时，还会
        包含 tool_calls 列表，其中 arguments 仍是 JSON 字符串。
        content 可以为空，是否能作为最终回答由 run() 再判断。

        收到响应后先记录请求 id、结束原因和用量。回复被截断、被拒绝，
        或以其他不支持的原因结束时，抛出 ModelOutputError；API 请求
        本身的异常继续向外传，由入口程序处理。
        """
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
