from copy import deepcopy


def call(name, arguments="{}", id="call_1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def answer(content=None, *calls):
    result = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = list(calls)
    return result


class ScriptedModel:
    """按剧本返回响应；保存快照，防止后续 append 掩盖当时的错误。"""

    def __init__(self, *answers):
        self.answers = iter(answers)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(deepcopy({"messages": messages, "tools": tools}))
        return deepcopy(next(self.answers))
