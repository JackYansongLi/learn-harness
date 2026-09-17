# 作业：补全论文助手的三个方法

你要把教程中的调查过程接起来：Main Agent接收用户任务，调用论文助手和环境助手，把它们的结果交给难度助手，再汇总报告。

课堂用 AlexNet 验收。PDF 读取工具和模型实验分别提供，代码仍须能读取用户指定的其他 PDF。

只补全 `agent.py` 中 Agent 的三个 TODO，可以添加 import 和小型辅助函数。同文件的 API 适配器和工具数据结构已经提供，不需要修改；也不要修改测试和工具函数。知识说明、PDF 工具、PyTorch 实验和 API 适配器已提供。

## 第一题：执行工具请求

实现 `dispatch(call)`。下面是模型为论文助手返回的一次工具请求。论文助手的工具表中已经注册了 `read_paper`，执行后应返回第 1 页的读取结果：

```python
call = {
    "id": "read_1",
    "type": "function",
    "function": {"name": "read_paper", "arguments": '{"pages":"1"}'},
}
```

`function.arguments` 是 JSON 字符串。解析后，用 `function.name` 查当前实例的 `self.tools`，把解析出的参数传给该工具的 `handler`。不同助手的工具表不同，所以不能绕过 `self.tools` 去调用其他助手的函数。

本课参数均为必填字符串：JSON 必须是 object，键恰好匹配 `parameters.properties`，值为字符串。无参数工具传 `{}`，不用实现通用 JSON Schema 校验器。

## 第二题：把工具结果发回模型

实现 `run(prompt)`。每次调用都新建一个 messages 列表，先放入 system 和 user 两条消息，内容分别是 `self.system` 和本次 prompt。

将当前工具表中每个 Tool 的 `schema()` 结果组成 schemas 列表，再调用 `self.model.complete(messages, schemas)`。它返回一条 assistant 消息：可能含有 `tool_calls`，也可能直接给出回答。

先保存这条 assistant 消息。有 `tool_calls` 时逐一调用 dispatch，再为每个结果追加一条 tool 消息。第一题的请求对应：

```python
{"role": "tool", "tool_call_id": "read_1", "content": "刚刚读取的结果"}
```

`tool_call_id` 必须对应请求的 id，模型才能知道这条结果回答的是哪次调用。保存完本轮所有工具结果，再带着更新后的 messages 请求模型。直到它不再请求工具，才返回有效的回答文本。

## 第三题：按类型创建专家

Main Agent把 task 的 handler 绑定到了自己的 `spawn_subagent` 方法。因此模型请求 task 后，Main Agent的 dispatch 会调用这个方法。

实现 `spawn_subagent(agent_type, description)`：

1. 在 `self.specialists` 里找专家配置，未知类型抛 `ValueError`。
2. 用本文件的 `Agent` 创建 child，复用 model，使用专家自己的 system 和 tools。
3. 复制工具表并移除 task，不修改原表，不给 child 传 specialists。
4. 沿用 max_turns，只把 description 传给 `child.run()`。
5. 返回摘要。超过 `MAX_SUMMARY_CHARS` 时截断，追加 `\n[summary truncated]`。

child.run 会按第二题的规则新建消息列表，使用专家的 system 和本次 description。Main Agent和所有Subagent因此分别保存自己的消息。再次调用同一类专家，也必须创建新实例、从新任务开始。

## 错误约定

| 情况                                                | 怎么处理                               |
| --------------------------------------------------- | -------------------------------------- |
| 未知工具、JSON 或参数错误                           | 返回 `Error:` 开头的字符串             |
| 工具抛出 ValueError、TypeError、OSError、AgentError | 转成 `Error:` 文本                     |
| 方法没写完、其他程序错误                            | 正常抛出，不用 `except Exception` 吞掉 |
| 没有工具请求，回答为空                              | 抛 `ModelOutputError`                  |
| max_turns 次模型调用后仍没有最终答案                | 抛 `StepLimitExceeded`                 |

常量和异常类在 `agent.py`。

## 先检查代码，再用真实模型验收

```bash
uv run pytest tests/test_agent.py -k 'dispatch or arguments' -q
uv run pytest tests/test_agent.py -k 'plain or multiple or turn or empty' -q
uv run pytest tests/test_agent.py -k 'child or expert or summary' -q
uv run pytest -q
```

pytest 用预设的模型回复检查你的代码，包括工具回传、轮数上限、消息隔离、专家配置和工具分配，不会请求真实 API。测试还会生成两份内容不同的 PDF，检查工具是否读取了传入的文件。

完成后配置 `.env`，用 AlexNet 做在线验收：

```bash
uv run --extra ml python -m tests.smoke
```

默认自动选择可用设备，也可以在 smoke 命令末尾加 `--device cuda`、`--device mps` 或 `--device cpu`。三种设备都可以完成作业。

在线验收检查三位专家是否被调用、是否读过 PDF 和实现、是否在选定设备上完成实验、是否检查数据目录并计算训练工作量。验收会核对模型参数、输入和输出的实际设备，不会把 CPU 实验当成 GPU 实验。调用过程保存在 `output/smoke.json`，报告在 `output/report.md`。这些文件留在本地。

自动验收检查执行过程。模型对论文的解释仍需核对，尤其不要把“随机数据跑通一步”写成“复现了论文成绩”。

## 提交

提交 `agent.py`、测试结果和 AlexNet 调查报告，再用几句话回答：

1. `task.handler` 执行时 self 是谁？进入 `child.run()` 后呢？
2. 难度助手看不到前两位的消息，Main Agent应在 description 中传什么？
3. 报告里哪条结论来自论文，哪条来自本机实测？分别指出证据。
4. 本机单步实验通过后，距离复现论文指标还缺哪些条件？

公开仓库暂不提供参考答案。使用同一套测试逐项检查自己的实现。
