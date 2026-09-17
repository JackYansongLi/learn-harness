# 在 Agent loop 里调用另一个 Agent

前面我们说过，Agent 开发就是在 loop 里加条件：每次收到模型回复，都要判断接下来执行什么、要不要继续。

这节课已经提供普通 Agent 的循环和工具调用。你只补上两处代码：创建另一个 Agent，调用它的 `run()`。负责分配任务的叫 Main Agent，被调用来完成一项任务的叫 Subagent。两者都执行同一个 `run()` 方法里的循环。

我们要写的是论文调查助手。你给它一份论文 PDF，它读论文、检查你的电脑，再告诉你还需要准备什么。测试时使用仓库中的 AlexNet 论文，程序也要能接收其他论文的 PDF。

## 1. 循环就在 agent.py 的 run() 里

打开 `agent.py`，找到 `Agent.run()`。其中的 `for` 就是本课的 Agent loop，每一轮请求一次模型。`max_turns` 限制最多请求多少次，避免程序一直运行下去。

模型回复后，先看它有没有请求工具：

- 有工具请求：执行工具，保存结果，再请求模型。
- 没有工具请求：检查回答是否为空；有答案就返回，空回答就报错。

这两个判断决定了循环继续还是结束。`run()` 已经写好，先顺着下面的步骤读一遍：

```text
新建本次任务的消息列表
最多循环 max_turns 次：
    请求模型，保存它的回复
    如果回复中有工具请求：
        逐个执行工具，保存每个工具的结果
        进入下一轮，把这些结果发给模型
    否则：
        如果回答为空，报错
        返回回答，结束本次 run
循环次数用完还没返回，就报错
```

```mermaid
flowchart TD
    A[新建消息列表，放入工作说明和任务] --> B{还有可用轮数？}
    B -->|有| C[请求模型，保存回复]
    B -->|没有| X[报错：超过轮数上限]
    C --> D{模型请求了工具？}
    D -->|有| E[逐个执行工具，保存结果]
    E --> B
    D -->|没有| F{回答为空？}
    F -->|是| Y[报错：没有有效回答]
    F -->|否| G[返回回答，结束循环]
```

注意图中回到循环开头的箭头。执行完工具还没有完成任务：模型需要看到工具结果，才能决定继续查资料，还是给出答案。

## 2. 先走一遍读取论文的循环

论文保存在本地。我们已经写好了 `read_paper` 读取函数，模型只需指定页码，函数就会返回这些页的文字。你不用实现 PDF 解析，只需把工具请求交给这个函数，再把结果交回模型。

例如，论文助手可能这样完成任务：

| 轮次 | 模型回复 | 你的程序做什么 |
| --- | --- | --- |
| 第 1 轮 | 请求 read_paper，读取第 1 页 | 执行读取，保存第 1 页文字，继续循环 |
| 第 2 轮 | 请求 read_paper，再读其他页 | 执行读取，保存结果，继续循环 |
| 第 3 轮 | 给出论文摘要，没有工具请求 | 返回摘要，结束循环 |

这是示意过程，实际读哪些页、读几轮由模型决定。程序负责的是同一件事：有工具请求就执行，没有工具请求就检查回答并结束。

要让下一轮看到前面读过的内容，需要一个 `messages` 列表。它先保存工作说明和任务，之后再追加模型回复和工具结果。`run()` 每轮把当前列表发给模型，模型才能接着已有结果往下查。

### 先看一次普通函数调用

读取函数已经在 `tools.py` 里写好。先不考虑模型，程序直接读第 1 页时，写法是：

```python
from pathlib import Path
from tools import read_paper

pdf = Path("examples/alexnet/paper.pdf")
result = read_paper(pdf, "1")
```

`read_paper` 是函数名，括号里的 pdf 和 "1" 是交给它的文件路径和页码。执行后，result 保存返回的文字及页码信息。所谓“读取工具”，实际做的就是这次函数调用。

模型不会直接执行这行代码。它回复的是工具名 `read_paper` 和页码 "1"，你的程序需要把这两个信息变成上面的函数调用。

### 先存下函数，需要时再调用

文件路径在程序启动时就确定了，模型只需选页码。可以把固定路径写进一个小函数，让它只接收页码：

```python
def read_selected_pages(pages):
    return read_paper(pdf, pages)
```

下面两行看着相似，做的事却不同：

```python
handler = read_selected_pages  # 没有括号：存下这个函数，暂时不执行
result = handler("1")  # 加上括号：执行存下的函数，读取第 1 页
```

**handler 在这里就是一个保存函数的名字。** 它保存的不是文字 `"read_selected_pages"`，也不是读出来的结果，而是稍后要调用的函数。本项目的每条工具记录都有一个 handler 字段，用它保存该工具实际要执行的函数。

`main.py` 里用 `lambda pages: read_paper(pdf, pages)` 写这个小函数，作用与上面的 read_selected_pages 相同。你可以先按上面的两行来理解“存下函数”和“执行函数”。

### 按工具名找到函数并执行

助手可能有多个工具，所以程序用一个字典保存它们：名字是查找入口，对应的记录里存着说明、参数要求和执行函数。例如，`self.tools["read_paper"]` 就是读取工具的记录。这里 self 指当前助手，self.tools 是它自己的工具字典。

收到模型的读取请求后，程序需要做：

```text
取出模型给的工具名 read_paper
  → 在当前助手的 tools 中找到这条记录
  → 检查页码参数是否合法
  → 调用这条记录的 handler，传入页码
  → 把读取结果交回 run，继续下一轮
```

**这几步合在一起，就是已经提供的 `dispatch()`，可以把它读成“执行工具”。** dispatch 这个方法负责查找、检查和调用；handler 这个字段保存被调用的函数。两者都只是这份代码里的名字，不需要另外学习一种 Agent 技术。

```mermaid
flowchart LR
    A[模型：读第 1 页] --> B[dispatch：找到读取工具，检查参数]
    B --> C[调用工具的 handler：执行读取函数]
    C --> D[返回论文文字]
    D --> E[run 保存结果，继续循环]
```

工具记录还保存了 name、description、parameters，分别是名称、用途和参数要求。`schema()` 把这三项整理成发给模型的说明。模型靠说明选择工具，本地程序靠 handler 执行函数。

因此，`run()` 负责反复请求模型；每轮需要使用工具时，交给 `dispatch()`；后者找到并调用工具记录中的 `handler`，把结果送回循环。

## 3. 执行的工具也可以是另一个 Agent

读取工具返回论文文字。如果把工具的执行函数换成“让另一个 Agent 做一项任务”，它就可以返回那个 Agent 的回答。

本课把这项工具叫作 `task`。Main Agent 请求它时，要交代两件事：找哪一位助手、让它做什么。例如：

```text
工具名：task
agent_type：paper
description：读取这篇论文，查清模型结构、数据量和训练设置，注明页码。
```

收到这个请求后，Main Agent 仍然走原来的循环：发现有工具请求，交给“执行工具”的 `dispatch()`，等待结果。它不需要另外一套循环来处理 task。

不同之处在于 `dispatch()` 找到的函数：

| 模型请求的工具 | 对应函数做什么 | 返回的结果 |
| --- | --- | --- |
| read_paper | 读取 PDF | 论文文字 |
| task | 调用 spawn_subagent，创建另一个 Agent，执行它的 run | 那个 Agent 的回答 |

`main.py` 已经把 `parent.spawn_subagent` 存进 task 的 handler 字段。因此按名字找到 task 后，执行它保存的函数，就会进入 `parent.spawn_subagent()`，开始创建子助手。

`spawn_subagent()` 根据 `agent_type` 选工作说明和工具，创建 child，再调用 `child.run(description)`。child 随即进入同样的循环：请求模型、判断工具请求、执行工具、继续或返回。

```mermaid
flowchart TD
    A[Main Agent 请求模型] --> B{有工具请求？}
    B -->|没有| C[检查并返回最终回答]
    B -->|请求 task| D[创建 child，调用 child.run]
    D --> E[child 请求模型]
    E --> F{有工具请求？}
    F -->|有| G[child 执行工具，保存结果]
    G --> E
    F -->|没有| H[child 检查并返回回答]
    H --> I[Main Agent 把回答保存为 task 结果]
    I --> A
```

这张图省略了空回答和轮数上限，保留了两条循环的关系。Main Agent 停在执行 task 的地方，等 child 返回；拿到结果后，它继续下一轮。

本课不让 child 再分配任务，所以创建 child 时要去掉 task 工具，也不给它传其他助手的配置。这样调用只到一层，不会一层接一层地创建下去。

## 4. 同一个 run()，怎么知道正在运行谁？

Main Agent 和 child 都是 `Agent` 类创建的对象，所以使用同一份 `run()` 代码。调用哪个对象的方法，里面的 `self` 就指向哪个对象。

| 正在执行 | self 是谁 | 使用谁的工具 |
| --- | --- | --- |
| parent.run、parent.dispatch | Main Agent | Main Agent 的工具表，里面只有 task |
| parent.spawn_subagent | Main Agent | 查它保存的助手配置，创建 child |
| child.run、child.dispatch | 当前 child | 这位助手自己的工具表 |

`parent.spawn_subagent` 表示“parent 的创建子助手方法”。把它存进 task 的 handler 后，再调用 handler，执行的仍然是 parent 的这个方法，所以 self 仍是 parent。

在这个方法中执行 `child.run(description)`，才进入 child 的循环。child 返回后，程序回到 parent 原先停下的位置，继续用 parent 的消息和工具：

```text
parent.run 的某一轮
  → parent.dispatch 收到 task
    → parent.spawn_subagent 创建 child
      → child.run 开始自己的循环
        → child.dispatch 执行读取等工具
        → child.run 继续请求模型，直到返回回答
    → parent.spawn_subagent 返回这段回答
  → parent.dispatch 返回 task 的结果
parent.run 保存结果，进入下一轮
```

这里没有启动另一个 Agent 进程，只是在同一个程序中调用另一个对象的方法。只有执行 PyTorch 实验的工具会另开检查进程；它与创建 child 是两件事。

## 5. 两条循环分别保存消息

除了工具不同，parent 和 child 的消息也要分开。已提供的 `run()` 会在每次调用时新建 `messages`，而不是把消息一直存在同一个公共列表里。

parent 的第一条任务是用户的调查要求，child 的第一条任务是 Main Agent 给它的 description。两边各自追加模型回复和工具结果，child 结束时只返回回答文本。

因此，论文助手读过的 PDF 文字留在自己的列表中，Main Agent 下一轮拿到的是 task 返回的论文摘要：

```mermaid
flowchart LR
    A[Main Agent 的消息：调查任务] -->|task 的 description| B[论文助手的消息：读取任务]
    B --> C[追加读到的 PDF 文字，继续循环]
    C -->|返回摘要| D[Main Agent 追加 task 结果，继续循环]
```

这也决定了下一项任务必须写清楚什么。难度助手看不到前两位的消息，Main Agent 需要把论文设置和环境结果写进给它的 description，不能只说“根据刚才的结果判断”。

例如，论文中的数据量、每批样本数、训练轮数和来源页码，以及本次实验用了哪块设备、是否成功，都要传过去。没有提供数据目录，就写“未提供目录”，不能写成“电脑里没有数据”。

难度助手拿到这些信息后，才能调用工具计算训练步数、检查指定目录，再说明还需要准备或确认什么。你最后检查报告时，也要核对这些依据是否完整。

## 6. 把这条循环用在论文调查里

有了这条循环，就可以给不同助手安排任务。本课设置了三位助手，每一位都使用同一个 `run()`，区别在于工作说明和工具：

| 助手 | 要查什么 | 已提供的工具 |
| --- | --- | --- |
| 论文助手 paper | 模型结构、数据量、训练设置和来源页码；有代码时比较实现 | 读取、搜索 PDF；读取指定源码或 AlexNet 实现 |
| 环境助手 environment | PyTorch 版本、可用设备；有配套实验时实际运行 | 环境检测、模型单步实验 |
| 难度助手 difficulty | 根据前两项结果判断下一步 | 检查数据目录、计算训练步数 |

这些工作说明放在 `knowledge/`，工具在 `main.py` 中分配。Main Agent 的说明要求它先找论文助手和环境助手，等两位返回后，再把结果交给难度助手。

“先找谁、后找谁”由模型按说明提出请求，程序按请求顺序执行。测试会检查难度助手是否过早启动；你也可以用 `--trace` 查看实际过程。

环境工具支持 CUDA、MPS 和 CPU。CUDA 用于 NVIDIA GPU，MPS 是 PyTorch 在 Mac 上使用 GPU 的后端，CPU 不需要 GPU。我用 Mac 演示，你按自己的设备选择。默认 auto 先检查 CUDA，再检查 MPS，都不可用就选 CPU。

仓库还提供了 AlexNet 单步实验。环境助手调用它后，工具会创建模型，用随机输入完成一次前向计算、反向传播和参数更新，并记录实际设备。成功只说明这一步能执行，不能当作达到了论文准确率。

这份实验是预先写好的，不会随着 PDF 内容自动生成。换其他论文时，可以继续读论文、查环境；如果还要运行那篇论文的模型，需要在 `tools.py` 添加对应实验，再在 `main.py` 分配工具。

## 7. 作业：接通主助手与子助手

打开 `agent.py` 的 `spawn_subagent()`。主助手请求 task 时，程序会进入这里。选助手、准备工具和限制回答长度已经写好，你只需要替换两处 `None`。

### 第一步：创建 child

此时 `spec` 是选中的助手配置，`spec.system` 是它的工作说明，`child_tools` 是它能使用的工具。用 `Agent(...)` 创建一个新对象，存入 child：

- model 使用 `self.model`，两位助手共用同一个 API 客户端。
- system 使用 `spec.system`，工具使用 `child_tools`。
- max_turns 使用 `self.max_turns`。
- 不传 specialists；本课只让主助手分配任务。

这里的 self 仍是主助手。创建 child 只是准备好了另一个助手，还没有请求模型。

### 第二步：让 child 开始工作

调用 child 的 `run()`，把本次任务 description 传进去，将返回值存入 summary。这次调用才会进入子助手的循环：它请求模型、使用自己的工具，直到给出回答。

不要把主助手的消息传进去。`run()` 会为 child 新建消息列表，先放入它的工作说明和 description。

child 返回后，下面已经写好的代码会把 summary 交回主助手。主助手的 `run()` 将它保存为 task 的结果，再请求模型，继续调查。

```text
你补的第一处：创建 child
你补的第二处：调用 child.run，得到 summary
已提供的代码：返回 summary → 保存为 task 结果 → 主助手继续循环
```

写完后，对照第 4 节的调用顺序，指出主助手在哪一行等待、子助手从哪里开始运行、回答又回到了哪里。能解释这三个位置，才算理解了这两处代码。

`dispatch()`、`run()`、错误处理和 API 请求都已提供，不属于本次作业，不需要修改。

## 8. 检查作业，再运行真实调查

写完后，在仓库目录运行测试：

```bash
uv sync --locked
uv run pytest -q
```

测试不调用真实模型，因此不需要密钥。与作业有关的测试会检查：是否创建了新的 child、是否使用选中助手的工作说明和工具、是否只传入 description，以及 child 的回答是否交回主助手。其他测试用于检查已提供的代码，你不需要修改。

测试通过后，配置密钥，再让助手实际调查 AlexNet 论文：

```bash
uv sync --locked --extra ml
# 已有 .env 就跳过复制，直接检查里面的密钥
cp -n .env.example .env
# 打开 .env，填入 DEEPSEEK_API_KEY
uv run --extra ml python main.py --trace
```

`main.py` 默认读取 `examples/alexnet/paper.pdf`，并提供 AlexNet 单步实验。明确指定设备时，在命令末尾加 `--device cuda`、`--device mps` 或 `--device cpu`。指定的设备不可用时会报告失败，不会换到其他设备。

运行时，找到一次工具请求，看看它的结果怎样进入下一轮；再找到一次 task，看看 child 怎样开始循环、返回后 parent 怎样继续。最后打开 `output/report.md`，对照论文和工具结果检查结论。

看过报告后，再用验收程序检查助手的实际调用：

```bash
uv run --extra ml python -m tests.smoke
```

验收会重新发起一次调查，检查三位助手、PDF 读取、模型实验、数据目录检查和训练步数计算是否实际执行，并核对参数、输入和输出的设备。它需要密钥，也会消耗 API 额度。记录保存在 `output/smoke.json`，报告保存在 `output/report.md`。

验收命令同样支持 `--device cuda`、`--device mps`、`--device cpu`。这些检查只能确认执行过程，论文结论仍需要你对照原文核查。

## 9. 提交什么

提交 `agent.py`、测试结果和 AlexNet 论文调查报告，并回答：

1. `task.handler` 执行时 self 是谁？进入 `child.run()` 后呢？
2. 难度助手看不到前两位的消息，Main Agent 应在 description 中传什么？
3. 报告里哪条结论来自论文，哪条来自本机实测？分别指出证据。
4. 本机单步实验通过后，距离复现论文指标还缺什么？

公开仓库暂不提供参考答案。用测试检查自己的实现，再用实际调用记录解释它如何运行。
