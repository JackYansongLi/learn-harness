# AlexNet：课堂测试与作业

这个案例用来练习给 Agent 分配复现前的调查任务：读 AlexNet 论文、检查你的电脑，再根据两份结果判断能先做哪些实验、还缺什么条件。

## 论文

Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton，*ImageNet Classification with Deep Convolutional Neural Networks*，2012。

[NeurIPS 原文 PDF](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

仓库已包含 [paper.pdf](paper.pdf)，克隆后即可使用，无需另行下载。上面的链接保留出版方来源。文件共 9 页，SHA-256：

```text
90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771
```

助手直接从这份 PDF 提取文字，不使用预写的论文摘要。其他论文使用主入口的 `--pdf` 参数传入。

## 实验

```bash
uv run --extra ml python main.py \
  --pdf examples/alexnet/paper.pdf --experiment alexnet --trace
```

`--pdf` 指定待读取的论文，`--experiment alexnet` 则为这次调查增加两个与 AlexNet 对应的工具：

- 论文助手的 `inspect_model`：读取本机 torchvision 的 AlexNet 类源码和版本。
- 环境助手的 `run_model_check`：用随机输入在指定设备完成前向、反向和一次 SGD 更新。

模型采用随机初始化，不下载权重。实验不下载 ImageNet，不启动完整训练。

默认自动选择可用设备，也可以加 `--device cuda`、`--device mps` 或 `--device cpu`。我用 Mac 演示 MPS，你按自己的设备选择。设备在创建工具时设定，助手调用 `run_model_check()` 时无需再传参数。

## 提交报告前核对

- 论文训练用的数据量、epoch、batch size 和硬件分别在哪一页？
- torchvision 的通道数、LRN、卷积分组与原文是否相同？
- 模型参数、输入和输出的实际设备是否与本次选择一致？参数有没有更新？
- 没提供数据目录时，报告有没有误写成“本机没有数据”？
- 训练步数来自哪些输入？约数算出的结果有没有写成精确统计？

本目录的 [checked-report.md](checked-report.md) 是我在 Mac 上运行后，对照 PDF 和工具输出核对的示例，不会作为助手输入。你写报告时，环境与设备要以自己的实测结果为准。自动生成的原始报告和完整调用记录留在本地 `output/`。

三位专家的通用知识说明在 [knowledge/](../../knowledge/)，不绑定这篇论文。
