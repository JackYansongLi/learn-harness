"""已提供的工具：PDF、源码、数据目录，以及独立进程中的 PyTorch 检查。"""

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from agent import AgentError

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def select_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    """根据设备选项返回设备名，供环境检查和单步实验使用。

    requested 是 auto、cuda、mps 或 cpu；两个布尔值由调用方检查后传入。
    auto 按 CUDA、MPS、CPU 的顺序选择；明确指定设备时原样返回。
    此处不验证指定设备是否可用，单步实验会另行检查；未知选项抛出 ValueError。
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError("device must be auto, cuda, mps or cpu")
    if requested != "auto":
        return requested
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


PROJECT = Path(__file__).resolve().parent


def paper_pages(pdf: Path) -> tuple[list[str], str]:
    """读取整份 PDF，供论文 Subagent 的读页和搜索工具共用。

    pdf 是本地 PDF 路径，每次调用都会重新读取文件并提取所有页的文字。
    返回一个二元组：按页排列的文字列表，以及文件内容的 SHA-256 摘要。
    摘要用于标识本次读取的文件内容；这里不截短各页文字。
    文件最多 20,000,000 字节，必须带有 PDF 文件头。
    超出大小限制、格式错误、解析失败或整份文档没有可提取文字时抛出 AgentError。
    本课只提取 PDF 已有的文本，不从扫描图片中识别文字（OCR）。
    """
    if pdf.stat().st_size > 20_000_000:
        raise AgentError("this lesson accepts PDF files up to 20 MB")
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise AgentError("expected a PDF file")
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise AgentError("cannot parse PDF; use an unencrypted text PDF") from exc
    if not pages or not any(text.strip() for text in pages):
        raise AgentError("PDF has no extractable text; OCR is not included in this lesson")
    return pages, hashlib.sha256(data).hexdigest()


def read_paper(pdf: Path, pages: str) -> str:
    """按页读取论文，把原文交给论文 Subagent。

    pdf 是配置好的本地论文路径；pages 是模型传入的页码字符串，例如 "1,2"。
    一次接受 1 到 3 个页码，从 1 开始，按传入顺序返回。
    页码指 PDF 中的页序号，不一定与论文页脚印出的编号相同。
    返回 JSON 字符串，包含文件名、SHA-256 摘要、总页数和所选各页的文字。
    每页最多返回前 12,000 个字符；truncated 表示该页文字是否被截短。
    页码格式错误或超出范围时抛出 ValueError；PDF 的读取检查由 paper_pages 完成。
    """
    try:
        numbers = [int(p.strip()) for p in pages.split(",")]
    except ValueError as exc:
        raise ValueError('pages must be comma-separated page numbers, e.g. "1,2"') from exc
    if not 1 <= len(numbers) <= 3 or any(n < 1 for n in numbers):
        raise ValueError("read 1 to 3 pages per call; page numbers start at 1")
    texts, digest = paper_pages(pdf)
    if any(n > len(texts) for n in numbers):
        raise ValueError(f"paper has {len(texts)} pages")
    return json.dumps(
        {
            "file": pdf.name,
            "sha256": digest,
            "total_pages": len(texts),
            "pages": [
                {"page": n, "text": texts[n - 1][:12000], "truncated": len(texts[n - 1]) > 12000}
                for n in numbers
            ],
        },
        ensure_ascii=False,
    )


def search_paper(pdf: Path, query: str) -> str:
    """在论文中查找一段文字，帮助论文 Subagent 找到需要读的页。

    pdf 是本地论文路径，query 是非空的查询字符串。
    搜索不区分大小写，按文字直接匹配，不使用语义搜索或正则表达式。
    每页只取第一次命中，从命中位置前 200 个字符到后 800 个字符截取片段。
    返回 JSON 字符串，包含文件名、SHA-256 摘要、查询文字及最多 8 个命中页。
    total_hits 统计命中的页数，不是关键词出现次数；它可能大于实际返回的条数。
    空查询抛出 ValueError；需要更多上下文时，由 Subagent 再调用 read_paper。
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    texts, digest = paper_pages(pdf)
    hits = []
    for n, text in enumerate(texts, 1):
        pos = text.casefold().find(query.casefold())
        if pos >= 0:
            hits.append({"page": n, "excerpt": text[max(0, pos - 200) : pos + 800]})
    return json.dumps(
        {
            "file": pdf.name,
            "sha256": digest,
            "query": query,
            "total_hits": len(hits),
            "hits": hits[:8],
        },
        ensure_ascii=False,
    )


def read_code(path: Path) -> str:
    """读取指定源码，让论文 Subagent 对照论文查看实现。

    path 是用户通过 --code 指定的文件，按 UTF-8 文本读取，不执行其中的代码。
    返回 JSON 字符串，包含文件名、前 16,000 个字符，以及是否截短的标记。
    文件不存在或不能按 UTF-8 读取时，异常交给外层工具调用处理。
    """
    with path.open(encoding="utf-8") as stream:
        text = stream.read(16001)
    return json.dumps(
        {"file": path.name, "text": text[:16000], "truncated": len(text) > 16000},
        ensure_ascii=False,
    )


def run_probe(action: str, device: str = "auto") -> str:
    """启动独立 Python 进程执行本机检查，并等待它返回结果。

    action 为 environment 时检查环境，为 implementation 时读取模型源码，
    为 step 时运行单步实验；device 指定 auto、cuda、mps 或 cpu。
    论文 Subagent 通过它查看实现，环境 Subagent 通过它检查环境和运行实验。
    子进程使用当前 Python 解释器，在仓库目录执行本文件的 main()。
    返回 JSON 字符串：保留子进程的结果字段，并补上 process_exit_code。
    子进程即使失败，只要输出了有效结果，也会原样保留供 Subagent 判断。
    超过 180 秒未结束，或标准输出不能解析为 JSON 时，抛出 AgentError。
    子进程禁用 MPS 算子的 CPU 回退；指定设备不可用时不另换设备重试。
    """
    if action not in {"environment", "implementation", "step"} or device not in DEVICE_CHOICES:
        raise ValueError("unsupported probe action or device")
    try:
        run = subprocess.run(
            [sys.executable, "-m", "tools", action, "--device", device],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentError("local probe timed out after 180 seconds") from exc
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError as exc:
        raise AgentError(f"local probe failed (exit {run.returncode})") from exc
    # MPS 不可用等失败也保留结构化结果，交给 Agent 解释，不能算作通过。
    result["process_exit_code"] = run.returncode
    return json.dumps(result, ensure_ascii=False)


def inspect_dataset(dataset: Path | None) -> str:
    """检查数据目录的组织方式，供复现难度 Subagent 调用。

    dataset 是 --dataset 指定的根目录，未提供时为 None。
    返回 JSON 字符串，区分未指定目录、路径不是目录和已检查目录三种状态。
    有效根目录下，分别统计 train 和 val 中直接包含的类别子目录数量。
    缺少 train 或 val 时，对应数量为 0；不会继续统计里面的图片。
    layout_checked 只表示检查过目录，不表示图片、标签或数据集完整性已经通过验证。
    未指定目录只表示这次没有检查，不能据此判断电脑中没有训练数据。
    """
    if dataset is None:
        return json.dumps(
            {
                "status": "not_configured",
                "option": "--dataset",
                "note": "未提供目录，不代表这台电脑一定没有数据。",
            },
            ensure_ascii=False,
        )
    if not dataset.is_dir():
        return json.dumps({"status": "missing_directory"})
    counts = {}
    for split in ("train", "val"):
        directory = dataset / split
        classes = [p for p in directory.iterdir() if p.is_dir()] if directory.is_dir() else []
        counts[split] = len(classes)
    return json.dumps(
        {
            "status": "layout_checked",
            "class_directories": counts,
            "scope": "仅检查 train/val 类别目录；未验证图片数、标签或完整性。",
        },
        ensure_ascii=False,
    )


def training_workload(samples: str, epochs: str, batch_size: str) -> str:
    """按给定训练参数计算步数，供复现难度 Subagent 估算工作量。

    samples、epochs、batch_size 分别是样本数、训练轮数、每批样本数的整数字符串。
    三个数都须在 1 到 1,000,000,000 之间；转换或范围检查失败时抛出 ValueError。
    每轮保留最后一批不足 batch_size 的样本，因此步数向上取整，再乘训练轮数。
    返回 JSON 字符串，包含每轮步数、总步数、样本累计使用次数及输入参数。
    只按传入数字计算，不检查本机是否已有数据，也不预测训练时间或准确率。
    """
    values = [int(value) for value in (samples, epochs, batch_size)]
    if any(value <= 0 or value > 1_000_000_000 for value in values):
        raise ValueError("samples, epochs and batch_size must be positive integers <= 1e9")
    n, e, b = values
    steps = (n + b - 1) // b
    return json.dumps(
        {
            "samples": n,
            "epochs": e,
            "batch_size": b,
            "steps_per_epoch": steps,
            "total_steps": steps * e,
            "sample_visits": n * e,
            "drop_last": False,
            "scope": "只按传入条件计算工作量，不预测训练时间。",
        },
        ensure_ascii=False,
    )


ALEXNET_PDF = PROJECT / "examples/alexnet/paper.pdf"


def environment(device: str = "auto") -> dict:
    """收集系统和 PyTorch 信息，供环境 Subagent 判断本机条件。

    由 run_probe 启动子进程，再由 main() 调用；device 是用户要求的设备模式。
    返回 Python 字典，包含系统、Python、torch、torchvision 版本和设备可用标记。
    selected_device 按 select_device 的规则选择，不代表已在该设备运行过模型。
    status 为 ok 只表示已取得环境信息；明确指定的设备仍可能显示为不可用。
    导入依赖抛出 ImportError 时，返回 missing_dependency、错误原因和安装命令。
    此处不运行模型，也不测试 CUDA 或 MPS 上的具体算子；这些由 train_step 检查。
    main() 会把这里返回的字典转换成 JSON，供父进程读取。
    """
    info = {
        "os": platform.system(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch
        import torchvision
    except ImportError as exc:
        return {
            **info,
            "status": "missing_dependency",
            "error": str(exc),
            "install": "uv sync --extra ml --locked",
        }
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    return {
        **info,
        "requested_device": device,
        "selected_device": select_device(device, cuda_available, mps_available),
        "status": "ok",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
    }


def implementation() -> dict:
    """读取本机 torchvision 的 AlexNet 类源码，供论文 Subagent 对照论文。

    无需参数；由 run_probe 的 implementation 检查进入 main() 后调用。
    返回 Python 字典，包含 torchvision 版本、类名、该类的源码和官方文档链接。
    不创建或训练模型；得到的是本机库的实现，不能当作 2012 年作者的原代码。
    """
    import torchvision
    from torchvision.models.alexnet import AlexNet

    return {
        "torchvision": torchvision.__version__,
        "implementation": "torchvision.models.AlexNet",
        "source": inspect.getsource(AlexNet),
        "docs": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html",
        "note": "这是本机安装的 torchvision 实现，不能直接当作 2012 年作者原代码。",
    }


def train_step(device: str) -> dict:
    """在本机执行一次 AlexNet 训练步骤，供环境 Subagent 检查能否运行。

    device 为 auto、cuda、mps 或 cpu；由 run_probe 的 step 检查进入 main() 后调用。
    auto 按可用性选择设备；明确指定的 CUDA 或 MPS 不可用时，返回 unavailable。
    此时不会改用其他设备；MPS 算子的 CPU 回退由子进程入口提前禁用。
    模型不加载预训练权重，输入是一张形状为 3×224×224 的随机图片，目标类别固定为 0。
    实验执行前向计算、交叉熵损失、反向传播和一次 SGD 参数更新。
    通过条件是输出形状为 [1, 1000]、损失有限、所有参数梯度存在且有限，
    并且最后一个分类层至少有一个权重发生变化；没有逐项检查所有层的权重变化。
    返回 Python 字典，包含通过状态、设备、张量形状、损失、梯度检查和耗时等结果。
    计时前后等待 GPU 操作完成；耗时不包含模型创建和随机输入准备。
    这里只用随机数据检查一次参数更新过程，不验证准确率，也不能推算完整训练时间。
    """
    import torch
    from torchvision.models import alexnet

    requested = device
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    device = select_device(requested, cuda_available, mps_available)
    if (device == "cuda" and not cuda_available) or (device == "mps" and not mps_available):
        return {
            "status": "unavailable",
            "requested_device": requested,
            "selected_device": device,
            "reason": f"{device} is not available; no other device was tried",
            "cpu_fallback": False,
        }
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = alexnet(weights=None).to(device).train()
    x = torch.randn(1, 3, 224, 224, device=device)
    target = torch.tensor([0], device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.classifier[-1].weight.detach().clone()
    synchronize(device)
    started = time.perf_counter()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    gradients_finite = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    changed = bool((model.classifier[-1].weight.detach() != before).any())
    value = float(loss.detach().cpu())
    passed = (
        list(logits.shape) == [1, 1000] and math.isfinite(value) and gradients_finite and changed
    )
    return {
        "status": "passed" if passed else "failed",
        "requested_device": requested,
        "selected_device": device,
        "device": str(logits.device),
        "input_device": str(x.device),
        "target_device": str(target.device),
        "parameter_device": str(next(model.parameters()).device),
        "input_shape": list(x.shape),
        "output_shape": list(logits.shape),
        "loss": value,
        "gradients_finite": gradients_finite,
        "weights_updated": changed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(elapsed, 4),
        "batch_size": 1,
        "optimizer_steps": 1,
        "weights": None,
        "data": "synthetic random tensors; no ImageNet samples",
        "cpu_fallback": False,
        "scope": "仅验证前向、反向及更新；不是准确率测试，也不能据单步耗时推算完整训练时间。",
    }


def synchronize(device: str):
    """等待指定 GPU 上已经提交的操作完成，供 train_step 准确记录耗时。

    device 是已选定的设备名；CUDA 和 MPS 分别调用 PyTorch 对应的等待函数。
    CPU 情况下直接结束；此函数不返回检查结果，也不改变设备选择。
    同步的是设备计算，不是 Main Agent 与 Subagent 的消息。
    """
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    """执行命令行指定的本机检查，把 JSON 结果打印给父进程。

    无函数参数，从命令行读取 action 和 --device，通常由 run_probe 启动。
    action 选择环境检查、模型源码读取或单步实验，对应下方三个执行分支。
    在检查函数导入 torch 之前，先禁用 MPS 算子的 CPU 回退。
    执行检查时遇到异常，会将异常类型和原因写入状态为 error 的结果字典。
    字典转换成 JSON 字符串后打印到标准输出，而不是作为函数返回值交回。
    error、failed、unavailable 或 missing_dependency 状态使进程以退出码 1 结束；
    其他结果正常结束，由 run_probe 收集输出并交给调用它的 Subagent。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["environment", "implementation", "step"])
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args()
    # 必须在导入 torch 前设置；请求 MPS 时不静默换成 CPU。
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        if args.action == "environment":
            result = environment(args.device)
        elif args.action == "implementation":
            result = implementation()
        else:
            result = train_step(args.device)
    except Exception as exc:
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "install": "uv sync --extra ml --locked",
        }
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") in {"error", "failed", "unavailable", "missing_dependency"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
