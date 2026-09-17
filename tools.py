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
    with path.open(encoding="utf-8") as stream:
        text = stream.read(16001)
    return json.dumps(
        {"file": path.name, "text": text[:16000], "truncated": len(text) > 16000},
        ensure_ascii=False,
    )


def run_probe(action: str, device: str = "auto") -> str:
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
    # MPS 不可用等失败也保留结构化结果，交给助手解释，不能算作通过。
    result["process_exit_code"] = run.returncode
    return json.dumps(result, ensure_ascii=False)


def inspect_dataset(dataset: Path | None) -> str:
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
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
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
