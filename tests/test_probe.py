"""设备选择与失败处理不依赖真实 GPU；硬件实测单独记录。"""

import json
import sys
from types import SimpleNamespace

import pytest

import tools as probe
from tools import select_device


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_selects_available_device(cuda, mps, expected):
    assert select_device("auto", cuda, mps) == expected


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_explicit_device_is_preserved(device):
    assert select_device(device, False, False) == device
    assert select_device(device, True, True) == device


def test_invalid_device():
    with pytest.raises(ValueError):
        select_device("gpu", True, True)


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda" if cuda else None),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: mps)
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_environment_reports_runtime_backend(monkeypatch, cuda, mps, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda, mps))
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(__version__="test-vision"))
    result = probe.environment()
    assert result["torch"] == "test-torch"
    assert result["mps_built"] is True and result["mps_available"] == mps
    assert result["cuda_available"] == cuda
    assert result["selected_device"] == expected
    assert result["requested_device"] == "auto"
    assert result["status"] == "ok"


def test_environment_missing_dependency_is_not_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    result = probe.environment()
    assert result["status"] == "missing_dependency"
    assert "--extra ml" in result["install"]


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_gpu_does_not_construct_cpu_model(monkeypatch, device):
    constructed = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torchvision.models",
        SimpleNamespace(
            alexnet=lambda **kw: constructed.append(kw),
        ),
    )
    result = probe.train_step(device)
    assert result["status"] == "unavailable"
    assert result["requested_device"] == device
    assert not result["cpu_fallback"]
    assert constructed == []


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_synchronize_only_selected_backend(monkeypatch, device):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: calls.append("cuda")),
            mps=SimpleNamespace(synchronize=lambda: calls.append("mps")),
        ),
    )
    probe.synchronize(device)
    assert calls == ([] if device == "cpu" else [device])


@pytest.mark.parametrize("status", ["error", "failed", "unavailable", "missing_dependency"])
def test_worker_failure_exits_nonzero(monkeypatch, capsys, status):
    monkeypatch.setattr(sys, "argv", ["probe", "environment"])
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    def check(device):
        assert device == "auto"
        assert probe.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        return {"status": status}

    monkeypatch.setattr(probe, "environment", check)
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == status
