import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--impl", choices=["exercise", "solution"], default="exercise")


def pytest_collection_modifyitems(items):
    for item in items:
        if "recording" in item.fixturenames:
            item.add_marker(pytest.mark.exercise)


@pytest.fixture
def agent_class(request):
    module = "solution.agent" if request.config.getoption("--impl") == "solution" else "agent"
    return importlib.import_module(module).Agent
