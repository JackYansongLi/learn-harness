import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        choices=["exercise", "solution"],
        default="exercise",
        help="待测代码：exercise=main.py（默认），solution=本地 solution/agent.py",
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("exercise1") or item.get_closest_marker("exercise2"):
            item.add_marker(pytest.mark.exercise)


@pytest.fixture
def agent_class():
    from agent import Agent

    return Agent


@pytest.fixture
def main_agent_class(request):
    module = "solution.agent" if request.config.getoption("--impl") == "solution" else "main"
    return importlib.import_module(module).MainAgent
