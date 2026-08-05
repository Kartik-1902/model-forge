import pytest
from app.tasks.base import ModelImplementation
from app.tasks.registry import TaskRegistry
from tests.unit.mock_tasks.dummy_task.models.dummy_model import DummyModel
from tests.unit.mock_tasks.dummy_task.task import DummyTask


def test_registry_discovery():
    registry = TaskRegistry()
    registry.discover("tests.unit.mock_tasks")

    # Verify task is discovered
    tasks = registry.list_tasks()
    assert "dummy_task" in tasks
    assert len(tasks) == 1

    # Verify task can be retrieved
    task = registry.get_task("dummy_task")
    assert isinstance(task, DummyTask)
    assert task.task_name == "dummy_task"

    # Verify models are discovered
    models = registry.list_models("dummy_task")
    assert "dummy_model" in models
    assert len(models) == 1

    # Verify model class can be retrieved
    model_cls = registry.get_model("dummy_task", "dummy_model")
    assert issubclass(model_cls, ModelImplementation)
    assert model_cls is DummyModel


def test_registry_task_info():
    registry = TaskRegistry()
    registry.discover("tests.unit.mock_tasks")

    info = registry.get_task_info("dummy_task")
    assert info["task_name"] == "dummy_task"
    assert "input_schema" in info
    assert "output_schema" in info
    assert "accuracy" in info["evaluation_metrics"]
    assert info["default_thresholds"] == {"accuracy": 0.5}
    assert info["models"] == ["dummy_model"]


def test_registry_missing_task():
    registry = TaskRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.get_task("nonexistent")


def test_registry_missing_model():
    registry = TaskRegistry()
    registry.discover("tests.unit.mock_tasks")
    with pytest.raises(KeyError, match="not registered"):
        registry.get_model("dummy_task", "nonexistent")
