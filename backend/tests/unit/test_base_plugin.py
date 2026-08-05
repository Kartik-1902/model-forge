from typing import Any

import pytest
from app.tasks.base import (
    EvaluationResult,
    ModelImplementation,
    PredictionResult,
    TaskDefinition,
    TrainingResult,
)
from pydantic import BaseModel


def test_task_definition_is_abstract():
    """Verify TaskDefinition cannot be instantiated directly (it's abstract)"""
    with pytest.raises(TypeError):
        TaskDefinition()


def test_model_implementation_is_abstract():
    """Verify ModelImplementation cannot be instantiated directly (it's abstract)"""
    with pytest.raises(TypeError):
        # Passing None for task just to trigger instantiation
        ModelImplementation(task=None)


# Minimal Concrete Classes for Testing
class DummyInputSchema(BaseModel):
    feature_a: int


class DummyOutputSchema(BaseModel):
    prediction: str


class ConcreteTaskDefinition(TaskDefinition):
    @property
    def task_name(self) -> str:
        return "dummy_task"

    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInputSchema

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutputSchema

    @property
    def evaluation_metrics(self) -> list[str]:
        return ["accuracy"]

    def get_default_thresholds(self) -> dict[str, float]:
        return {"accuracy": 0.8}


class ConcreteModelImplementation(ModelImplementation):
    @property
    def model_name(self) -> str:
        return "dummy_model"

    @property
    def framework(self) -> str:
        return "dummy_framework"

    def train(self, train_data: Any) -> TrainingResult:
        return TrainingResult(
            metrics={"loss": 0.1}, artifact_path="/tmp", training_duration_seconds=1.0
        )

    def predict(self, input_data: Any) -> PredictionResult:
        return PredictionResult(prediction="class_a")

    def evaluate(self, evaluation_data: Any) -> EvaluationResult:
        return EvaluationResult(
            metrics={"accuracy": 0.9}, num_samples=100, evaluation_duration_seconds=0.5
        )

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "ModelImplementation":
        return cls(task=ConcreteTaskDefinition())


def test_concrete_task_definition():
    """Verify a concrete TaskDefinition subclass instantiates and returns expected values"""
    task = ConcreteTaskDefinition()
    assert task.task_name == "dummy_task"
    assert task.input_schema is DummyInputSchema
    assert task.output_schema is DummyOutputSchema
    assert task.evaluation_metrics == ["accuracy"]
    assert task.get_default_thresholds() == {"accuracy": 0.8}


def test_concrete_model_implementation():
    """Verify a concrete ModelImplementation subclass instantiates and functions correctly"""
    task = ConcreteTaskDefinition()
    model = ConcreteModelImplementation(task=task)

    assert model.model_name == "dummy_model"
    assert model.framework == "dummy_framework"
    assert model.task is task

    info = model.get_model_info()
    assert info.model_name == "dummy_model"
    assert info.task_name == "dummy_task"
    assert info.framework == "dummy_framework"
    assert info.input_schema["title"] == "DummyInputSchema"
    assert info.output_schema["title"] == "DummyOutputSchema"


def test_partial_task_implementation_raises_type_error():
    """Verify that a partial task implementation (missing one abstract method) raises TypeError"""

    class PartialTaskDefinition(TaskDefinition):
        @property
        def task_name(self) -> str:
            return "partial"

        # Intentionally missing input_schema, output_schema, evaluation_metrics, get_default_thresholds

    with pytest.raises(TypeError):
        PartialTaskDefinition()


def test_partial_model_implementation_raises_type_error():
    """Verify that a partial model implementation (missing abstract methods) raises TypeError"""

    class PartialModelImplementation(ModelImplementation):
        @property
        def model_name(self) -> str:
            return "partial"

        # Intentionally missing framework, train, predict, evaluate, save, load, get_model_info

    task = ConcreteTaskDefinition()
    with pytest.raises(TypeError):
        PartialModelImplementation(task=task)


def test_models_share_task_instance():
    """Verify that multiple models initialized with the same task share that exact task instance."""
    task = ConcreteTaskDefinition()
    model1 = ConcreteModelImplementation(task=task)
    model2 = ConcreteModelImplementation(task=task)

    assert model1.task is task
    assert model2.task is task
    assert model1.task is model2.task


def test_model_contract_types():
    """Verify that model methods return the types specified in the contract."""
    task = ConcreteTaskDefinition()
    model = ConcreteModelImplementation(task=task)

    train_res = model.train(None)
    assert isinstance(train_res, TrainingResult)

    predict_res = model.predict(None)
    assert isinstance(predict_res, PredictionResult)

    eval_res = model.evaluate(None)
    assert isinstance(eval_res, EvaluationResult)
