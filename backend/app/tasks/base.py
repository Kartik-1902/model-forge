from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

@dataclass
class TrainingResult():
    metrics: dict[ str, float]
    artifact_path: str 
    training_duration_seconds: float
    additional_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult():
    prediction: Any 
    confidence: float | None = None
    metadata: dict [ str , Any] = field(default_factory=dict)

@dataclass
class EvaluationResult():
    metrics: dict[str, float]          # {'accuracy': 0.94, 'f1': 0.91, ...}
    num_samples: int
    evaluation_duration_seconds: float
    additional_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelInfo():
    model_name: str
    task_name: str
    framework: str
    input_schema: dict [str, Any]    # JSON Schema
    output_schema: dict [str, Any]   # JSON Schema
    parameter_count: int | None = None
    artifact_size_bytes: int | None = None
    description: str = ""


class TaskDefinition(ABC):
    """Defines an ML task type. Each task declares its contract
    but does not implement training or inference."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Canonical name: 'image_classification', 'text_classification', etc."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic model describing the expected prediction input."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Pydantic model describing the prediction output."""
        pass

    @property
    @abstractmethod
    def evaluation_metrics(self) -> list[str]:
        """Metrics this task reports: ['accuracy', 'precision', 'recall', 'f1']"""
        pass

    @abstractmethod
    def get_default_thresholds(self) -> dict[str, float]:
        """Minimum metric values for a model to pass evaluation.
        Example: {'accuracy': 0.7, 'f1': 0.65}
        Can be overridden per training request."""
        pass

from abc import ABC, abstractmethod
from typing import Any

class ModelImplementation(ABC):
    """
    Base contract for every ML model implementation.

    A ModelImplementation knows HOW to train, predict, evaluate,
    save, and load a specific model for a given ML task.
    """

    def __init__(self, task: TaskDefinition):
        self._task = task

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model name (e.g. 'ResNet50', 'XGBoost')."""
        pass

    @property
    def task(self) -> TaskDefinition:
        """
        Task implemented by this model.

        Shared immutable reference supplied during construction.
        """
        return self._task

    @property
    @abstractmethod
    def framework(self) -> str:
        """Framework used by this model (PyTorch, TensorFlow, sklearn, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, train_data: Any) -> TrainingResult:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """Run inference."""
        pass

    @abstractmethod
    def evaluate(self, evaluation_data: Any) -> EvaluationResult:
        """Evaluate the model."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ModelImplementation":
        """Load a previously saved model."""
        pass

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------
    def get_model_info(self) -> ModelInfo:
        """
        Returns metadata describing the model.

        Subclasses may override this if they need to provide
        additional information such as parameter count or
        artifact size.
        """
        return ModelInfo(
            model_name=self.model_name,
            task_name=self.task.task_name,
            framework=self.framework,
            input_schema=self.task.input_schema.model_json_schema(),
            output_schema=self.task.output_schema.model_json_schema(),
        )