from typing import Any

from app.tasks.base import EvaluationResult, ModelImplementation, PredictionResult, TrainingResult


class DummyModel(ModelImplementation):
    @property
    def model_name(self) -> str:
        return "dummy_model"

    @property
    def framework(self) -> str:
        return "dummy_framework"

    def train(self, train_data: Any) -> TrainingResult:
        return TrainingResult(
            metrics={"accuracy": 1.0}, artifact_path="/tmp/dummy", training_duration_seconds=1.0
        )

    def predict(self, input_data: Any) -> PredictionResult:
        return PredictionResult(prediction={"label": "dummy_prediction"})

    def evaluate(self, evaluation_data: Any) -> EvaluationResult:
        return EvaluationResult(
            metrics={"accuracy": 1.0}, num_samples=10, evaluation_duration_seconds=0.5
        )

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "ModelImplementation":
        # We need a task to instantiate, but tests won't call this directly
        # in a way that requires it, so we can return None or similar,
        # or just raise NotImplementedError for this mock since the registry
        # doesn't call load().
        raise NotImplementedError("Not used by registry discovery")
