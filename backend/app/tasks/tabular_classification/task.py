from pydantic import BaseModel
from app.tasks.base import TaskDefinition

class TabularInput(BaseModel):
    features: dict[str, float | int | str]

class TabularOutput(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]

class TabularClassificationTask(TaskDefinition):
    @property
    def task_name(self) -> str:
        return "tabular_classification"

    @property
    def input_schema(self) -> type[BaseModel]:
        return TabularInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return TabularOutput

    @property
    def evaluation_metrics(self) -> list[str]:
        return ["accuracy", "precision", "recall", "f1"]

    def get_default_thresholds(self) -> dict[str, float]:
        return {"accuracy": 0.7, "f1": 0.65}