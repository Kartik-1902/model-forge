from pydantic import BaseModel
from app.tasks.base import TaskDefinition

class DummyInput(BaseModel):
    text: str

class DummyOutput(BaseModel):
    label: str

class DummyTask(TaskDefinition):
    @property
    def task_name(self) -> str:
        return "dummy_task"
    
    @property
    def input_schema(self) -> type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DummyOutput

    @property
    def evaluation_metrics(self) -> list[str]:
        return ["accuracy"]
    
    def get_default_thresholds(self) -> dict[str, float]:
        return {"accuracy": 0.5}
