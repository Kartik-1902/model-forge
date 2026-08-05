from pydantic import BaseModel, Field

class RandomForestParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=100)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
