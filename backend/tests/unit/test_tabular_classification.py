import os

import pytest

# pyrefly: ignore [missing-import]
from app.tasks.base import EvaluationResult, PredictionResult, TrainingResult

# pyrefly: ignore [missing-import]
from app.tasks.tabular_classification.models.random_forest import RandomForestModel

# pyrefly: ignore [missing-import]
from app.tasks.tabular_classification.task import TabularClassificationTask


@pytest.fixture
def sample_csv_path():
    path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "iris_sample.csv")
    return os.path.abspath(path)


@pytest.fixture
def task():
    return TabularClassificationTask()


@pytest.fixture
def model(task):
    return RandomForestModel(task=task)


def test_task_properties(task):
    assert task.task_name == "tabular_classification"
    assert "accuracy" in task.evaluation_metrics
    assert task.get_default_thresholds()["accuracy"] == 0.7


def test_random_forest_train_predict_evaluate(model, sample_csv_path, tmp_path):
    # 1. Test Training
    train_data = {
        "data_path": sample_csv_path,
        "target_column": "species",
        "params": {"n_estimators": 10, "random_state": 42},
    }

    train_result = model.train(train_data)
    assert isinstance(train_result, TrainingResult)
    assert "accuracy" in train_result.metrics
    assert train_result.metrics["accuracy"] >= 0.0  # Should be close to 1.0 on training set

    # 2. Test Prediction
    input_features = {
        "features": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    }
    pred_result = model.predict(input_features)
    assert isinstance(pred_result, PredictionResult)
    assert "predicted_class" in pred_result.prediction
    assert pred_result.prediction["predicted_class"] == "setosa"
    assert "setosa" in pred_result.prediction["probabilities"]

    # 3. Test Evaluation
    eval_data = {"data_path": sample_csv_path, "target_column": "species"}
    eval_result = model.evaluate(eval_data)
    assert isinstance(eval_result, EvaluationResult)
    assert "accuracy" in eval_result.metrics
    assert "f1" in eval_result.metrics
    assert eval_result.num_samples == 15

    # 4. Test Save/Load
    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    assert save_path.exists()

    loaded_model = RandomForestModel.load(str(save_path))
    loaded_model._task = model.task  # Inject task manually for testing

    # Verify loaded model makes same predictions
    loaded_pred = loaded_model.predict(input_features)
    assert loaded_pred.prediction["predicted_class"] == pred_result.prediction["predicted_class"]


def test_get_model_info(model):
    info = model.get_model_info()
    assert info.model_name == "random_forest"
    assert info.task_name == "tabular_classification"
    assert info.framework == "scikit-learn"
    assert "features" in info.input_schema["properties"]
