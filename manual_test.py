import sys
import os

# Add 'backend' to the Python path so imports work correctly from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from app.tasks.registry import init_registry

print("=== 1. Initializing Registry & Discovery ===")
registry = init_registry()
tasks = registry.list_tasks()
print(f"Discovered Tasks: {tasks}")

for task_name in tasks:
    models = registry.list_models(task_name)
    print(f"Models for '{task_name}': {models}")

print("\n=== 2. Fetching Tabular Task ===")
tabular_task = registry.get_task("tabular_classification")
print(f"Task Name: {tabular_task.task_name}")
print(f"Metrics: {tabular_task.evaluation_metrics}")

print("\n=== 3. Testing Random Forest Model ===")
rf_model_cls = registry.get_model("tabular_classification", "random_forest")
rf_model = rf_model_cls(task=tabular_task)
print(f"Model instantiated successfully: {rf_model.model_name}")

print("\n=== 4. Training on Sample Data ===")
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'tests', 'fixtures', 'iris_sample.csv'))

train_data = {
    "data_path": csv_path,
    "target_column": "species",
    "params": {"n_estimators": 5}
}

result = rf_model.train(train_data)
print(f"Training completed in {result.training_duration_seconds:.4f}s")
print(f"Training Metrics: {result.metrics}")

print("\n=== 5. Making a Prediction ===")
input_data = {
    "features": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
}
prediction = rf_model.predict(input_data)
print(f"Prediction: {prediction.prediction}")
print(f"Confidence: {prediction.confidence}")

print("\nSuccess! The registry discovery and tabular task work end-to-end.")
