import time
from typing import Any
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.tasks.base import (
    ModelImplementation, 
    TrainingResult, 
    PredictionResult, 
    EvaluationResult
)
from app.tasks.tabular_classification.config import RandomForestParams
from app.tasks.tabular_classification.preprocessing import (
    load_csv, 
    split_features_target, 
    build_preprocessor
)

class RandomForestModel(ModelImplementation):
    _model_name = "random_forest"
    
    def __init__(self, task):
        super().__init__(task)
        self._pipeline: Pipeline | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def framework(self) -> str:
        return "scikit-learn"

    def train(self, train_data: Any) -> TrainingResult:
        """
        train_data expects a dict:
        {
            "data_path": str,
            "target_column": str,
            "params": dict (optional)
        }
        """
        start_time = time.time()
        
        data_path = train_data["data_path"]
        target_column = train_data["target_column"]
        raw_params = train_data.get("params", {})
        
        # Parse params
        params = RandomForestParams(**raw_params)
        
        # Load and preprocess
        df = load_csv(data_path)
        X, y = split_features_target(df, target_column)
        
        preprocessor = build_preprocessor(X)
        classifier = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state
        )
        
        self._pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        self._pipeline.fit(X, y)
        
        # Calculate training accuracy
        y_pred = self._pipeline.predict(X)
        train_acc = accuracy_score(y, y_pred)
        
        # Note: We aren't saving it here yet since the celery worker handles it,
        # but the interface expects artifact_path so we can provide a dummy or empty for now,
        # or we could save it to a temp dir. Let's return empty and let caller use .save()
        
        training_duration = time.time() - start_time
        return TrainingResult(
            metrics={"accuracy": float(train_acc)},
            artifact_path="",  
            training_duration_seconds=training_duration
        )

    def predict(self, input_data: Any) -> PredictionResult:
        """
        input_data expects a dict matching the TabularInput schema:
        { "features": { ... } }
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")
            
        features_dict = input_data.get("features", {})
        df = pd.DataFrame([features_dict])
        
        prediction = self._pipeline.predict(df)[0]
        
        # Attempt to get probabilities if available
        probabilities = {}
        confidence = None
        if hasattr(self._pipeline.named_steps['classifier'], "predict_proba"):
            probs = self._pipeline.predict_proba(df)[0]
            classes = self._pipeline.classes_
            probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            confidence = max(probabilities.values()) if probabilities else None
            
        return PredictionResult(
            prediction={"predicted_class": str(prediction), "probabilities": probabilities},
            confidence=confidence
        )

    def evaluate(self, evaluation_data: Any) -> EvaluationResult:
        """
        evaluation_data expects a dict:
        {
            "data_path": str,
            "target_column": str
        }
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")
            
        start_time = time.time()
        
        data_path = evaluation_data["data_path"]
        target_column = evaluation_data["target_column"]
        
        df = load_csv(data_path)
        X, y = split_features_target(df, target_column)
        
        y_pred = self._pipeline.predict(X)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        eval_duration = time.time() - start_time
        
        return EvaluationResult(
            metrics={
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            },
            num_samples=len(y),
            evaluation_duration_seconds=eval_duration
        )

    def save(self, path: str) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model is not trained, nothing to save.")
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str) -> "ModelImplementation":
        """
        Reconstructs the model. 
        Note: The registry or caller must set the `.task` property or inject it 
        after loading, since load() is a classmethod and doesn't get the task directly.
        """
        # For DI to work, we need a task. We'll pass None for now and let the caller
        # inject the task later, or we modify the load signature. 
        # For now, we create an instance with None and assume it gets patched if needed,
        # or the caller uses the registry.
        instance = cls(task=None) # type: ignore
        instance._pipeline = joblib.load(path)
        return instance
