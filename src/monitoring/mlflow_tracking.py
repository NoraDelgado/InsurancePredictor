"""
MLflow Tracking Module for Insurance Cost Predictor.

This module provides MLOps capabilities including:
- Experiment tracking
- Model versioning
- Metric logging
- Artifact management
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime


class MLFlowTracker:
    """
    Track experiments and model performance with MLflow.
    
    Provides a high-level interface for logging experiments,
    parameters, metrics, and artifacts.
    """
    
    def __init__(
        self,
        experiment_name: str = "insurance_predictor",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize the MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: URI for MLflow tracking server.
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run.
            tags: Optional tags for the run.
            
        Returns:
            Active MLflow run.
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_training_run(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Log a complete training run to MLflow.
        
        Args:
            model: Trained model.
            model_name: Name of the model.
            params: Training parameters.
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            feature_names: Optional list of feature names.
            additional_metrics: Optional additional metrics to log.
            
        Returns:
            Run ID.
        """
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            self._log_params_safe(params)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate and log metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train, prefix="train_")
            test_metrics = self._calculate_metrics(y_test, y_pred_test, prefix="test_")
            
            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)
            
            if additional_metrics:
                mlflow.log_metrics(additional_metrics)
            
            # Log cross-validation metrics if available
            cv_score = self._get_cv_score(model, X_train, y_train)
            if cv_score is not None:
                mlflow.log_metric("cv_r2_mean", cv_score)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_') and feature_names:
                importance_dict = dict(zip(feature_names, model.feature_importances_.tolist()))
                mlflow.log_dict(importance_dict, "feature_importance.json")
            
            # Log dataset info
            dataset_info = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "timestamp": datetime.now().isoformat()
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")
            
            return run.info.run_id
    
    def log_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        best_model_name: str
    ) -> str:
        """
        Log model comparison results.
        
        Args:
            results: Dictionary of model name to metrics.
            best_model_name: Name of the best performing model.
            
        Returns:
            Run ID.
        """
        with mlflow.start_run(run_name="model_comparison") as run:
            # Log each model's metrics with prefix
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    safe_name = f"{model_name}_{metric_name}".replace(" ", "_").lower()
                    mlflow.log_metric(safe_name, value)
            
            # Log best model info
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_r2", results[best_model_name].get('r2', 0))
            
            # Log comparison as artifact
            mlflow.log_dict(results, "model_comparison.json")
            
            return run.info.run_id
    
    def log_preprocessing_artifacts(
        self,
        preprocessor: Any,
        feature_engineer: Any = None,
        artifact_path: str = "preprocessing"
    ) -> None:
        """
        Log preprocessing artifacts.
        
        Args:
            preprocessor: Preprocessor object.
            feature_engineer: Optional feature engineer object.
            artifact_path: Path for artifacts.
        """
        import joblib
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save preprocessor
            preprocessor_path = Path(tmpdir) / "preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(str(preprocessor_path), artifact_path)
            
            # Save feature engineer if provided
            if feature_engineer is not None:
                fe_path = Path(tmpdir) / "feature_engineer.pkl"
                joblib.dump(feature_engineer, fe_path)
                mlflow.log_artifact(str(fe_path), artifact_path)
    
    def _log_params_safe(self, params: Dict[str, Any]) -> None:
        """Log parameters with type conversion for MLflow compatibility."""
        for key, value in params.items():
            try:
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                # Convert to string if not a basic type
                if not isinstance(value, (int, float, str, bool)):
                    value = str(value)
                mlflow.log_param(key, value)
            except Exception:
                mlflow.log_param(key, str(value))
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        return {
            f"{prefix}r2": r2_score(y_true, y_pred),
            f"{prefix}mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}mape": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
        }
    
    def _get_cv_score(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Optional[float]:
        """Get cross-validation score if possible."""
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            return float(scores.mean())
        except Exception:
            return None
    
    def get_best_run(self, metric: str = "test_r2") -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize.
            
        Returns:
            Best run info or None.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if len(runs) > 0:
            return runs.iloc[0].to_dict()
        return None
    
    def load_best_model(self, metric: str = "test_r2") -> Optional[Any]:
        """
        Load the best model based on a metric.
        
        Args:
            metric: Metric to optimize.
            
        Returns:
            Loaded model or None.
        """
        best_run = self.get_best_run(metric)
        if best_run:
            model_uri = f"runs:/{best_run['run_id']}/model"
            return mlflow.sklearn.load_model(model_uri)
        return None


class ModelRegistry:
    """
    Manage model versions in MLflow Model Registry.
    """
    
    def __init__(self, model_name: str = "insurance_predictor"):
        """
        Initialize the model registry.
        
        Args:
            model_name: Registered model name.
        """
        self.model_name = model_name
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(
        self,
        run_id: str,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID.
            description: Optional model description.
            
        Returns:
            Model version.
        """
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, self.model_name)
        
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=result.version,
                description=description
            )
        
        return result.version
    
    def transition_model_stage(
        self,
        version: str,
        stage: str = "Production"
    ) -> None:
        """
        Transition a model version to a stage.
        
        Args:
            version: Model version.
            stage: Target stage (Staging, Production, Archived).
        """
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
    
    def load_production_model(self) -> Any:
        """
        Load the production model.
        
        Returns:
            Production model.
        """
        model_uri = f"models:/{self.model_name}/Production"
        return mlflow.sklearn.load_model(model_uri)
    
    def get_latest_versions(self) -> List[Dict[str, Any]]:
        """
        Get latest model versions.
        
        Returns:
            List of model version info.
        """
        versions = self.client.get_latest_versions(self.model_name)
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "description": v.description
            }
            for v in versions
        ]

