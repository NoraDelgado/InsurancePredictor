"""
Model Training Module for Insurance Cost Predictor.

This module provides SOTA model training pipeline with hyperparameter
optimization using Optuna, including ensemble methods and automated model selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple, List, Callable
import optuna
from optuna.samplers import TPESampler
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class InsuranceModelTrainer:
    """
    SOTA model training pipeline with hyperparameter optimization.
    
    Features:
    - Multiple model architectures (linear, tree-based, boosting)
    - Optuna-based hyperparameter optimization
    - Ensemble methods (stacking, voting)
    - Automated model selection
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
    
    def get_base_models(self) -> Dict[str, Any]:
        """
        Initialize base models with default parameters.
        
        Returns:
            Dictionary of model name to model instance.
        """
        return {
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'xgboost': XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=200,
                depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=0
            )
        }
    
    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Tuple[XGBRegressor, Dict[str, Any]]:
        """
        Optimize XGBoost using Optuna.
        
        Args:
            X: Training features.
            y: Training target.
            n_trials: Number of optimization trials.
            timeout: Maximum optimization time in seconds.
            
        Returns:
            Tuple of (optimized model, best parameters).
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': 0
            }
            
            model = XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=self.n_jobs)
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = self.n_jobs
        best_params['verbosity'] = 0
        
        best_model = XGBRegressor(**best_params)
        best_model.fit(X, y)
        
        self.best_params['xgboost'] = best_params
        return best_model, best_params
    
    def optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Tuple[LGBMRegressor, Dict[str, Any]]:
        """
        Optimize LightGBM using Optuna.
        
        Args:
            X: Training features.
            y: Training target.
            n_trials: Number of optimization trials.
            timeout: Maximum optimization time in seconds.
            
        Returns:
            Tuple of (optimized model, best parameters).
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=self.n_jobs)
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = self.n_jobs
        best_params['verbose'] = -1
        
        best_model = LGBMRegressor(**best_params)
        best_model.fit(X, y)
        
        self.best_params['lightgbm'] = best_params
        return best_model, best_params
    
    def optimize_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Tuple[CatBoostRegressor, Dict[str, Any]]:
        """
        Optimize CatBoost using Optuna.
        
        Args:
            X: Training features.
            y: Training target.
            n_trials: Number of optimization trials.
            timeout: Maximum optimization time in seconds.
            
        Returns:
            Tuple of (optimized model, best parameters).
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_state': self.random_state,
                'verbose': 0
            }
            
            model = CatBoostRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=self.n_jobs)
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['verbose'] = 0
        
        best_model = CatBoostRegressor(**best_params)
        best_model.fit(X, y)
        
        self.best_params['catboost'] = best_params
        return best_model, best_params
    
    def create_stacking_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Optional[List[Tuple[str, Any]]] = None
    ) -> StackingRegressor:
        """
        Create a stacking ensemble of best models.
        
        Args:
            X: Training features.
            y: Training target.
            base_models: Optional list of (name, model) tuples.
            
        Returns:
            Fitted StackingRegressor.
        """
        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs
                )),
                ('xgb', XGBRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0
                )),
                ('lgbm', LGBMRegressor(
                    n_estimators=100, random_state=self.random_state, verbose=-1
                )),
                ('cat', CatBoostRegressor(
                    iterations=100, random_state=self.random_state, verbose=0
                ))
            ]
        
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(random_state=self.random_state),
            cv=5,
            n_jobs=self.n_jobs
        )
        stacking_model.fit(X, y)
        
        self.models['stacking'] = stacking_model
        return stacking_model
    
    def create_voting_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Optional[List[Tuple[str, Any]]] = None
    ) -> VotingRegressor:
        """
        Create a voting ensemble of best models.
        
        Args:
            X: Training features.
            y: Training target.
            base_models: Optional list of (name, model) tuples.
            
        Returns:
            Fitted VotingRegressor.
        """
        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs
                )),
                ('xgb', XGBRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0
                )),
                ('lgbm', LGBMRegressor(
                    n_estimators=100, random_state=self.random_state, verbose=-1
                ))
            ]
        
        voting_model = VotingRegressor(estimators=base_models, n_jobs=self.n_jobs)
        voting_model.fit(X, y)
        
        self.models['voting'] = voting_model
        return voting_model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        
        return {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
    
    def train_and_evaluate_all(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        optimize: bool = True,
        n_trials: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models and evaluate performance.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            optimize: Whether to run hyperparameter optimization.
            n_trials: Number of optimization trials per model.
            
        Returns:
            Dictionary of model name to evaluation metrics.
        """
        print("Training base models...")
        models = self.get_base_models()
        
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.results[name] = self.evaluate_model(model, X_test, y_test)
            self.models[name] = model
            print(f"    R²: {self.results[name]['r2']:.4f}, MAE: ${self.results[name]['mae']:.2f}")
        
        if optimize:
            print("\nOptimizing XGBoost...")
            optimized_xgb, _ = self.optimize_xgboost(X_train, y_train, n_trials=n_trials)
            self.results['xgboost_optimized'] = self.evaluate_model(optimized_xgb, X_test, y_test)
            self.models['xgboost_optimized'] = optimized_xgb
            print(f"  R²: {self.results['xgboost_optimized']['r2']:.4f}")
            
            print("\nOptimizing LightGBM...")
            optimized_lgbm, _ = self.optimize_lightgbm(X_train, y_train, n_trials=n_trials)
            self.results['lightgbm_optimized'] = self.evaluate_model(optimized_lgbm, X_test, y_test)
            self.models['lightgbm_optimized'] = optimized_lgbm
            print(f"  R²: {self.results['lightgbm_optimized']['r2']:.4f}")
        
        # Create ensemble
        print("\nCreating stacking ensemble...")
        stacking = self.create_stacking_ensemble(X_train, y_train)
        self.results['stacking'] = self.evaluate_model(stacking, X_test, y_test)
        print(f"  R²: {self.results['stacking']['r2']:.4f}")
        
        # Find best model
        best_name = max(self.results, key=lambda x: self.results[x]['r2'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name} with R² = {self.results[best_name]['r2']:.4f}")
        
        return self.results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a sorted DataFrame."""
        df = pd.DataFrame(self.results).T
        return df.sort_values('r2', ascending=False)
    
    def save_best_model(self, path: str | Path) -> None:
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save. Run train_and_evaluate_all first.")
        
        path = Path(path)
        joblib.dump(self.best_model, path)
        print(f"Saved {self.best_model_name} to {path}")
    
    def save_all_models(self, directory: str | Path) -> None:
        """Save all trained models to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = directory / f"{name}.pkl"
            joblib.dump(model, model_path)
        
        # Save results
        results_path = directory / "results.csv"
        self.get_results_dataframe().to_csv(results_path)
        
        print(f"Saved {len(self.models)} models to {directory}")


def train_production_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_save_path: str | Path,
    optimize: bool = True,
    n_trials: int = 50
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a production-ready model with full optimization.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        model_save_path: Path to save the best model.
        optimize: Whether to run hyperparameter optimization.
        n_trials: Number of optimization trials.
        
    Returns:
        Tuple of (best model, evaluation metrics).
    """
    trainer = InsuranceModelTrainer()
    
    results = trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test,
        optimize=optimize,
        n_trials=n_trials
    )
    
    trainer.save_best_model(model_save_path)
    
    return trainer.best_model, results[trainer.best_model_name]

