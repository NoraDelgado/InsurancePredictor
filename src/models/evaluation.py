"""
Model Evaluation Module for Insurance Cost Predictor.

This module provides comprehensive model evaluation with visualizations,
cross-validation analysis, and performance reporting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations.
    
    Provides metrics calculation, residual analysis, learning curves,
    and performance visualization.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model to evaluate.
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            feature_names: Optional list of feature names.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        # Generate predictions
        self.y_pred_train = model.predict(X_train)
        self.y_pred_test = model.predict(X_test)
        self.residuals = y_test - self.y_pred_test
    
    def get_metrics(self, dataset: str = 'test') -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            dataset: Which dataset to evaluate ('train' or 'test').
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if dataset == 'train':
            y_true = self.y_train
            y_pred = self.y_pred_train
        else:
            y_true = self.y_test
            y_pred = self.y_pred_test
        
        # Handle potential division issues
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
        
        return {
            'R² Score': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE (%)': mape,
            'Max Error': np.max(np.abs(y_true - y_pred)),
            'Median AE': np.median(np.abs(y_true - y_pred)),
            'Std Error': np.std(y_true - y_pred)
        }
    
    def get_metrics_comparison(self) -> pd.DataFrame:
        """
        Compare metrics between train and test sets.
        
        Returns:
            DataFrame comparing train and test metrics.
        """
        train_metrics = self.get_metrics('train')
        test_metrics = self.get_metrics('test')
        
        return pd.DataFrame({
            'Train': train_metrics,
            'Test': test_metrics
        })
    
    def check_overfitting(self, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Check for overfitting by comparing train and test performance.
        
        Args:
            threshold: Maximum acceptable R² difference.
            
        Returns:
            Dictionary with overfitting analysis.
        """
        train_r2 = r2_score(self.y_train, self.y_pred_train)
        test_r2 = r2_score(self.y_test, self.y_pred_test)
        r2_diff = train_r2 - test_r2
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_difference': r2_diff,
            'is_overfitting': r2_diff > threshold,
            'severity': 'high' if r2_diff > 0.1 else ('medium' if r2_diff > 0.05 else 'low')
        }
    
    def plot_residuals(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot comprehensive residual analysis charts.
        
        Args:
            save_path: Optional path to save the figure.
            figsize: Figure size tuple.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(self.y_pred_test, self.residuals, alpha=0.5, edgecolors='none')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values ($)', fontsize=11)
        axes[0, 0].set_ylabel('Residuals ($)', fontsize=11)
        axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(self.y_pred_test, self.residuals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.y_pred_test.min(), self.y_pred_test.max(), 100)
        axes[0, 0].plot(x_line, p(x_line), 'g--', alpha=0.8, label='Trend')
        axes[0, 0].legend()
        
        # 2. Residual Histogram
        axes[0, 1].hist(self.residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=np.mean(self.residuals), color='g', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(self.residuals):.0f}')
        axes[0, 1].set_xlabel('Residuals ($)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        
        # 3. Actual vs Predicted
        axes[1, 0].scatter(self.y_test, self.y_pred_test, alpha=0.5, edgecolors='none')
        min_val = min(self.y_test.min(), self.y_pred_test.min())
        max_val = max(self.y_test.max(), self.y_pred_test.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Values ($)', fontsize=11)
        axes[1, 0].set_ylabel('Predicted Values ($)', fontsize=11)
        axes[1, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Residual Normality)', fontsize=12, fontweight='bold')
        axes[1, 1].get_lines()[0].set_markerfacecolor('steelblue')
        axes[1, 1].get_lines()[0].set_alpha(0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(
        self,
        cv: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot learning curve to detect overfitting/underfitting.
        
        Args:
            cv: Number of cross-validation folds.
            save_path: Optional path to save the figure.
            figsize: Figure size tuple.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot with confidence intervals
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', linewidth=2, label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', linewidth=2, label='Cross-Validation Score')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add final scores annotation
        plt.annotate(
            f'Final CV Score: {test_mean[-1]:.3f}',
            xy=(train_sizes[-1], test_mean[-1]),
            xytext=(train_sizes[-1] * 0.7, test_mean[-1] - 0.05),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray')
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Plot distribution of actual vs predicted values.
        
        Args:
            save_path: Optional path to save the figure.
            figsize: Figure size tuple.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Distribution comparison
        axes[0].hist(self.y_test, bins=30, alpha=0.6, label='Actual', color='blue', edgecolor='black')
        axes[0].hist(self.y_pred_test, bins=30, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
        axes[0].set_xlabel('Insurance Cost ($)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution: Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Error distribution by actual value quantiles
        actual_quantiles = pd.qcut(self.y_test, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_quantile = pd.DataFrame({
            'Quantile': actual_quantiles,
            'Absolute Error': np.abs(self.residuals)
        })
        
        error_by_quantile.boxplot(column='Absolute Error', by='Quantile', ax=axes[1])
        axes[1].set_xlabel('Actual Cost Quantile', fontsize=11)
        axes[1].set_ylabel('Absolute Error ($)', fontsize=11)
        axes[1].set_title('Error Distribution by Cost Level', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(
        self,
        top_n: int = 15,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> pd.DataFrame:
        """
        Plot feature importance for tree-based models.
        
        Args:
            top_n: Number of top features to display.
            save_path: Optional path to save the figure.
            figsize: Figure size tuple.
            
        Returns:
            DataFrame with feature importance values.
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model doesn't have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            features = self.feature_names
        else:
            features = [f'Feature {i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        top_features = importance_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Formatted report string.
        """
        metrics = self.get_metrics()
        overfitting = self.check_overfitting()
        
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\n📊 PERFORMANCE METRICS (Test Set)")
        report.append("-" * 40)
        for metric, value in metrics.items():
            if 'R²' in metric:
                report.append(f"  {metric}: {value:.4f}")
            elif '%' in metric:
                report.append(f"  {metric}: {value:.2f}%")
            else:
                report.append(f"  {metric}: ${value:,.2f}")
        
        report.append("\n🔍 OVERFITTING ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Train R²: {overfitting['train_r2']:.4f}")
        report.append(f"  Test R²: {overfitting['test_r2']:.4f}")
        report.append(f"  Difference: {overfitting['r2_difference']:.4f}")
        report.append(f"  Status: {'⚠️ Overfitting detected' if overfitting['is_overfitting'] else '✅ No significant overfitting'}")
        report.append(f"  Severity: {overfitting['severity'].upper()}")
        
        report.append("\n📈 RESIDUAL STATISTICS")
        report.append("-" * 40)
        report.append(f"  Mean Residual: ${np.mean(self.residuals):,.2f}")
        report.append(f"  Std Residual: ${np.std(self.residuals):,.2f}")
        report.append(f"  Skewness: {pd.Series(self.residuals).skew():.3f}")
        report.append(f"  Kurtosis: {pd.Series(self.residuals).kurtosis():.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model name to trained model.
        X_test: Test features.
        y_test: Test target.
        
    Returns:
        DataFrame comparing model performance.
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'R² Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE (%)': np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-10))) * 100
        }
    
    df = pd.DataFrame(results).T
    return df.sort_values('R² Score', ascending=False)

