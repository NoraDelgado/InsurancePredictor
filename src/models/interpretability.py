"""
Model Interpretability Module for Insurance Cost Predictor.

This module provides SHAP-based model interpretability and explainability,
critical for healthcare/insurance domain to ensure trust and compliance.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ModelInterpreter:
    """
    Model interpretability using SHAP and other explainability methods.
    
    Provides:
    - Global feature importance
    - Local prediction explanations
    - Feature interaction analysis
    - Dependence plots
    """
    
    def __init__(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the interpreter.
        
        Args:
            model: Trained model to interpret.
            X_train: Training data for background distribution.
            feature_names: List of feature names.
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.feature_names = feature_names or self.X_train.columns.tolist()
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def fit(self, background_samples: int = 100) -> 'ModelInterpreter':
        """
        Fit the SHAP explainer.
        
        Args:
            background_samples: Number of samples for background distribution.
            
        Returns:
            Self for method chaining.
        """
        # Sample background data if needed
        if len(self.X_train) > background_samples:
            background = shap.sample(self.X_train, background_samples)
        else:
            background = self.X_train
        
        # Try TreeExplainer first (faster for tree-based models)
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fall back to KernelExplainer for other models
            try:
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    background
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create SHAP explainer: {e}")
        
        self._is_fitted = True
        return self
    
    def compute_shap_values(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Data to explain. Uses training data if None.
            
        Returns:
            SHAP values array.
        """
        if not self._is_fitted:
            self.fit()
        
        if X is None:
            X = self.X_train
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_summary(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_display: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP summary (beeswarm) plot.
        
        Args:
            X: Data to explain.
            max_display: Maximum features to display.
            save_path: Optional path to save the figure.
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None or len(self.shap_values) != len(X):
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance (Beeswarm)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_display: int = 15,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Plot SHAP feature importance bar chart.
        
        Args:
            X: Data to explain.
            max_display: Maximum features to display.
            save_path: Optional path to save the figure.
            
        Returns:
            DataFrame with feature importance values.
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None or len(self.shap_values) != len(X):
            self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def explain_prediction(
        self,
        instance: Union[pd.DataFrame, np.ndarray, Dict],
        plot: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            instance: Single instance to explain.
            plot: Whether to show waterfall plot.
            save_path: Optional path to save the figure.
            
        Returns:
            Dictionary with prediction explanation.
        """
        if not self._is_fitted:
            self.fit()
        
        # Convert to appropriate format
        if isinstance(instance, dict):
            instance = pd.DataFrame([instance])
        elif isinstance(instance, np.ndarray):
            instance = pd.DataFrame([instance], columns=self.feature_names)
        
        # Get SHAP values for this instance
        instance_shap = self.explainer.shap_values(instance.values)
        
        # Get prediction
        prediction = self.model.predict(instance.values)[0]
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = (
                self.explainer.expected_value[0] 
                if isinstance(self.explainer.expected_value, (list, np.ndarray)) 
                else self.explainer.expected_value
            )
        else:
            base_value = self.model.predict(self.X_train.values).mean()
        
        # Create explanation dictionary
        feature_contributions = []
        for i, fname in enumerate(self.feature_names):
            feature_contributions.append({
                'feature': fname,
                'value': float(instance.values[0, i]),
                'shap_value': float(instance_shap[0, i]),
                'contribution': 'positive' if instance_shap[0, i] > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        explanation = {
            'prediction': prediction,
            'base_value': base_value,
            'feature_contributions': feature_contributions,
            'top_positive_factors': [f for f in feature_contributions if f['shap_value'] > 0][:3],
            'top_negative_factors': [f for f in feature_contributions if f['shap_value'] < 0][:3]
        }
        
        # Plot waterfall
        if plot:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=instance_shap[0],
                    base_values=base_value,
                    data=instance.values[0],
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title('Prediction Explanation (Waterfall)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        return explanation
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP dependence plot for a feature.
        
        Args:
            feature: Feature to analyze.
            interaction_feature: Optional feature for interaction coloring.
            X: Data to use for plot.
            save_path: Optional path to save the figure.
        """
        if X is None:
            X = self.X_train
        
        if self.shap_values is None or len(self.shap_values) != len(X):
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence: {feature}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_force(
        self,
        instance: Union[pd.DataFrame, np.ndarray, Dict],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot SHAP force plot for a single prediction.
        
        Args:
            instance: Single instance to explain.
            save_path: Optional path to save the figure.
        """
        if not self._is_fitted:
            self.fit()
        
        # Convert to appropriate format
        if isinstance(instance, dict):
            instance = pd.DataFrame([instance])
        elif isinstance(instance, np.ndarray):
            instance = pd.DataFrame([instance], columns=self.feature_names)
        
        # Get SHAP values
        instance_shap = self.explainer.shap_values(instance.values)
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = (
                self.explainer.expected_value[0]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        else:
            base_value = self.model.predict(self.X_train.values).mean()
        
        # Create force plot
        shap.initjs()
        force_plot = shap.force_plot(
            base_value,
            instance_shap[0],
            instance.values[0],
            feature_names=self.feature_names
        )
        
        if save_path:
            shap.save_html(save_path, force_plot)
        
        return force_plot
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.
        
        Returns:
            DataFrame with feature importance values.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        importance = np.abs(self.shap_values).mean(0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_normalized': importance / importance.sum()
        }).sort_values('importance', ascending=False)
    
    def get_interaction_values(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute SHAP interaction values.
        
        Args:
            X: Data to compute interactions for.
            
        Returns:
            SHAP interaction values array.
        """
        if X is None:
            X = self.X_train
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Only TreeExplainer supports interaction values
        if isinstance(self.explainer, shap.TreeExplainer):
            return self.explainer.shap_interaction_values(X)
        else:
            raise NotImplementedError("Interaction values only available for tree-based models")
    
    def generate_explanation_text(
        self,
        instance: Union[pd.DataFrame, np.ndarray, Dict],
        prediction: float
    ) -> str:
        """
        Generate human-readable explanation text for a prediction.
        
        Args:
            instance: Instance to explain.
            prediction: Model prediction value.
            
        Returns:
            Human-readable explanation string.
        """
        explanation = self.explain_prediction(instance, plot=False)
        
        text = []
        text.append(f"Predicted Insurance Cost: ${prediction:,.2f}")
        text.append(f"\nBase Cost (Average): ${explanation['base_value']:,.2f}")
        text.append("\n📈 Factors INCREASING your cost:")
        
        for factor in explanation['top_positive_factors']:
            text.append(f"  • {factor['feature']}: +${abs(factor['shap_value']):,.2f}")
        
        text.append("\n📉 Factors DECREASING your cost:")
        for factor in explanation['top_negative_factors']:
            text.append(f"  • {factor['feature']}: -${abs(factor['shap_value']):,.2f}")
        
        return "\n".join(text)


def analyze_risk_factors(
    interpreter: ModelInterpreter,
    instance: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze risk factors for an insurance prediction.
    
    Args:
        interpreter: Fitted ModelInterpreter.
        instance: Input data dictionary.
        
    Returns:
        Dictionary with risk factor analysis.
    """
    explanation = interpreter.explain_prediction(instance, plot=False)
    
    risk_factors = []
    
    for contrib in explanation['feature_contributions']:
        if contrib['shap_value'] > 0:
            # Determine impact level
            abs_impact = abs(contrib['shap_value'])
            if abs_impact > 5000:
                impact = 'High'
                contribution = '+40-60%'
            elif abs_impact > 2000:
                impact = 'Medium'
                contribution = '+15-25%'
            else:
                impact = 'Low'
                contribution = '+5-10%'
            
            risk_factors.append({
                'factor': contrib['feature'],
                'impact': impact,
                'contribution': contribution,
                'shap_value': contrib['shap_value']
            })
    
    # Sort by impact
    impact_order = {'High': 0, 'Medium': 1, 'Low': 2}
    risk_factors.sort(key=lambda x: (impact_order.get(x['impact'], 3), -abs(x['shap_value'])))
    
    return {
        'prediction': explanation['prediction'],
        'risk_factors': risk_factors[:5],  # Top 5 risk factors
        'protective_factors': [
            f for f in explanation['feature_contributions']
            if f['shap_value'] < -500
        ][:3]
    }

