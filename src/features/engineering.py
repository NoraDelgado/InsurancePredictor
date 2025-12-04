"""
Feature Engineering Module for Insurance Cost Predictor.

This module creates advanced features for insurance cost prediction,
including domain-driven features, interaction terms, and risk scores.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


class InsuranceFeatureEngineer:
    """
    Create advanced features for insurance cost prediction.
    
    Implements domain-driven feature engineering including:
    - Age-based risk features
    - BMI health categories
    - Blood pressure risk indicators
    - Interaction features between key predictors
    - Composite risk scores
    """
    
    def __init__(self, include_interactions: bool = True, include_risk_score: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            include_interactions: Whether to create interaction features.
            include_risk_score: Whether to create composite risk score.
        """
        self.include_interactions = include_interactions
        self.include_risk_score = include_risk_score
        self.feature_names: List[str] = []
        self.categorical_mappings: Dict[str, Dict] = {}
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if the feature engineer has been fitted."""
        return self._is_fitted
    
    def fit(self, df: pd.DataFrame) -> 'InsuranceFeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            df: Training DataFrame.
            
        Returns:
            Self for method chaining.
        """
        # Store categorical mappings for consistent encoding
        self._create_categorical_mappings(df)
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating engineered features.
        
        Args:
            df: DataFrame to transform.
            
        Returns:
            DataFrame with engineered features.
        """
        df = df.copy()
        
        # Create age-based features
        df = self._create_age_features(df)
        
        # Create BMI-based features
        df = self._create_bmi_features(df)
        
        # Create blood pressure features
        df = self._create_bp_features(df)
        
        # Create interaction features
        if self.include_interactions:
            df = self._create_interaction_features(df)
        
        # Create risk score
        if self.include_risk_score:
            df = self._create_risk_score(df)
        
        # Encode any new categorical features
        df = self._encode_categorical_features(df)
        
        self.feature_names = df.columns.tolist()
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform.
            
        Returns:
            Transformed DataFrame with engineered features.
        """
        self.fit(df)
        return self.transform(df)
    
    def _create_categorical_mappings(self, df: pd.DataFrame) -> None:
        """Create mappings for categorical features."""
        # Age group mapping
        self.categorical_mappings['age_group'] = {
            'young': 0,
            'young_adult': 1,
            'middle': 2,
            'middle_senior': 3,
            'senior': 4,
            'elderly': 5
        }
        
        # BMI category mapping
        self.categorical_mappings['bmi_category'] = {
            'underweight': 0,
            'normal': 1,
            'overweight': 2,
            'obese_1': 3,
            'obese_2': 4,
            'obese_3': 5
        }
        
        # BP category mapping
        self.categorical_mappings['bp_category'] = {
            'low': 0,
            'normal': 1,
            'elevated': 2,
            'high_1': 3,
            'high_2': 4
        }
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-derived features."""
        if 'age' not in df.columns:
            return df
        
        # Age groups (domain-specific risk categories)
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 65, 150],
            labels=['young', 'young_adult', 'middle', 'middle_senior', 'senior', 'elderly'],
            include_lowest=True
        )
        
        # Age squared (captures non-linear relationship)
        df['age_squared'] = df['age'] ** 2
        
        # Age decade
        df['age_decade'] = (df['age'] // 10).astype(int)
        
        # Is senior flag
        df['is_senior'] = (df['age'] >= 55).astype(int)
        
        return df
    
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-derived features using medical classifications."""
        if 'bmi' not in df.columns:
            return df
        
        # WHO BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['underweight', 'normal', 'overweight', 'obese_1', 'obese_2', 'obese_3'],
            include_lowest=True
        )
        
        # BMI risk flags
        df['is_underweight'] = (df['bmi'] < 18.5).astype(int)
        df['is_overweight'] = (df['bmi'] >= 25).astype(int)
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        df['is_severely_obese'] = (df['bmi'] >= 35).astype(int)
        
        # BMI squared
        df['bmi_squared'] = df['bmi'] ** 2
        
        # BMI deviation from healthy (25 is upper healthy limit)
        df['bmi_deviation'] = np.abs(df['bmi'] - 22.5)  # 22.5 is midpoint of healthy range
        
        return df
    
    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create blood pressure derived features."""
        if 'bloodpressure' not in df.columns:
            return df
        
        # Blood pressure categories (simplified)
        df['bp_category'] = pd.cut(
            df['bloodpressure'],
            bins=[0, 80, 90, 100, 110, 250],
            labels=['low', 'normal', 'elevated', 'high_1', 'high_2'],
            include_lowest=True
        )
        
        # Hypertension flags
        df['is_low_bp'] = (df['bloodpressure'] < 80).astype(int)
        df['is_normal_bp'] = ((df['bloodpressure'] >= 80) & (df['bloodpressure'] <= 90)).astype(int)
        df['is_hypertensive'] = (df['bloodpressure'] > 90).astype(int)
        df['is_high_hypertensive'] = (df['bloodpressure'] > 110).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key predictors."""
        # Ensure required columns exist
        has_smoker = 'smoker' in df.columns
        has_bmi = 'bmi' in df.columns
        has_age = 'age' in df.columns
        has_diabetic = 'diabetic' in df.columns
        has_children = 'children' in df.columns
        
        # Handle smoker column (might be encoded or not)
        if has_smoker:
            smoker_numeric = df['smoker']
            if df['smoker'].dtype == 'object':
                smoker_numeric = (df['smoker'] == 'Yes').astype(int)
        
        # Handle diabetic column
        if has_diabetic:
            diabetic_numeric = df['diabetic']
            if df['diabetic'].dtype == 'object':
                diabetic_numeric = (df['diabetic'] == 'Yes').astype(int)
        
        # Smoker x BMI (smokers with high BMI are highest risk)
        if has_smoker and has_bmi:
            df['smoker_bmi'] = smoker_numeric * df['bmi']
        
        # Smoker x Age
        if has_smoker and has_age:
            df['smoker_age'] = smoker_numeric * df['age']
        
        # Age x BMI
        if has_age and has_bmi:
            df['age_bmi'] = df['age'] * df['bmi']
        
        # Diabetic x BMI
        if has_diabetic and has_bmi:
            df['diabetic_bmi'] = diabetic_numeric * df['bmi']
        
        # Smoker x Diabetic (compound risk)
        if has_smoker and has_diabetic:
            df['smoker_diabetic'] = smoker_numeric * diabetic_numeric
        
        # Children x Age (family planning impact)
        if has_children and has_age:
            df['children_age'] = df['children'] * df['age']
        
        # Has dependents flag
        if has_children:
            df['has_children'] = (df['children'] > 0).astype(int)
            df['many_children'] = (df['children'] >= 3).astype(int)
        
        return df
    
    def _create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk score based on domain knowledge."""
        risk_score = pd.Series(0.0, index=df.index)
        
        # Smoking is highest risk factor
        if 'smoker' in df.columns:
            if df['smoker'].dtype == 'object':
                risk_score += (df['smoker'] == 'Yes').astype(float) * 3.0
            else:
                risk_score += df['smoker'].astype(float) * 3.0
        
        # Obesity risk
        if 'is_obese' in df.columns:
            risk_score += df['is_obese'].astype(float) * 1.5
        elif 'bmi' in df.columns:
            risk_score += (df['bmi'] >= 30).astype(float) * 1.5
        
        # Hypertension risk
        if 'is_hypertensive' in df.columns:
            risk_score += df['is_hypertensive'].astype(float) * 1.2
        elif 'bloodpressure' in df.columns:
            risk_score += (df['bloodpressure'] > 90).astype(float) * 1.2
        
        # Diabetic risk
        if 'diabetic' in df.columns:
            if df['diabetic'].dtype == 'object':
                risk_score += (df['diabetic'] == 'Yes').astype(float) * 1.3
            else:
                risk_score += df['diabetic'].astype(float) * 1.3
        
        # Age risk
        if 'age' in df.columns:
            risk_score += (df['age'] > 45).astype(float) * 1.0
            risk_score += (df['age'] > 55).astype(float) * 0.5  # Additional for seniors
        
        # Children risk (minor factor)
        if 'children' in df.columns:
            risk_score += (df['children'] > 2).astype(float) * 0.3
        
        df['risk_score'] = risk_score
        
        # Risk categories
        df['risk_category'] = pd.cut(
            df['risk_score'],
            bins=[-0.1, 1.5, 3.5, 5.5, 10],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode new categorical features to numeric."""
        categorical_cols = ['age_group', 'bmi_category', 'bp_category', 'risk_category']
        
        for col in categorical_cols:
            if col in df.columns and col in self.categorical_mappings:
                mapping = self.categorical_mappings[col]
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after engineering."""
        return self.feature_names.copy()
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names."""
        numeric_types = ['int64', 'float64', 'int32', 'float32']
        return [col for col in self.feature_names 
                if any(t in str(type(col)) for t in numeric_types)]
    
    def save(self, path: str | Path) -> None:
        """Save feature engineer to disk."""
        path = Path(path)
        artifacts = {
            'include_interactions': self.include_interactions,
            'include_risk_score': self.include_risk_score,
            'feature_names': self.feature_names,
            'categorical_mappings': self.categorical_mappings,
            '_is_fitted': self._is_fitted
        }
        joblib.dump(artifacts, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'InsuranceFeatureEngineer':
        """Load feature engineer from disk."""
        path = Path(path)
        artifacts = joblib.load(path)
        
        engineer = cls(
            include_interactions=artifacts.get('include_interactions', True),
            include_risk_score=artifacts.get('include_risk_score', True)
        )
        engineer.feature_names = artifacts.get('feature_names', [])
        engineer.categorical_mappings = artifacts.get('categorical_mappings', {})
        engineer._is_fitted = artifacts.get('_is_fitted', True)
        
        return engineer


def get_feature_importance_analysis(
    df: pd.DataFrame,
    target_col: str = 'claim'
) -> pd.DataFrame:
    """
    Analyze feature importance using correlation and mutual information.
    
    Args:
        df: DataFrame with features and target.
        target_col: Name of target column.
        
    Returns:
        DataFrame with feature importance metrics.
    """
    from sklearn.feature_selection import mutual_info_regression
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Drop any rows with NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Calculate correlations
    correlations = X.corrwith(y).abs()
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    # Combine into DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'correlation': correlations.values,
        'mutual_info': mi_series.values
    })
    
    # Normalize scores
    importance_df['correlation_normalized'] = (
        importance_df['correlation'] / importance_df['correlation'].max()
    )
    importance_df['mi_normalized'] = (
        importance_df['mutual_info'] / importance_df['mutual_info'].max()
    )
    
    # Combined score
    importance_df['combined_score'] = (
        importance_df['correlation_normalized'] + importance_df['mi_normalized']
    ) / 2
    
    return importance_df.sort_values('combined_score', ascending=False)

