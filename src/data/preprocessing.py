"""
Data Preprocessing Module for Insurance Cost Predictor.

This module provides a production-ready preprocessing pipeline
for transforming raw insurance data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, List, Optional, Tuple, Any, Literal
import joblib
from pathlib import Path


class InsuranceDataPreprocessor:
    """
    Production-ready data preprocessing pipeline for insurance data.
    
    Handles missing values, categorical encoding, and numerical scaling
    with support for both training and inference modes.
    
    Attributes:
        scaling_method: Method for scaling numerical features ('robust' or 'standard').
        label_encoders: Dictionary of fitted LabelEncoders for categorical columns.
        scaler: Fitted scaler for numerical columns.
        imputer: Fitted imputer for handling missing values.
    """
    
    NUMERIC_COLUMNS = ['age', 'bmi', 'bloodpressure', 'children']
    CATEGORICAL_COLUMNS = ['gender', 'diabetic', 'smoker']
    TARGET_COLUMN = 'claim'
    
    def __init__(
        self, 
        scaling_method: Literal['robust', 'standard'] = 'robust',
        imputer_neighbors: int = 5
    ):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: Method for scaling ('robust' is more outlier-resistant).
            imputer_neighbors: Number of neighbors for KNN imputation.
        """
        self.scaling_method = scaling_method
        self.imputer_neighbors = imputer_neighbors
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[RobustScaler | StandardScaler] = None
        self.imputer: Optional[KNNImputer] = None
        self.categorical_imputers: Dict[str, str] = {}
        
        self.feature_columns = self.NUMERIC_COLUMNS + self.CATEGORICAL_COLUMNS
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted."""
        return self._is_fitted
    
    def fit(self, df: pd.DataFrame) -> 'InsuranceDataPreprocessor':
        """
        Fit the preprocessing pipeline on training data.
        
        Args:
            df: Training DataFrame with all required columns.
            
        Returns:
            Self for method chaining.
        """
        df = df.copy()
        df = self._select_features(df)
        
        # Fit imputer for numeric columns
        self._fit_numeric_imputer(df)
        
        # Fit categorical imputers (mode values)
        self._fit_categorical_imputers(df)
        
        # Apply imputation before fitting encoders
        df = self._apply_imputation(df)
        
        # Fit label encoders
        self._fit_label_encoders(df)
        
        # Apply encoding before fitting scaler
        df = self._apply_encoding(df)
        
        # Fit scaler
        self._fit_scaler(df)
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Args:
            df: DataFrame to transform.
            
        Returns:
            Transformed DataFrame ready for model input.
            
        Raises:
            RuntimeError: If preprocessor hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        df = df.copy()
        df = self._select_features(df)
        
        # Apply transformations in order
        df = self._apply_imputation(df)
        df = self._apply_encoding(df)
        df = self._apply_scaling(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform data in one step.
        
        Args:
            df: DataFrame to fit and transform.
            
        Returns:
            Transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values (if any transformation was applied).
        
        Currently, target is not transformed, so this returns the input unchanged.
        
        Args:
            y: Predicted values.
            
        Returns:
            Original scale values.
        """
        return y
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the required feature columns."""
        available_cols = [col for col in self.feature_columns if col in df.columns]
        return df[available_cols].copy()
    
    def _fit_numeric_imputer(self, df: pd.DataFrame) -> None:
        """Fit KNN imputer for numeric columns."""
        numeric_data = df[self.NUMERIC_COLUMNS].copy()
        self.imputer = KNNImputer(n_neighbors=self.imputer_neighbors)
        self.imputer.fit(numeric_data)
    
    def _fit_categorical_imputers(self, df: pd.DataFrame) -> None:
        """Store mode values for categorical imputation."""
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                mode_val = df[col].mode()
                self.categorical_imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
    
    def _fit_label_encoders(self, df: pd.DataFrame) -> None:
        """Fit label encoders for categorical columns."""
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
    
    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit scaler for numeric columns."""
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(df[self.NUMERIC_COLUMNS])
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to handle missing values."""
        df = df.copy()
        
        # Impute numeric columns using KNN
        numeric_cols = [col for col in self.NUMERIC_COLUMNS if col in df.columns]
        if numeric_cols and self.imputer is not None:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])
        
        # Impute categorical columns using mode
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns and col in self.categorical_imputers:
                df[col] = df[col].fillna(self.categorical_imputers[col])
        
        return df
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        df = df.copy()
        
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen categories gracefully
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric columns."""
        df = df.copy()
        
        if self.scaler is not None:
            numeric_cols = [col for col in self.NUMERIC_COLUMNS if col in df.columns]
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        return self.feature_columns.copy()
    
    def save(self, path: str | Path) -> None:
        """
        Save preprocessor artifacts to disk.
        
        Args:
            path: Path to save the artifacts.
        """
        path = Path(path)
        
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'categorical_imputers': self.categorical_imputers,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.NUMERIC_COLUMNS,
            'categorical_columns': self.CATEGORICAL_COLUMNS,
            'scaling_method': self.scaling_method,
            'imputer_neighbors': self.imputer_neighbors,
            '_is_fitted': self._is_fitted
        }
        
        joblib.dump(artifacts, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'InsuranceDataPreprocessor':
        """
        Load preprocessor from saved artifacts.
        
        Args:
            path: Path to the saved artifacts.
            
        Returns:
            Loaded InsuranceDataPreprocessor instance.
        """
        path = Path(path)
        artifacts = joblib.load(path)
        
        preprocessor = cls(
            scaling_method=artifacts.get('scaling_method', 'robust'),
            imputer_neighbors=artifacts.get('imputer_neighbors', 5)
        )
        
        preprocessor.label_encoders = artifacts['label_encoders']
        preprocessor.scaler = artifacts['scaler']
        preprocessor.imputer = artifacts['imputer']
        preprocessor.categorical_imputers = artifacts.get('categorical_imputers', {})
        preprocessor.feature_columns = artifacts['feature_columns']
        preprocessor._is_fitted = artifacts.get('_is_fitted', True)
        
        return preprocessor


def prepare_data_for_training(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, InsuranceDataPreprocessor]:
    """
    Prepare data for model training with train/test split.
    
    Args:
        df: Raw DataFrame with all columns.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor).
    """
    from sklearn.model_selection import train_test_split
    
    # Remove Id column if present
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Drop rows with missing target
    df = df.dropna(subset=['claim'])
    
    # Separate features and target
    X = df.drop(columns=['claim', 'region'], errors='ignore')
    y = df['claim']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit preprocessor on training data only
    preprocessor = InsuranceDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def preprocess_single_input(
    data: Dict[str, Any],
    preprocessor: InsuranceDataPreprocessor
) -> np.ndarray:
    """
    Preprocess a single input for prediction.
    
    Args:
        data: Dictionary with input features.
        preprocessor: Fitted preprocessor instance.
        
    Returns:
        Preprocessed feature array ready for model prediction.
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Transform
    processed = preprocessor.transform(df)
    
    return processed.values

