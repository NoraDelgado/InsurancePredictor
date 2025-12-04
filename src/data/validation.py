"""
Data Validation Module for Insurance Cost Predictor.

This module provides comprehensive data validation, profiling,
and quality assessment for the insurance dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """
    Validate and profile the insurance dataset.
    
    Performs schema validation, data quality checks, and statistical profiling.
    """
    
    EXPECTED_COLUMNS = [
        'age', 'gender', 'bmi', 'bloodpressure', 
        'diabetic', 'children', 'smoker', 'region', 'claim'
    ]
    
    COLUMN_TYPES = {
        'age': 'float64',
        'gender': 'object',
        'bmi': 'float64',
        'bloodpressure': 'int64',
        'diabetic': 'object',
        'children': 'int64',
        'smoker': 'object',
        'region': 'object',
        'claim': 'float64'
    }
    
    VALID_VALUES = {
        'gender': ['male', 'female'],
        'diabetic': ['Yes', 'No'],
        'smoker': ['Yes', 'No'],
        'region': ['northeast', 'northwest', 'southeast', 'southwest']
    }
    
    VALUE_RANGES = {
        'age': (18, 100),
        'bmi': (10.0, 60.0),
        'bloodpressure': (60, 200),
        'children': (0, 10),
        'claim': (0, 100000)
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the validator with a DataFrame.
        
        Args:
            df: The insurance DataFrame to validate.
        """
        self.df = df.copy()
        self.validation_report: Dict[str, Any] = {}
    
    def validate_schema(self) -> ValidationResult:
        """
        Validate DataFrame schema matches expected structure.
        
        Returns:
            ValidationResult with schema validation details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check for required columns (excluding 'Id' which is optional)
        required_cols = set(self.EXPECTED_COLUMNS)
        actual_cols = set(self.df.columns)
        
        missing_cols = required_cols - actual_cols
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        extra_cols = actual_cols - required_cols - {'Id'}
        if extra_cols:
            warnings.append(f"Extra columns found (will be ignored): {extra_cols}")
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats={
                'expected_columns': list(required_cols),
                'actual_columns': list(actual_cols),
                'missing_columns': list(missing_cols),
                'extra_columns': list(extra_cols)
            }
        )
        
        self.validation_report['schema'] = result
        return result
    
    def validate_data_quality(self) -> ValidationResult:
        """
        Comprehensive data quality assessment.
        
        Checks for missing values, duplicates, and outliers.
        
        Returns:
            ValidationResult with data quality details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}
        
        # Basic stats
        stats['total_records'] = len(self.df)
        stats['duplicate_records'] = int(self.df.duplicated().sum())
        
        if stats['duplicate_records'] > 0:
            warnings.append(f"Found {stats['duplicate_records']} duplicate records")
        
        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        
        stats['missing_values'] = missing_counts.to_dict()
        stats['missing_percentage'] = missing_pct.to_dict()
        
        # Flag columns with significant missing data
        high_missing = missing_pct[missing_pct > 5]
        if not high_missing.empty:
            warnings.append(f"Columns with >5% missing data: {high_missing.to_dict()}")
        
        # Outlier detection using IQR for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers: Dict[str, int] = {}
        
        for col in numeric_cols:
            if col == 'Id':
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        stats['outliers'] = outliers
        if outliers:
            warnings.append(f"Outliers detected: {outliers}")
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
        
        self.validation_report['data_quality'] = result
        return result
    
    def validate_value_ranges(self) -> ValidationResult:
        """
        Validate that values fall within expected ranges.
        
        Returns:
            ValidationResult with range validation details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}
        
        # Check numeric ranges
        for col, (min_val, max_val) in self.VALUE_RANGES.items():
            if col not in self.df.columns:
                continue
            
            col_data = self.df[col].dropna()
            below_min = (col_data < min_val).sum()
            above_max = (col_data > max_val).sum()
            
            stats[col] = {
                'min': float(col_data.min()) if len(col_data) > 0 else None,
                'max': float(col_data.max()) if len(col_data) > 0 else None,
                'below_range': int(below_min),
                'above_range': int(above_max)
            }
            
            if below_min > 0:
                warnings.append(f"{col}: {below_min} values below minimum {min_val}")
            if above_max > 0:
                warnings.append(f"{col}: {above_max} values above maximum {max_val}")
        
        # Check categorical valid values
        for col, valid_vals in self.VALID_VALUES.items():
            if col not in self.df.columns:
                continue
            
            col_data = self.df[col].dropna()
            invalid_vals = set(col_data.unique()) - set(valid_vals)
            
            if invalid_vals:
                errors.append(f"{col}: Invalid values found: {invalid_vals}")
            
            stats[col] = {
                'unique_values': list(col_data.unique()),
                'invalid_values': list(invalid_vals) if invalid_vals else []
            }
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
        
        self.validation_report['value_ranges'] = result
        return result
    
    def generate_profile(self) -> pd.DataFrame:
        """
        Generate comprehensive data profile.
        
        Returns:
            DataFrame with profiling statistics for each column.
        """
        profile = self.df.describe(include='all').T
        profile['missing'] = self.df.isnull().sum()
        profile['missing_pct'] = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        profile['dtype'] = self.df.dtypes
        profile['unique'] = self.df.nunique()
        
        return profile
    
    def run_full_validation(self) -> Dict[str, ValidationResult]:
        """
        Run all validation checks.
        
        Returns:
            Dictionary containing all validation results.
        """
        self.validate_schema()
        self.validate_data_quality()
        self.validate_value_ranges()
        
        return self.validation_report
    
    def get_clean_data(self, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Get cleaned version of the data.
        
        Args:
            drop_duplicates: Whether to drop duplicate records.
            
        Returns:
            Tuple of (cleaned DataFrame, dictionary of changes made).
        """
        df = self.df.copy()
        changes: Dict[str, int] = {}
        
        # Remove Id column if present (not needed for modeling)
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
            changes['dropped_id_column'] = 1
        
        # Drop duplicates
        if drop_duplicates:
            original_len = len(df)
            df = df.drop_duplicates()
            changes['duplicates_removed'] = original_len - len(df)
        
        return df, changes


def validate_single_input(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate a single input record for prediction.
    
    Args:
        data: Dictionary containing input features.
        
    Returns:
        ValidationResult indicating if input is valid.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    required_fields = ['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult(is_valid=False, errors=errors)
    
    # Validate age
    age = data.get('age')
    if not isinstance(age, (int, float)) or age < 18 or age > 100:
        errors.append(f"Age must be between 18 and 100, got: {age}")
    
    # Validate gender
    gender = data.get('gender', '').lower()
    if gender not in ['male', 'female']:
        errors.append(f"Gender must be 'male' or 'female', got: {gender}")
    
    # Validate BMI
    bmi = data.get('bmi')
    if not isinstance(bmi, (int, float)) or bmi < 10 or bmi > 60:
        errors.append(f"BMI must be between 10 and 60, got: {bmi}")
    
    # Validate blood pressure
    bp = data.get('bloodpressure')
    if not isinstance(bp, (int, float)) or bp < 60 or bp > 200:
        errors.append(f"Blood pressure must be between 60 and 200, got: {bp}")
    
    # Validate diabetic
    diabetic = data.get('diabetic', '')
    if diabetic not in ['Yes', 'No']:
        errors.append(f"Diabetic must be 'Yes' or 'No', got: {diabetic}")
    
    # Validate children
    children = data.get('children')
    if not isinstance(children, int) or children < 0 or children > 10:
        errors.append(f"Children must be between 0 and 10, got: {children}")
    
    # Validate smoker
    smoker = data.get('smoker', '')
    if smoker not in ['Yes', 'No']:
        errors.append(f"Smoker must be 'Yes' or 'No', got: {smoker}")
    
    is_valid = len(errors) == 0
    
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

