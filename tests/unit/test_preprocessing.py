"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import InsuranceDataPreprocessor, prepare_data_for_training


class TestInsuranceDataPreprocessor:
    """Tests for InsuranceDataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample test data."""
        return pd.DataFrame({
            'age': [25, 35, np.nan, 45, 55],
            'gender': ['male', 'female', 'male', 'female', 'male'],
            'bmi': [22.5, 28.0, 32.0, np.nan, 26.5],
            'bloodpressure': [80, 90, 100, 85, 95],
            'diabetic': ['No', 'Yes', 'No', 'Yes', 'No'],
            'children': [0, 2, 1, 3, 0],
            'smoker': ['No', 'No', 'Yes', 'No', 'Yes'],
            'claim': [5000, 15000, 25000, 12000, 30000]
        })
    
    @pytest.fixture
    def preprocessor(self) -> InsuranceDataPreprocessor:
        """Create preprocessor instance."""
        return InsuranceDataPreprocessor()
    
    def test_initialization(self, preprocessor: InsuranceDataPreprocessor):
        """Test preprocessor initializes correctly."""
        assert preprocessor.scaling_method == 'robust'
        assert preprocessor.imputer_neighbors == 5
        assert not preprocessor.is_fitted
    
    def test_fit_marks_as_fitted(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that fit() marks preprocessor as fitted."""
        preprocessor.fit(sample_data)
        assert preprocessor.is_fitted
    
    def test_transform_before_fit_raises_error(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that transform() before fit() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_handles_missing_values(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that missing values are handled."""
        result = preprocessor.fit_transform(sample_data)
        assert result.isnull().sum().sum() == 0
    
    def test_encodes_categoricals(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that categorical columns are encoded."""
        result = preprocessor.fit_transform(sample_data)
        
        # Check that categorical columns are now numeric
        assert result['gender'].dtype in [np.int64, np.int32, np.float64]
        assert result['smoker'].dtype in [np.int64, np.int32, np.float64]
        assert result['diabetic'].dtype in [np.int64, np.int32, np.float64]
    
    def test_scales_numericals(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that numeric columns are scaled."""
        result = preprocessor.fit_transform(sample_data)
        
        # After scaling, values should be roughly centered
        # RobustScaler centers around median, so values should be in reasonable range
        for col in ['age', 'bmi', 'bloodpressure', 'children']:
            assert result[col].abs().max() < 10  # Scaled values should be small
    
    def test_fit_transform_produces_correct_shape(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that output has expected number of columns."""
        result = preprocessor.fit_transform(sample_data)
        expected_cols = len(preprocessor.NUMERIC_COLUMNS) + len(preprocessor.CATEGORICAL_COLUMNS)
        assert result.shape[1] == expected_cols
    
    def test_feature_names(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that feature names are correctly tracked."""
        preprocessor.fit(sample_data)
        feature_names = preprocessor.get_feature_names()
        
        assert 'age' in feature_names
        assert 'gender' in feature_names
        assert 'smoker' in feature_names
    
    def test_save_and_load(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame, tmp_path):
        """Test saving and loading preprocessor."""
        # Fit and save
        preprocessor.fit(sample_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(save_path)
        
        # Load and verify
        loaded = InsuranceDataPreprocessor.load(save_path)
        assert loaded.is_fitted
        assert loaded.scaling_method == preprocessor.scaling_method
    
    def test_transform_consistency(self, preprocessor: InsuranceDataPreprocessor, sample_data: pd.DataFrame):
        """Test that multiple transforms produce consistent results."""
        preprocessor.fit(sample_data)
        
        result1 = preprocessor.transform(sample_data)
        result2 = preprocessor.transform(sample_data)
        
        pd.testing.assert_frame_equal(result1, result2)


class TestPrepareDataForTraining:
    """Tests for prepare_data_for_training function."""
    
    @pytest.fixture
    def full_data(self) -> pd.DataFrame:
        """Create sample data with all required columns."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'Id': range(n),
            'age': np.random.randint(18, 65, n).astype(float),
            'gender': np.random.choice(['male', 'female'], n),
            'bmi': np.random.uniform(18, 40, n),
            'bloodpressure': np.random.randint(70, 120, n),
            'diabetic': np.random.choice(['Yes', 'No'], n),
            'children': np.random.randint(0, 5, n),
            'smoker': np.random.choice(['Yes', 'No'], n),
            'region': np.random.choice(['northeast', 'southeast', 'southwest', 'northwest'], n),
            'claim': np.random.uniform(1000, 50000, n)
        })
    
    def test_returns_correct_components(self, full_data: pd.DataFrame):
        """Test that function returns all expected components."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(full_data)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, InsuranceDataPreprocessor)
    
    def test_train_test_split_ratio(self, full_data: pd.DataFrame):
        """Test that train/test split follows specified ratio."""
        X_train, X_test, y_train, y_test, _ = prepare_data_for_training(
            full_data, test_size=0.2
        )
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        assert 0.18 <= test_ratio <= 0.22  # Allow small variance
    
    def test_drops_id_column(self, full_data: pd.DataFrame):
        """Test that Id column is dropped."""
        X_train, X_test, _, _, _ = prepare_data_for_training(full_data)
        
        assert 'Id' not in X_train.columns
        assert 'Id' not in X_test.columns
    
    def test_drops_region_column(self, full_data: pd.DataFrame):
        """Test that region column is dropped."""
        X_train, X_test, _, _, _ = prepare_data_for_training(full_data)
        
        assert 'region' not in X_train.columns
        assert 'region' not in X_test.columns
    
    def test_preprocessor_is_fitted(self, full_data: pd.DataFrame):
        """Test that returned preprocessor is fitted."""
        _, _, _, _, preprocessor = prepare_data_for_training(full_data)
        assert preprocessor.is_fitted

