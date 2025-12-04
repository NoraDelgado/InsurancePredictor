# 🏥 Health Insurance Cost Predictor - Implementation Plan

## Executive Summary

This document outlines a comprehensive step-by-step implementation plan to build a **production-ready health insurance cost prediction system** using state-of-the-art machine learning techniques and modern web development practices. The goal is to predict individual medical charges (claims) based on demographic and health factors, deployed through a sleek, modern web application.

---

## 📋 Table of Contents

1. [Phase 1: Project Foundation](#phase-1-project-foundation)
2. [Phase 2: Data Engineering & Analysis](#phase-2-data-engineering--analysis)
3. [Phase 3: Advanced Feature Engineering](#phase-3-advanced-feature-engineering)
4. [Phase 4: Model Development & Optimization](#phase-4-model-development--optimization)
5. [Phase 5: Model Interpretability](#phase-5-model-interpretability)
6. [Phase 6: Backend API Development](#phase-6-backend-api-development)
7. [Phase 7: Frontend Development](#phase-7-frontend-development)
8. [Phase 8: Testing & Quality Assurance](#phase-8-testing--quality-assurance)
9. [Phase 9: Deployment & DevOps](#phase-9-deployment--devops)
10. [Phase 10: Monitoring & Maintenance](#phase-10-monitoring--maintenance)

---

## Phase 1: Project Foundation

### 1.1 Environment Setup

#### Create Project Structure
```
insurance_project_portfolio/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── interpretability.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   └── predictions.py
│   ├── schemas/
│   │   └── request_models.py
│   └── middleware/
│       └── cors.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── styles/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_interpretation.ipynb
├── models/
│   ├── trained/
│   └── artifacts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── config/
│   ├── config.yaml
│   └── logging.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── .github/
│   └── workflows/
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── README.md
├── tech-spec.md
└── design-philosophy.md
```

#### Initialize Development Environment
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install matplotlib seaborn plotly
pip install shap lime eli5
pip install fastapi uvicorn pydantic
pip install pytest pytest-cov httpx
pip install mlflow optuna
pip install joblib cloudpickle
```

### 1.2 Version Control & Git Configuration

```bash
git init
git add .
git commit -m "Initial project structure"
```

**`.gitignore` essentials:**
```
venv/
__pycache__/
*.pkl
*.joblib
.env
*.pyc
node_modules/
dist/
.DS_Store
```

---

## Phase 2: Data Engineering & Analysis

### 2.1 Comprehensive Data Loading & Validation

```python
# src/data/validation.py
import pandas as pd
from typing import Tuple, Dict, Any
import numpy as np

class DataValidator:
    """Validate and profile the insurance dataset."""
    
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
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_report: Dict[str, Any] = {}
    
    def validate_schema(self) -> bool:
        """Validate DataFrame schema matches expected structure."""
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        extra_cols = set(self.df.columns) - set(self.EXPECTED_COLUMNS) - {'Id'}
        
        self.validation_report['schema'] = {
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols),
            'is_valid': len(missing_cols) == 0
        }
        return len(missing_cols) == 0
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        report = {
            'total_records': len(self.df),
            'duplicates': self.df.duplicated().sum(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        }
        
        # Statistical outlier detection using IQR
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((self.df[col] < Q1 - 1.5 * IQR) | 
                           (self.df[col] > Q3 + 1.5 * IQR)).sum()
            outliers[col] = outlier_count
        
        report['outliers'] = outliers
        self.validation_report['data_quality'] = report
        return report
    
    def generate_profile(self) -> pd.DataFrame:
        """Generate comprehensive data profile."""
        profile = self.df.describe(include='all').T
        profile['missing'] = self.df.isnull().sum()
        profile['missing_pct'] = self.df.isnull().sum() / len(self.df) * 100
        profile['dtype'] = self.df.dtypes
        return profile
```

### 2.2 Exploratory Data Analysis (EDA)

```python
# notebooks/01_exploratory_analysis.ipynb content structure

"""
## 1. Data Overview
- Load and inspect dataset
- Check data types and memory usage
- Identify missing values and duplicates

## 2. Univariate Analysis
- Distribution plots for numerical features (age, bmi, bloodpressure, children, claim)
- Count plots for categorical features (gender, diabetic, smoker, region)
- Box plots for outlier detection

## 3. Bivariate Analysis
- Correlation heatmap for numerical features
- Scatter plots: age vs claim, bmi vs claim (colored by smoker status)
- Bar plots: average claim by gender, smoker, diabetic, region

## 4. Multivariate Analysis
- Pair plots colored by smoker status
- 3D scatter: age x bmi x claim (colored by smoker)
- Pivot tables: claim by region x diabetic, gender x smoker

## 5. Statistical Tests
- T-test: claim differences between smokers and non-smokers
- ANOVA: claim differences across regions
- Chi-square: independence tests for categorical variables

## 6. Key Insights Documentation
- Smoking is the strongest predictor of high claims
- BMI shows positive correlation with claims
- Age shows moderate positive correlation with claims
- Regional differences exist but are less significant
"""
```

### 2.3 Advanced Data Preprocessing

```python
# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Tuple, Dict, Optional
import joblib

class InsuranceDataPreprocessor:
    """
    Production-ready data preprocessing pipeline for insurance data.
    Handles missing values, encoding, scaling, and outlier treatment.
    """
    
    def __init__(self, scaling_method: str = 'robust'):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[RobustScaler | StandardScaler] = None
        self.scaling_method = scaling_method
        self.numeric_columns = ['age', 'bmi', 'bloodpressure', 'children']
        self.categorical_columns = ['gender', 'diabetic', 'smoker']
        self.feature_columns = self.numeric_columns + self.categorical_columns
        self.imputer: Optional[KNNImputer] = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing pipeline and transform data."""
        df = df.copy()
        
        # 1. Handle missing values
        df = self._handle_missing_values(df, fit=True)
        
        # 2. Encode categorical variables
        df = self._encode_categoricals(df, fit=True)
        
        # 3. Scale numerical features
        df = self._scale_numericals(df, fit=True)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        df = df.copy()
        df = self._handle_missing_values(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numericals(df, fit=False)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values using KNN imputation for numerics."""
        # Separate numeric and categorical
        numeric_data = df[self.numeric_columns].copy()
        
        if fit:
            # Use KNN imputer for better imputation
            self.imputer = KNNImputer(n_neighbors=5)
            numeric_data = pd.DataFrame(
                self.imputer.fit_transform(numeric_data),
                columns=self.numeric_columns,
                index=df.index
            )
        else:
            numeric_data = pd.DataFrame(
                self.imputer.transform(numeric_data),
                columns=self.numeric_columns,
                index=df.index
            )
        
        df[self.numeric_columns] = numeric_data
        
        # Handle categorical missing values with mode
        for col in self.categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        for col in self.categorical_columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def _scale_numericals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using RobustScaler (outlier-resistant)."""
        if fit:
            if self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        else:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df
    
    def save_artifacts(self, path: str):
        """Save preprocessing artifacts for production use."""
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        joblib.dump(artifacts, path)
    
    @classmethod
    def load_artifacts(cls, path: str) -> 'InsuranceDataPreprocessor':
        """Load preprocessing artifacts from saved file."""
        artifacts = joblib.load(path)
        preprocessor = cls()
        preprocessor.label_encoders = artifacts['label_encoders']
        preprocessor.scaler = artifacts['scaler']
        preprocessor.imputer = artifacts['imputer']
        preprocessor.feature_columns = artifacts['feature_columns']
        preprocessor.numeric_columns = artifacts['numeric_columns']
        preprocessor.categorical_columns = artifacts['categorical_columns']
        return preprocessor
```

---

## Phase 3: Advanced Feature Engineering

### 3.1 Feature Creation Strategy

```python
# src/features/engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict

class InsuranceFeatureEngineer:
    """
    Create advanced features for insurance cost prediction.
    Implements domain-driven feature engineering.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features."""
        df = df.copy()
        
        # 1. Age-based features
        df = self._create_age_features(df)
        
        # 2. BMI-based features
        df = self._create_bmi_features(df)
        
        # 3. Blood pressure features
        df = self._create_bp_features(df)
        
        # 4. Interaction features
        df = self._create_interaction_features(df)
        
        # 5. Risk score composite
        df = self._create_risk_score(df)
        
        self.feature_names = df.columns.tolist()
        self.fitted = True
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-derived features."""
        # Age groups (domain-specific risk categories)
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['young', 'young_adult', 'middle', 'middle_senior', 'senior', 'elderly']
        )
        
        # Age squared (captures non-linear relationship)
        df['age_squared'] = df['age'] ** 2
        
        # Age decade
        df['age_decade'] = (df['age'] // 10).astype(int)
        
        return df
    
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-derived features using medical classifications."""
        # WHO BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['underweight', 'normal', 'overweight', 'obese_1', 'obese_2', 'obese_3']
        )
        
        # BMI risk flag (high risk if obese)
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
        
        # BMI squared
        df['bmi_squared'] = df['bmi'] ** 2
        
        return df
    
    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create blood pressure derived features."""
        # Hypertension categories
        df['bp_category'] = pd.cut(
            df['bloodpressure'],
            bins=[0, 80, 90, 100, 110, 200],
            labels=['low', 'normal', 'elevated', 'high_1', 'high_2']
        )
        
        # Hypertension flag
        df['is_hypertensive'] = (df['bloodpressure'] > 90).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key predictors."""
        # Smoker x BMI (smokers with high BMI are highest risk)
        df['smoker_bmi'] = df['smoker'] * df['bmi']
        
        # Smoker x Age
        df['smoker_age'] = df['smoker'] * df['age']
        
        # Age x BMI
        df['age_bmi'] = df['age'] * df['bmi']
        
        # Diabetic x BMI
        df['diabetic_bmi'] = df['diabetic'] * df['bmi']
        
        # Smoker x Diabetic (compound risk)
        df['smoker_diabetic'] = df['smoker'] * df['diabetic']
        
        # Children x Age (family planning impact)
        df['children_age'] = df['children'] * df['age']
        
        return df
    
    def _create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk score based on domain knowledge."""
        # Weighted risk score based on medical risk factors
        df['risk_score'] = (
            df['smoker'] * 3.0 +  # Smoking is highest risk
            df['is_obese'] * 1.5 +
            df['is_hypertensive'] * 1.2 +
            df['diabetic'] * 1.3 +
            (df['age'] > 45).astype(int) * 1.0 +
            (df['children'] > 2).astype(int) * 0.5
        )
        
        return df
```

### 3.2 Feature Selection

```python
# Feature selection using multiple methods
from sklearn.feature_selection import (
    mutual_info_regression,
    SelectKBest,
    f_regression,
    RFE
)
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """Multi-method feature selection for optimal feature set."""
    
    def __init__(self, n_features: int = 15):
        self.n_features = n_features
        self.selected_features: List[str] = []
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Use ensemble of methods to select best features."""
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X, y)
        mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Method 2: F-regression
        f_scores, _ = f_regression(X, y)
        f_ranking = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
        
        # Method 3: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_ranking = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Combine rankings (Borda count)
        rankings = pd.DataFrame({
            'mi_rank': range(len(mi_ranking)),
            'f_rank': X.columns.map(lambda x: list(f_ranking.index).index(x)),
            'rf_rank': X.columns.map(lambda x: list(rf_ranking.index).index(x))
        }, index=mi_ranking.index)
        
        rankings['avg_rank'] = rankings.mean(axis=1)
        self.selected_features = rankings.sort_values('avg_rank').head(self.n_features).index.tolist()
        
        return self.selected_features
```

---

## Phase 4: Model Development & Optimization

### 4.1 Model Zoo Implementation

```python
# src/models/training.py
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
import optuna
from typing import Dict, Any, Tuple
import joblib

class InsuranceModelTrainer:
    """
    SOTA model training pipeline with hyperparameter optimization.
    Includes ensemble methods and automated model selection.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_model_name = None
        self.results: Dict[str, Dict[str, float]] = {}
    
    def get_base_models(self) -> Dict[str, Any]:
        """Initialize base models with default parameters."""
        return {
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
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
                n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
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
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> XGBRegressor:
        """Optimize XGBoost using Optuna."""
        
        def objective(trial):
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
                'n_jobs': -1
            }
            
            model = XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_model = XGBRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1)
        best_model.fit(X, y)
        
        return best_model
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> LGBMRegressor:
        """Optimize LightGBM using Optuna."""
        
        def objective(trial):
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
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_model = LGBMRegressor(**study.best_params, random_state=self.random_state, n_jobs=-1, verbose=-1)
        best_model.fit(X, y)
        
        return best_model
    
    def create_stacking_ensemble(self, X: np.ndarray, y: np.ndarray) -> StackingRegressor:
        """Create a stacking ensemble of best models."""
        base_estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
            ('lgbm', LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)),
            ('cat', CatBoostRegressor(iterations=100, random_state=self.random_state, verbose=0))
        ]
        
        stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(),
            cv=5,
            n_jobs=-1
        )
        stacking_model.fit(X, y)
        return stacking_model
    
    def train_and_evaluate_all(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Train all models and evaluate performance."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        models = self.get_base_models()
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            self.models[name] = model
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        self.results = results
        
        return results
```

### 4.2 Model Evaluation Framework

```python
# src/models/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Tuple

class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""
    
    def __init__(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'R² Score': r2_score(self.y_test, self.y_pred),
            'MAE': mean_absolute_error(self.y_test, self.y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'MAPE': mean_absolute_percentage_error(self.y_test, self.y_pred) * 100,
            'Max Error': np.max(np.abs(self.y_test - self.y_pred)),
            'Median AE': np.median(np.abs(self.y_test - self.y_pred))
        }
    
    def plot_residuals(self, save_path: str = None):
        """Plot residual analysis charts."""
        residuals = self.y_test - self.y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(self.y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Residual histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        
        # Actual vs Predicted
        axes[1, 0].scatter(self.y_test, self.y_pred, alpha=0.5)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Actual vs Predicted')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curve(self, save_path: str = None):
        """Plot learning curve to detect overfitting/underfitting."""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.plot(train_sizes, train_mean, label='Training Score')
        plt.plot(train_sizes, test_mean, label='Cross-Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

## Phase 5: Model Interpretability

### 5.1 SHAP Analysis

```python
# src/models/interpretability.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

class ModelInterpreter:
    """
    Model interpretability using SHAP and other explainability methods.
    Critical for healthcare/insurance domain to ensure trust and compliance.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: list):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def compute_shap_values(self, X_explain: pd.DataFrame = None):
        """Compute SHAP values for model explanations."""
        if X_explain is None:
            X_explain = self.X_train
        
        # Use TreeExplainer for tree-based models
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fall back to KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict, 
                shap.sample(self.X_train, 100)
            )
        
        self.shap_values = self.explainer.shap_values(X_explain)
        return self.shap_values
    
    def plot_summary(self, save_path: str = None):
        """Plot SHAP summary (beeswarm) plot."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_train, feature_names=self.feature_names)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, save_path: str = None):
        """Plot SHAP feature importance bar chart."""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_train, 
            feature_names=self.feature_names,
            plot_type="bar"
        )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_prediction(self, instance: pd.DataFrame, idx: int = 0):
        """Generate explanation for a single prediction."""
        if self.shap_values is None:
            self.compute_shap_values(instance)
        
        # Waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[idx],
                base_values=self.explainer.expected_value,
                data=instance.iloc[idx].values,
                feature_names=self.feature_names
            )
        )
    
    def plot_dependence(self, feature: str, interaction_feature: str = None, save_path: str = None):
        """Plot SHAP dependence plot for a feature."""
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            self.shap_values, 
            self.X_train,
            feature_names=self.feature_names,
            interaction_index=interaction_feature
        )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as a DataFrame."""
        importance = np.abs(self.shap_values).mean(0)
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
```

---

## Phase 6: Backend API Development

### 6.1 FastAPI Implementation

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager

# Pydantic models for request/response validation
class InsuranceInput(BaseModel):
    """Input schema for insurance prediction."""
    age: int = Field(..., ge=18, le=100, description="Age of the individual")
    gender: str = Field(..., description="Gender (male/female)")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    bloodpressure: int = Field(..., ge=60, le=200, description="Blood pressure reading")
    diabetic: str = Field(..., description="Diabetic status (Yes/No)")
    children: int = Field(..., ge=0, le=10, description="Number of children")
    smoker: str = Field(..., description="Smoking status (Yes/No)")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Gender must be male or female')
        return v.lower()
    
    @validator('diabetic', 'smoker')
    def validate_yes_no(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('Value must be Yes or No')
        return v.capitalize()
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "gender": "male",
                "bmi": 28.5,
                "bloodpressure": 120,
                "diabetic": "No",
                "children": 2,
                "smoker": "No"
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predicted_cost: float
    confidence_interval: dict
    risk_factors: List[dict]
    recommendation: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str

# Model loading
class ModelService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
    
    def load_models(self, model_path: str, preprocessor_path: str):
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def predict(self, input_data: InsuranceInput) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Preprocess
        processed = self.preprocessor.transform(df)
        
        # Predict
        prediction = self.model.predict(processed)[0]
        
        # Generate risk analysis
        risk_factors = self._analyze_risk_factors(input_data)
        
        return {
            "predicted_cost": round(float(prediction), 2),
            "confidence_interval": {
                "lower": round(float(prediction * 0.85), 2),
                "upper": round(float(prediction * 1.15), 2)
            },
            "risk_factors": risk_factors,
            "recommendation": self._generate_recommendation(risk_factors)
        }
    
    def _analyze_risk_factors(self, input_data: InsuranceInput) -> List[dict]:
        factors = []
        
        if input_data.smoker == "Yes":
            factors.append({"factor": "Smoking", "impact": "High", "contribution": "+40-60%"})
        
        if input_data.bmi >= 30:
            factors.append({"factor": "Obesity", "impact": "Medium", "contribution": "+15-25%"})
        
        if input_data.bloodpressure > 90:
            factors.append({"factor": "Hypertension", "impact": "Medium", "contribution": "+10-20%"})
        
        if input_data.diabetic == "Yes":
            factors.append({"factor": "Diabetes", "impact": "Medium", "contribution": "+10-15%"})
        
        if input_data.age > 45:
            factors.append({"factor": "Age", "impact": "Medium", "contribution": "+10-20%"})
        
        return factors
    
    def _generate_recommendation(self, risk_factors: List[dict]) -> str:
        if not risk_factors:
            return "Your health profile indicates low risk. Maintain healthy habits!"
        
        high_risk = [f for f in risk_factors if f["impact"] == "High"]
        if high_risk:
            return "Consider lifestyle changes to reduce major risk factors like smoking."
        
        return "Focus on managing identified risk factors for optimal health outcomes."

model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_service.load_models(
        "models/trained/best_model.pkl",
        "models/artifacts/preprocessor.pkl"
    )
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Health Insurance Cost Predictor API",
    description="Predict health insurance costs based on demographic and health factors",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_insurance_cost(input_data: InsuranceInput):
    """
    Predict health insurance cost based on input features.
    
    Returns predicted cost, confidence interval, risk factors, and recommendations.
    """
    try:
        result = model_service.predict(input_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Health Insurance Cost Predictor API",
        "docs": "/docs",
        "health": "/health"
    }
```

---

## Phase 7: Frontend Development

### 7.1 Modern React/Next.js Frontend

```typescript
// frontend/src/components/PredictionForm.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface FormData {
  age: number;
  gender: string;
  bmi: number;
  bloodpressure: number;
  diabetic: string;
  children: number;
  smoker: string;
}

interface PredictionResult {
  predicted_cost: number;
  confidence_interval: { lower: number; upper: number };
  risk_factors: Array<{ factor: string; impact: string; contribution: string }>;
  recommendation: string;
}

export const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    age: 30,
    gender: 'male',
    bmi: 25.0,
    bloodpressure: 80,
    diabetic: 'No',
    children: 0,
    smoker: 'No'
  });
  
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="prediction-container">
      <form onSubmit={handleSubmit} className="prediction-form">
        {/* Form fields with modern styling */}
      </form>
      
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="result-card"
          >
            <h2>Predicted Annual Cost</h2>
            <div className="cost-display">
              ${result.predicted_cost.toLocaleString()}
            </div>
            <div className="confidence-interval">
              Range: ${result.confidence_interval.lower.toLocaleString()} - 
              ${result.confidence_interval.upper.toLocaleString()}
            </div>
            
            {result.risk_factors.length > 0 && (
              <div className="risk-factors">
                <h3>Risk Factors</h3>
                {result.risk_factors.map((factor, idx) => (
                  <div key={idx} className={`risk-item risk-${factor.impact.toLowerCase()}`}>
                    <span className="factor-name">{factor.factor}</span>
                    <span className="factor-impact">{factor.contribution}</span>
                  </div>
                ))}
              </div>
            )}
            
            <div className="recommendation">
              <p>{result.recommendation}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
```

### 7.2 Styling with Modern CSS

```css
/* frontend/src/styles/main.css */
:root {
  /* Elegant color palette - inspired by medical/health aesthetics */
  --primary-gradient: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
  --secondary-gradient: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
  --background-gradient: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  
  --surface-1: rgba(30, 41, 59, 0.8);
  --surface-2: rgba(51, 65, 85, 0.6);
  --surface-glass: rgba(255, 255, 255, 0.05);
  
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-accent: #2dd4bf;
  
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  
  --font-display: 'Clash Display', 'SF Pro Display', system-ui, sans-serif;
  --font-body: 'Inter', 'SF Pro Text', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
  
  --border-radius-sm: 8px;
  --border-radius-md: 12px;
  --border-radius-lg: 20px;
  --border-radius-xl: 28px;
  
  --shadow-glow: 0 0 60px rgba(45, 212, 191, 0.15);
  --shadow-card: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-body);
  background: var(--background-gradient);
  color: var(--text-primary);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Hero Section */
.hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  position: relative;
}

.hero::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(45, 212, 191, 0.15), transparent),
    radial-gradient(ellipse 60% 40% at 80% 60%, rgba(124, 58, 237, 0.1), transparent);
  pointer-events: none;
}

.hero-title {
  font-family: var(--font-display);
  font-size: clamp(2.5rem, 6vw, 4.5rem);
  font-weight: 600;
  text-align: center;
  background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
  letter-spacing: -0.02em;
}

.hero-subtitle {
  font-size: clamp(1rem, 2vw, 1.25rem);
  color: var(--text-secondary);
  text-align: center;
  max-width: 600px;
  line-height: 1.7;
}

/* Prediction Form Card */
.prediction-container {
  width: 100%;
  max-width: 800px;
  margin: 3rem auto;
  perspective: 1000px;
}

.prediction-form {
  background: var(--surface-glass);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--border-radius-xl);
  padding: 3rem;
  box-shadow: var(--shadow-card), var(--shadow-glow);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.form-input,
.form-select {
  background: var(--surface-2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--border-radius-md);
  padding: 1rem 1.25rem;
  font-size: 1rem;
  color: var(--text-primary);
  transition: all 0.3s ease;
  font-family: var(--font-body);
}

.form-input:focus,
.form-select:focus {
  outline: none;
  border-color: var(--text-accent);
  box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2);
}

.form-input::placeholder {
  color: var(--text-secondary);
}

/* Submit Button */
.submit-button {
  width: 100%;
  margin-top: 2rem;
  padding: 1.25rem 2rem;
  font-size: 1.125rem;
  font-weight: 600;
  font-family: var(--font-display);
  color: #0f172a;
  background: var(--primary-gradient);
  border: none;
  border-radius: var(--border-radius-lg);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.submit-button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transition: left 0.5s ease;
}

.submit-button:hover::before {
  left: 100%;
}

.submit-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 40px rgba(45, 212, 191, 0.3);
}

.submit-button:active {
  transform: translateY(0);
}

/* Result Card */
.result-card {
  margin-top: 2rem;
  background: var(--surface-glass);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--border-radius-xl);
  padding: 3rem;
  text-align: center;
}

.cost-display {
  font-family: var(--font-display);
  font-size: clamp(2.5rem, 8vw, 4rem);
  font-weight: 700;
  background: var(--primary-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 1rem 0;
}

.confidence-interval {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--text-secondary);
  padding: 0.75rem 1.5rem;
  background: var(--surface-2);
  border-radius: var(--border-radius-md);
  display: inline-block;
}

/* Risk Factors */
.risk-factors {
  margin-top: 2rem;
  text-align: left;
}

.risk-factors h3 {
  font-family: var(--font-display);
  font-size: 1.25rem;
  margin-bottom: 1rem;
  color: var(--text-primary);
}

.risk-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  margin-bottom: 0.5rem;
  border-radius: var(--border-radius-md);
  border-left: 4px solid;
}

.risk-high {
  background: rgba(239, 68, 68, 0.1);
  border-color: var(--danger);
}

.risk-medium {
  background: rgba(245, 158, 11, 0.1);
  border-color: var(--warning);
}

.risk-low {
  background: rgba(16, 185, 129, 0.1);
  border-color: var(--success);
}

.factor-name {
  font-weight: 500;
}

.factor-impact {
  font-family: var(--font-mono);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

/* Recommendation */
.recommendation {
  margin-top: 2rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, rgba(45, 212, 191, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
  border-radius: var(--border-radius-lg);
  border: 1px solid rgba(45, 212, 191, 0.2);
}

.recommendation p {
  color: var(--text-primary);
  line-height: 1.7;
}

/* Animations */
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: var(--shadow-card), 0 0 30px rgba(45, 212, 191, 0.1); }
  50% { box-shadow: var(--shadow-card), 0 0 60px rgba(45, 212, 191, 0.2); }
}

.prediction-form {
  animation: pulse-glow 4s ease-in-out infinite;
}

/* Loading State */
.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--surface-2);
  border-top-color: var(--text-accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 2rem auto;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
  .prediction-form {
    padding: 2rem 1.5rem;
  }
  
  .form-grid {
    grid-template-columns: 1fr;
  }
}
```

---

## Phase 8: Testing & Quality Assurance

### 8.1 Testing Strategy

```python
# tests/unit/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import InsuranceDataPreprocessor

class TestInsuranceDataPreprocessor:
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [25, 35, np.nan, 45],
            'gender': ['male', 'female', 'male', 'female'],
            'bmi': [22.5, 28.0, 32.0, np.nan],
            'bloodpressure': [80, 90, 100, 85],
            'diabetic': ['No', 'Yes', 'No', 'Yes'],
            'children': [0, 2, 1, 3],
            'smoker': ['No', 'No', 'Yes', 'No']
        })
    
    @pytest.fixture
    def preprocessor(self):
        return InsuranceDataPreprocessor()
    
    def test_handles_missing_values(self, sample_data, preprocessor):
        result = preprocessor.fit_transform(sample_data)
        assert result.isnull().sum().sum() == 0
    
    def test_encodes_categoricals(self, sample_data, preprocessor):
        result = preprocessor.fit_transform(sample_data)
        assert result['gender'].dtype in [np.int64, np.int32]
        assert result['smoker'].dtype in [np.int64, np.int32]
    
    def test_scales_numericals(self, sample_data, preprocessor):
        result = preprocessor.fit_transform(sample_data)
        # Check approximate normalization (mean close to 0)
        for col in preprocessor.numeric_columns:
            assert abs(result[col].mean()) < 1.0  # RobustScaler won't necessarily center at 0

# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

class TestPredictionAPI:
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_predict_valid_input(self):
        payload = {
            "age": 35,
            "gender": "male",
            "bmi": 28.5,
            "bloodpressure": 120,
            "diabetic": "No",
            "children": 2,
            "smoker": "No"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_cost" in response.json()
    
    def test_predict_invalid_age(self):
        payload = {
            "age": -5,  # Invalid
            "gender": "male",
            "bmi": 28.5,
            "bloodpressure": 120,
            "diabetic": "No",
            "children": 2,
            "smoker": "No"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
```

---

## Phase 9: Deployment & DevOps

### 9.1 Docker Configuration

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/trained/best_model.pkl
      - PREPROCESSOR_PATH=/app/models/artifacts/preprocessor.pkl
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend
```

### 9.3 CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker images
        run: docker-compose build
      
      - name: Push to registry
        if: github.ref == 'refs/heads/main'
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker-compose push

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Add deployment commands here
          echo "Deploying to production..."
```

---

## Phase 10: Monitoring & Maintenance

### 10.1 MLOps with MLflow

```python
# src/monitoring/mlflow_tracking.py
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

class MLFlowTracker:
    """Track experiments and model performance with MLflow."""
    
    def __init__(self, experiment_name: str = "insurance_predictor"):
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(
        self, 
        model,
        model_name: str,
        params: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list
    ):
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Make predictions and log metrics
            y_pred = model.predict(X_test)
            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(np.mean((y_test - y_pred) ** 2))
            }
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                mlflow.log_dict(importance_dict, "feature_importance.json")
            
            return mlflow.active_run().info.run_id
```

### 10.2 Model Performance Monitoring

```python
# src/monitoring/drift_detection.py
import numpy as np
from scipy import stats
from typing import Dict, List

class DataDriftDetector:
    """Detect data drift in production."""
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """Detect drift using KS test for each feature."""
        drift_report = {}
        
        for i, feature in enumerate(self.feature_names):
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            
            drift_report[feature] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
        
        return drift_report
```

---

## 📊 Timeline & Milestones

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 1 day | Project structure, environment setup |
| Phase 2 | 2 days | Data validation, EDA, preprocessing pipeline |
| Phase 3 | 1 day | Feature engineering module |
| Phase 4 | 3 days | Model development, optimization, selection |
| Phase 5 | 1 day | SHAP analysis, interpretability |
| Phase 6 | 2 days | FastAPI backend, API documentation |
| Phase 7 | 3 days | React frontend, UI/UX design |
| Phase 8 | 2 days | Testing suite, code coverage |
| Phase 9 | 2 days | Docker, CI/CD, deployment |
| Phase 10 | 1 day | Monitoring, MLflow integration |

**Total Estimated Duration: ~18 days**

---

## ✅ Success Criteria

1. **Model Performance**
   - R² Score ≥ 0.85
   - MAE < $2,500
   - MAPE < 15%

2. **API Performance**
   - Response time < 200ms
   - 99.9% uptime
   - Proper error handling

3. **Frontend Quality**
   - Lighthouse score > 90
   - Mobile responsive
   - Accessible (WCAG 2.1 AA)

4. **Code Quality**
   - Test coverage > 80%
   - Type annotations
   - Documentation coverage

---

## 🚀 Next Steps

1. Review and approve this implementation plan
2. Set up development environment
3. Begin Phase 1: Project Foundation
4. Daily standups to track progress
5. Iterative development with continuous feedback

---

*This implementation plan combines industry best practices, SOTA machine learning techniques, and modern web development standards to deliver a production-ready health insurance cost prediction system.*

