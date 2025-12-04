"""
Training Script for Health Insurance Cost Predictor
Trains an XGBoost model to predict insurance claims based on patient features.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """Load the dataset and display basic information."""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nData types:")
    print(df.dtypes)
    
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)
    
    # Drop the Id column if it exists
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Separate features and target
    X = df.drop('claim', axis=1)
    y = df['claim']
    
    # Define column types
    numerical_cols = ['age', 'bmi', 'bloodpressure', 'children']
    categorical_cols = ['gender', 'diabetic', 'smoker', 'region']
    
    # Handle missing values in numerical columns using median
    print("\nImputing missing values with median...")
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # Encode categorical columns
    print("Encoding categorical columns...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Scale numerical features
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"\nPreprocessed feature shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y, label_encoders, scaler, num_imputer, numerical_cols, categorical_cols


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model and evaluate performance."""
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBoost Model")
    print("=" * 60)
    
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n--- Model Performance ---")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training MAE: ${train_mae:,.2f}")
    print(f"\nTest R² Score: {test_r2:.4f}")
    print(f"Test MAE: ${test_mae:,.2f}")
    print(f"Test RMSE: ${test_rmse:,.2f}")
    
    # Feature importance
    print("\n--- Feature Importance ---")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    return model


def save_artifacts(model, label_encoders, scaler, imputer, numerical_cols, categorical_cols):
    """Save trained model and preprocessing artifacts."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Artifacts")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models/trained', exist_ok=True)
    os.makedirs('models/artifacts', exist_ok=True)
    
    # Save model
    model_path = 'models/trained/best_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = 'models/artifacts/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save imputer
    imputer_path = 'models/artifacts/imputer.pkl'
    joblib.dump(imputer, imputer_path)
    print(f"Imputer saved to: {imputer_path}")
    
    # Save label encoders
    for col, le in label_encoders.items():
        le_path = f'models/artifacts/label_encoder_{col}.pkl'
        joblib.dump(le, le_path)
        print(f"Label encoder for {col} saved to: {le_path}")
    
    # Save column configuration - use the actual order from training
    # The order is: age, gender, bmi, bloodpressure, diabetic, children, smoker, region
    config = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_order': ['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'region']
    }
    config_path = 'models/artifacts/config.pkl'
    joblib.dump(config, config_path)
    print(f"Config saved to: {config_path}")
    
    print("\n[OK] All artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("HEALTH INSURANCE COST PREDICTOR - MODEL TRAINING")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_and_explore_data('data/raw/insurance.csv')
    
    # Preprocess data
    X, y, label_encoders, scaler, imputer, numerical_cols, categorical_cols = preprocess_data(df)
    
    # Split data
    print("\n" + "=" * 60)
    print("Splitting data (80% train, 20% test)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    save_artifacts(model, label_encoders, scaler, imputer, numerical_cols, categorical_cols)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API: uvicorn api.main:app --reload")
    print("2. Start the frontend: cd frontend && npm run dev")


if __name__ == "__main__":
    main()

