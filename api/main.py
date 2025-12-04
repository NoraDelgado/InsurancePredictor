"""
FastAPI Application for Health Insurance Cost Predictor
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Insurance Cost Predictor API",
    description="API for predicting health insurance charges based on demographic and health factors.",
    version="1.0.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    age: float = Field(..., ge=18, le=100, description="Age of the primary beneficiary")
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body mass index")
    bloodpressure: int = Field(..., ge=60, le=200, description="Blood pressure reading")
    diabetic: str = Field(..., description="Is diabetic: 'Yes' or 'No'")
    children: int = Field(..., ge=0, le=10, description="Number of children")
    smoker: str = Field(..., description="Is smoker: 'Yes' or 'No'")
    region: str = Field(..., description="Region: 'northeast', 'northwest', 'southeast', 'southwest'")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v.lower() not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        return v.lower()

    @field_validator('diabetic', 'smoker')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v not in ['Yes', 'No']:
            raise ValueError("Must be 'Yes' or 'No'")
        return v

    @field_validator('region')
    @classmethod
    def validate_region(cls, v: str) -> str:
        valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
        if v.lower() not in valid_regions:
            raise ValueError(f"Region must be one of: {valid_regions}")
        return v.lower()


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predicted_charge: float = Field(..., description="Estimated health insurance charge")
    model_version: str = Field(..., description="Version of the model")


# Global variables for model and artifacts
model = None
scaler = None
imputer = None
label_encoders = {}
config = None
MODEL_VERSION = "1.0.0"


def get_artifact_path(filename: str) -> str:
    """Get the full path to an artifact file."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'models', 'artifacts', filename)


def get_model_path() -> str:
    """Get the full path to the model file."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'models', 'trained', 'best_model.pkl')


@app.on_event("startup")
async def load_artifacts():
    """Load the trained model and preprocessing artifacts on startup."""
    global model, scaler, imputer, label_encoders, config
    
    try:
        # Load model
        model_path = get_model_path()
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = get_artifact_path('scaler.pkl')
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Load imputer
        imputer_path = get_artifact_path('imputer.pkl')
        logger.info(f"Loading imputer from {imputer_path}")
        imputer = joblib.load(imputer_path)
        
        # Load config
        config_path = get_artifact_path('config.pkl')
        logger.info(f"Loading config from {config_path}")
        config = joblib.load(config_path)
        
        # Load label encoders
        for col in config['categorical_cols']:
            le_path = get_artifact_path(f'label_encoder_{col}.pkl')
            logger.info(f"Loading label encoder for {col}")
            label_encoders[col] = joblib.load(le_path)
        
        logger.info("✅ All artifacts loaded successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Failed to load artifacts: {e}")
        logger.error("Please run train_model.py first to train the model.")
        raise RuntimeError(f"Required artifact not found: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error loading artifacts: {e}")
        raise RuntimeError(f"Failed to load artifacts: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict health insurance cost based on input features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    
    try:
        # Create DataFrame with columns in the correct order from the start
        numerical_cols = config['numerical_cols']
        categorical_cols = config['categorical_cols']
        
        # Build input in the feature order the model expects
        input_data = pd.DataFrame({
            'age': [request.age],
            'bmi': [request.bmi],
            'bloodpressure': [request.bloodpressure],
            'children': [request.children],
            'gender': [request.gender],
            'diabetic': [request.diabetic],
            'smoker': [request.smoker],
            'region': [request.region]
        })
        
        logger.info(f"Input data: {input_data.to_dict()}")
        
        # Apply imputer to numerical columns
        input_data[numerical_cols] = imputer.transform(input_data[numerical_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        
        # Scale numerical columns
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Ensure correct column order using the feature_order from config
        df = input_data[config['feature_order']]
        
        logger.info(f"Processed features: {df.values.tolist()}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        logger.info(f"Prediction: ${prediction:,.2f}")
        
        return PredictionResponse(
            predicted_charge=round(float(prediction), 2),
            model_version=MODEL_VERSION
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Health Insurance Cost Predictor API",
        "version": MODEL_VERSION,
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/docs": "API documentation"
        }
    }
