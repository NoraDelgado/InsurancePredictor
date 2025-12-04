"""
Pydantic Schemas for Insurance Cost Predictor API.

This module defines request and response models with validation
for the prediction API endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class Gender(str, Enum):
    """Gender enum for validation."""
    MALE = "male"
    FEMALE = "female"


class YesNo(str, Enum):
    """Yes/No enum for binary fields."""
    YES = "Yes"
    NO = "No"


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class InsuranceInput(BaseModel):
    """
    Input schema for insurance prediction.
    
    Validates all input fields with appropriate constraints.
    """
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age of the individual (18-100)",
        examples=[35]
    )
    gender: str = Field(
        ...,
        description="Gender (male/female)",
        examples=["male"]
    )
    bmi: float = Field(
        ...,
        ge=10.0,
        le=60.0,
        description="Body Mass Index (10.0-60.0)",
        examples=[28.5]
    )
    bloodpressure: int = Field(
        ...,
        ge=60,
        le=200,
        description="Blood pressure reading (60-200)",
        examples=[120]
    )
    diabetic: str = Field(
        ...,
        description="Diabetic status (Yes/No)",
        examples=["No"]
    )
    children: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of children (0-10)",
        examples=[2]
    )
    smoker: str = Field(
        ...,
        description="Smoking status (Yes/No)",
        examples=["No"]
    )
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate and normalize gender."""
        v_lower = v.lower().strip()
        if v_lower not in ['male', 'female']:
            raise ValueError('Gender must be "male" or "female"')
        return v_lower
    
    @field_validator('diabetic', 'smoker')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate and normalize Yes/No fields."""
        v_title = v.strip().title()
        if v_title not in ['Yes', 'No']:
            raise ValueError('Value must be "Yes" or "No"')
        return v_title
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "gender": "male",
                    "bmi": 28.5,
                    "bloodpressure": 120,
                    "diabetic": "No",
                    "children": 2,
                    "smoker": "No"
                },
                {
                    "age": 45,
                    "gender": "female",
                    "bmi": 32.0,
                    "bloodpressure": 95,
                    "diabetic": "Yes",
                    "children": 1,
                    "smoker": "Yes"
                }
            ]
        }
    }


class RiskFactor(BaseModel):
    """Individual risk factor with impact assessment."""
    factor: str = Field(..., description="Name of the risk factor")
    impact: RiskLevel = Field(..., description="Impact level (Low/Medium/High)")
    contribution: str = Field(..., description="Estimated contribution to cost")


class ConfidenceInterval(BaseModel):
    """Confidence interval for prediction."""
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")


class PredictionResponse(BaseModel):
    """
    Response schema for predictions.
    
    Includes prediction, confidence interval, risk factors, and recommendations.
    """
    predicted_cost: float = Field(
        ...,
        description="Predicted annual insurance cost in USD"
    )
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description="Confidence interval for the prediction"
    )
    risk_factors: List[RiskFactor] = Field(
        default=[],
        description="List of identified risk factors"
    )
    recommendation: str = Field(
        ...,
        description="Health recommendation based on risk profile"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_cost": 8542.50,
                "confidence_interval": {
                    "lower": 7261.13,
                    "upper": 9823.88
                },
                "risk_factors": [
                    {
                        "factor": "Age",
                        "impact": "Medium",
                        "contribution": "+10-20%"
                    }
                ],
                "recommendation": "Your health profile indicates moderate risk. Consider regular health checkups."
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")
    model_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional model information"
    )


class BatchPredictionInput(BaseModel):
    """Input for batch predictions."""
    instances: List[InsuranceInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of instances to predict (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_instances: int = Field(..., description="Total number of instances processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    suggestion: Optional[str] = Field(None, description="Suggested fix")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Validation Error",
                "detail": "Age must be between 18 and 100",
                "suggestion": "Please enter a valid age value"
            }
        }
    }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    r2_score: float = Field(..., description="R² score")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")


class ModelInfoResponse(BaseModel):
    """Detailed model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Model version")
    training_date: Optional[str] = Field(None, description="When the model was trained")
    metrics: Optional[ModelMetrics] = Field(None, description="Model performance metrics")
    features: List[str] = Field(..., description="Features used by the model")

