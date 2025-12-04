"""
Integration tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Test that health returns status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_returns_model_loaded(self, client):
        """Test that health returns model_loaded field."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self, client):
        """Test that root returns API information."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""
    
    @pytest.fixture
    def valid_payload(self):
        """Create valid prediction payload."""
        return {
            "age": 35,
            "gender": "male",
            "bmi": 28.5,
            "bloodpressure": 120,
            "diabetic": "No",
            "children": 2,
            "smoker": "No"
        }
    
    def test_predict_returns_200(self, client, valid_payload):
        """Test that valid prediction returns 200."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
    
    def test_predict_returns_predicted_cost(self, client, valid_payload):
        """Test that prediction includes cost."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "predicted_cost" in data
        assert isinstance(data["predicted_cost"], (int, float))
        assert data["predicted_cost"] >= 0
    
    def test_predict_returns_confidence_interval(self, client, valid_payload):
        """Test that prediction includes confidence interval."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "confidence_interval" in data
        assert "lower" in data["confidence_interval"]
        assert "upper" in data["confidence_interval"]
    
    def test_predict_returns_risk_factors(self, client, valid_payload):
        """Test that prediction includes risk factors."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "risk_factors" in data
        assert isinstance(data["risk_factors"], list)
    
    def test_predict_returns_recommendation(self, client, valid_payload):
        """Test that prediction includes recommendation."""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "recommendation" in data
        assert isinstance(data["recommendation"], str)
    
    def test_predict_invalid_age_low(self, client, valid_payload):
        """Test that age below minimum returns 422."""
        valid_payload["age"] = 10
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_age_high(self, client, valid_payload):
        """Test that age above maximum returns 422."""
        valid_payload["age"] = 150
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_gender(self, client, valid_payload):
        """Test that invalid gender returns 422."""
        valid_payload["gender"] = "other"
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_bmi_low(self, client, valid_payload):
        """Test that BMI below minimum returns 422."""
        valid_payload["bmi"] = 5.0
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_smoker(self, client, valid_payload):
        """Test that invalid smoker value returns 422."""
        valid_payload["smoker"] = "maybe"
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_missing_field(self, client):
        """Test that missing required field returns 422."""
        incomplete_payload = {
            "age": 35,
            "gender": "male"
            # Missing other required fields
        }
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422
    
    def test_predict_smoker_increases_cost(self, client, valid_payload):
        """Test that smoking increases predicted cost."""
        # Non-smoker prediction
        valid_payload["smoker"] = "No"
        response1 = client.post("/predict", json=valid_payload)
        cost_nonsmoker = response1.json()["predicted_cost"]
        
        # Smoker prediction
        valid_payload["smoker"] = "Yes"
        response2 = client.post("/predict", json=valid_payload)
        cost_smoker = response2.json()["predicted_cost"]
        
        assert cost_smoker > cost_nonsmoker
    
    def test_predict_case_insensitive_gender(self, client, valid_payload):
        """Test that gender is case insensitive."""
        valid_payload["gender"] = "MALE"
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_returns_200(self, client):
        """Test that batch prediction returns 200."""
        payload = {
            "instances": [
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
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
    
    def test_batch_predict_returns_correct_count(self, client):
        """Test that batch returns correct number of predictions."""
        payload = {
            "instances": [
                {"age": 30, "gender": "male", "bmi": 25.0, "bloodpressure": 80, "diabetic": "No", "children": 0, "smoker": "No"},
                {"age": 40, "gender": "female", "bmi": 28.0, "bloodpressure": 90, "diabetic": "No", "children": 2, "smoker": "No"},
                {"age": 50, "gender": "male", "bmi": 30.0, "bloodpressure": 100, "diabetic": "Yes", "children": 1, "smoker": "Yes"}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        
        assert data["total_instances"] == 3
        assert len(data["predictions"]) == 3
    
    def test_batch_predict_includes_processing_time(self, client):
        """Test that batch includes processing time."""
        payload = {
            "instances": [
                {"age": 30, "gender": "male", "bmi": 25.0, "bloodpressure": 80, "diabetic": "No", "children": 0, "smoker": "No"}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info_returns_200(self, client):
        """Test that model info returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200
    
    def test_model_info_returns_features(self, client):
        """Test that model info includes features list."""
        response = client.get("/model/info")
        data = response.json()
        
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0

