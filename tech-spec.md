# 🔧 Technical Specifications

## Health Insurance Cost Predictor

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Data Specifications](#data-specifications)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [API Specifications](#api-specifications)
6. [Frontend Architecture](#frontend-architecture)
7. [Security & Compliance](#security--compliance)
8. [Performance Requirements](#performance-requirements)
9. [Infrastructure](#infrastructure)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Web Browser   │  │   Mobile App    │  │   API Client    │         │
│  │   (React SPA)   │  │   (Future)      │  │   (External)    │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
└───────────┼────────────────────┼────────────────────┼───────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              CDN / EDGE                                  │
│           (Static Assets, SSL Termination, DDoS Protection)             │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOAD BALANCER                                  │
│                    (NGINX / AWS ALB / Traefik)                          │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Rate Limiting│  │   Auth/JWT   │  │   Logging    │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Application                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│   │
│  │  │ /predict   │  │ /health    │  │ /explain   │  │ /metrics   ││   │
│  │  │  endpoint  │  │  endpoint  │  │  endpoint  │  │  endpoint  ││   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘│   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                      Service Layer                               │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                     │   │
│  │  │  Prediction Svc  │  │  Explanation Svc │                     │   │
│  │  └──────────────────┘  └──────────────────┘                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ML MODEL LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Model Service                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Preprocessor │  │ XGBoost/LGBM │  │ SHAP Engine  │          │   │
│  │  │   Pipeline   │  │    Model     │  │  (Explain)   │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Model Registry (MLflow)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OBSERVABILITY                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Prometheus  │  │    Grafana   │  │   Sentry     │                  │
│  │   (Metrics)  │  │ (Dashboards) │  │   (Errors)   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌─────────┐
│  User   │────▶│Frontend │────▶│   FastAPI   │────▶│  Model  │
│         │◀────│  React  │◀────│   Backend   │◀────│ Service │
└─────────┘     └─────────┘     └─────────────┘     └─────────┘
     │               │                 │                 │
     │               │                 │                 │
     ▼               ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                       │
│         (Logging, Metrics, Tracing, Error Tracking)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Runtime | Python | 3.11+ | Core runtime |
| API Framework | FastAPI | 0.104+ | REST API |
| ASGI Server | Uvicorn | 0.24+ | Production server |
| Validation | Pydantic | 2.5+ | Data validation |
| ML Framework | scikit-learn | 1.3+ | Base ML operations |
| Gradient Boosting | XGBoost | 2.0+ | Primary model |
| Gradient Boosting | LightGBM | 4.1+ | Alternative model |
| Gradient Boosting | CatBoost | 1.2+ | Categorical handling |
| Interpretability | SHAP | 0.43+ | Model explanations |
| Experiment Tracking | MLflow | 2.8+ | MLOps |
| Data Processing | Pandas | 2.1+ | Data manipulation |
| Numerical Computing | NumPy | 1.26+ | Array operations |
| Hyperparameter Tuning | Optuna | 3.4+ | Automated tuning |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | React | 18+ | UI library |
| Meta Framework | Next.js | 14+ | Full-stack React |
| State Management | React Query | 5+ | Server state |
| Styling | Tailwind CSS | 3.3+ | Utility-first CSS |
| Animation | Framer Motion | 10+ | Animations |
| Forms | React Hook Form | 7+ | Form handling |
| Validation | Zod | 3+ | Schema validation |
| HTTP Client | Axios | 1.6+ | API calls |
| Icons | Lucide React | 0.29+ | Icon library |
| Charts | Recharts | 2.9+ | Data visualization |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Containerization | Docker | Container runtime |
| Orchestration | Docker Compose | Local orchestration |
| CI/CD | GitHub Actions | Automation |
| Cloud (Optional) | AWS/GCP/Azure | Cloud hosting |
| CDN | CloudFlare | Edge caching |
| Monitoring | Prometheus | Metrics collection |
| Visualization | Grafana | Dashboards |
| Error Tracking | Sentry | Error monitoring |

---

## Data Specifications

### Input Schema

```typescript
interface InsuranceInput {
  age: number;           // Range: 18-100, required
  gender: 'male' | 'female';  // required
  bmi: number;           // Range: 10.0-60.0, required
  bloodpressure: number; // Range: 60-200, required
  diabetic: 'Yes' | 'No';     // required
  children: number;      // Range: 0-10, required
  smoker: 'Yes' | 'No';       // required
}
```

### Output Schema

```typescript
interface PredictionResponse {
  predicted_cost: number;         // USD, 2 decimal places
  confidence_interval: {
    lower: number;                // 85% of prediction
    upper: number;                // 115% of prediction
  };
  risk_factors: RiskFactor[];     // Array of identified risks
  recommendation: string;          // Health recommendation
}

interface RiskFactor {
  factor: string;                 // e.g., "Smoking"
  impact: 'High' | 'Medium' | 'Low';
  contribution: string;           // e.g., "+40-60%"
}
```

### Data Validation Rules

| Field | Type | Validation |
|-------|------|------------|
| age | integer | 18 ≤ value ≤ 100 |
| gender | string | enum: ['male', 'female'] |
| bmi | float | 10.0 ≤ value ≤ 60.0 |
| bloodpressure | integer | 60 ≤ value ≤ 200 |
| diabetic | string | enum: ['Yes', 'No'] |
| children | integer | 0 ≤ value ≤ 10 |
| smoker | string | enum: ['Yes', 'No'] |

### Feature Engineering

| Derived Feature | Formula | Purpose |
|-----------------|---------|---------|
| age_squared | age² | Captures non-linear age effects |
| age_group | Categorical bins | Risk grouping |
| bmi_category | WHO classifications | Health risk categories |
| is_obese | bmi ≥ 30 | Binary risk flag |
| is_hypertensive | bp > 90 | Binary risk flag |
| smoker_bmi | smoker × bmi | Interaction term |
| smoker_age | smoker × age | Interaction term |
| risk_score | Weighted composite | Overall risk indicator |

---

## Machine Learning Pipeline

### Training Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────▶│  Validation │────▶│  Cleaning   │
│ (CSV/JSON)  │     │   Schema    │     │  Missing    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model     │◀────│   Feature   │◀────│  Encoding & │
│  Selection  │     │ Engineering │     │   Scaling   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Hyperparams │────▶│   Training  │────▶│ Evaluation  │
│   Tuning    │     │  (CV=5)     │     │  Metrics    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Artifacts  │◀────│   Model     │◀────│    SHAP     │
│   Export    │     │  Registry   │     │  Analysis   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Inference Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  API Input  │────▶│  Validation │────▶│ Preprocessing│
│   (JSON)    │     │  (Pydantic) │     │  Transform   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Response   │◀────│    SHAP     │◀────│   Model     │
│  Assembly   │     │ (Optional)  │     │  Predict    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Model Configuration

```yaml
# models/config.yaml
model:
  name: xgboost
  version: 2.0
  
hyperparameters:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 1.0
  min_child_weight: 3

preprocessing:
  scaler: robust
  imputer: knn
  imputer_neighbors: 5
  
features:
  numeric:
    - age
    - bmi
    - bloodpressure
    - children
  categorical:
    - gender
    - diabetic
    - smoker
  engineered:
    - age_squared
    - bmi_squared
    - smoker_bmi
    - smoker_age
    - risk_score

training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42
```

---

## API Specifications

### Endpoints

#### POST /predict

**Request:**
```http
POST /predict HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{
  "age": 35,
  "gender": "male",
  "bmi": 28.5,
  "bloodpressure": 120,
  "diabetic": "No",
  "children": 2,
  "smoker": "No"
}
```

**Response (200 OK):**
```json
{
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
  "recommendation": "Maintain healthy lifestyle habits."
}
```

**Error Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

#### GET /health

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2024-12-04T10:30:00Z"
}
```

### Rate Limiting

| Tier | Requests/min | Requests/day |
|------|-------------|--------------|
| Free | 10 | 100 |
| Basic | 60 | 1,000 |
| Pro | 300 | 10,000 |
| Enterprise | Custom | Custom |

### Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Frontend Architecture

### Component Hierarchy

```
App
├── Layout
│   ├── Header
│   │   ├── Logo
│   │   ├── Navigation
│   │   └── ThemeToggle
│   └── Footer
├── Pages
│   ├── HomePage
│   │   ├── Hero
│   │   ├── PredictionForm
│   │   │   ├── FormField (×7)
│   │   │   └── SubmitButton
│   │   └── ResultCard
│   │       ├── CostDisplay
│   │       ├── ConfidenceInterval
│   │       ├── RiskFactors
│   │       └── Recommendation
│   ├── AboutPage
│   └── DocsPage
└── Shared
    ├── LoadingSpinner
    ├── ErrorBoundary
    └── Toast
```

### State Management

```typescript
// React Query for server state
const usePrediction = () => {
  return useMutation({
    mutationFn: async (data: InsuranceInput) => {
      const response = await api.post('/predict', data);
      return response.data;
    },
    onSuccess: (data) => {
      // Handle success
    },
    onError: (error) => {
      // Handle error
    }
  });
};

// Zustand for client state (optional)
interface AppState {
  theme: 'light' | 'dark';
  isFormSubmitting: boolean;
  lastPrediction: PredictionResponse | null;
}
```

---

## Security & Compliance

### Security Measures

| Layer | Measure | Implementation |
|-------|---------|----------------|
| Transport | TLS 1.3 | SSL certificates |
| Input | Validation | Pydantic schemas |
| Authentication | JWT | Bearer tokens |
| Authorization | RBAC | Role-based access |
| Rate Limiting | Token bucket | FastAPI middleware |
| Logging | Audit trail | Structured logging |
| Secrets | Encryption | Environment variables |

### Data Privacy

- No PII stored in logs
- Input data not persisted
- GDPR/CCPA compliant
- Data anonymization for analytics

---

## Performance Requirements

### Response Time SLAs

| Percentile | Target |
|------------|--------|
| p50 | < 100ms |
| p90 | < 200ms |
| p99 | < 500ms |

### Throughput

| Metric | Target |
|--------|--------|
| Concurrent users | 1,000 |
| Requests/second | 500 |
| Availability | 99.9% |

### Resource Limits

```yaml
# Docker resource constraints
api:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 512M
```

---

## Infrastructure

### Docker Configuration

```dockerfile
# Production Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment (Optional)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: insurance-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: insurance-predictor
  template:
    metadata:
      labels:
        app: insurance-predictor
    spec:
      containers:
      - name: api
        image: insurance-predictor:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

---

## Appendix

### Environment Variables

```env
# Application
APP_NAME=insurance-predictor
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Models
MODEL_PATH=/app/models/trained/best_model.pkl
PREPROCESSOR_PATH=/app/models/artifacts/preprocessor.pkl

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# Monitoring
SENTRY_DSN=https://...
PROMETHEUS_PORT=9090

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Dependency Versions

See `requirements.txt` and `package.json` for exact versions.

---

*Last Updated: December 2024*
*Version: 1.0.0*

