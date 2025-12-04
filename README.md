# 🏥 Health Insurance Cost Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=next.js&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predict health insurance costs with state-of-the-art machine learning**

[🌐 Live Demo](https://noradelgado.github.io/InsurancePredictor/) • [📖 Documentation](#documentation) • [🚀 Quick Start](#quick-start) • [📡 API Reference](#api-reference)

</div>

---

## 📋 Overview

This project predicts individual medical charges (claims) billed by health insurance based on demographic and health factors. It combines advanced machine learning techniques with a modern, responsive web application to deliver accurate predictions.

### ✨ Key Features

- 🎯 **High Accuracy** - R² ~0.80 using optimized XGBoost model
- ⚡ **Fast API** - Sub-200ms response times with FastAPI backend
- 🎨 **Modern UI** - Beautiful, responsive Next.js frontend with glass-morphism design
- 🌐 **Live Demo** - Deployed on GitHub Pages + Render.com
- 📊 **Feature Importance** - Smoking status is the #1 predictor (87% importance)

---

## 🌐 Live Demo

**Frontend**: [https://noradelgado.github.io/InsurancePredictor/](https://noradelgado.github.io/InsurancePredictor/)

**API Docs**: [https://insurance-predictor-api.onrender.com/docs](https://insurance-predictor-api.onrender.com/docs)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### Installation

```powershell
# Clone the repository
git clone https://github.com/NoraDelgado/InsurancePredictor.git
cd InsurancePredictor

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running Locally

**Terminal 1: Start API server**
```powershell
.\venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```

**Terminal 2: Start frontend**
```powershell
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.80 |
| **MAE** | $4,108 |
| **RMSE** | $5,727 |
| **Training Samples** | 1,072 |
| **Test Samples** | 268 |

### Feature Importance

```
┌─────────────────────────────────────────────────────────────┐
│ Smoker             ████████████████████████████████  87.1%  │
│ Blood Pressure     ███                               4.1%   │
│ BMI                ██                                2.5%   │
│ Region             █                                 1.6%   │
│ Children           █                                 1.3%   │
│ Gender             █                                 1.2%   │
│ Age                █                                 1.1%   │
│ Diabetic           █                                 1.0%   │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Smoking status is by far the most important factor, accounting for 87% of the prediction power!

---

## 🗂️ Project Structure

```
InsurancePredictor/
├── 📁 api/                    # FastAPI backend
│   ├── main.py                # API entry point
│   └── 📁 schemas/            # Pydantic models
├── 📁 frontend/               # Next.js frontend
│   ├── 📁 src/
│   │   ├── 📁 components/     # React components
│   │   ├── 📁 lib/            # API client & types
│   │   └── 📁 app/            # Next.js pages
│   └── package.json
├── 📁 models/                 # Saved models
│   ├── 📁 trained/            # Production models
│   └── 📁 artifacts/          # Preprocessing artifacts
├── 📁 data/raw/               # Dataset
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── render.yaml                # Render.com deployment config
└── README.md                  # This file
```

---

## 📡 API Reference

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://insurance-predictor-api.onrender.com`

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

#### Predict Insurance Cost

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 35,
  "gender": "male",
  "bmi": 28.5,
  "bloodpressure": 120,
  "diabetic": "No",
  "children": 2,
  "smoker": "Yes",
  "region": "southeast"
}
```

**Response:**
```json
{
  "predicted_charge": 30062.51,
  "model_version": "1.0.0"
}
```

### Input Validation

| Field | Type | Constraints |
|-------|------|-------------|
| `age` | number | 18-100 |
| `gender` | string | "male" \| "female" |
| `bmi` | number | 10.0-60.0 |
| `bloodpressure` | integer | 60-200 |
| `diabetic` | string | "Yes" \| "No" |
| `children` | integer | 0-10 |
| `smoker` | string | "Yes" \| "No" |
| `region` | string | "northeast" \| "northwest" \| "southeast" \| "southwest" |

---

## 🚀 Deployment

### Frontend (GitHub Pages)

The frontend is automatically deployed to GitHub Pages when you push to the `main` branch.

**URL**: `https://noradelgado.github.io/InsurancePredictor/`

### Backend (Render.com)

1. Go to [Render.com](https://render.com) and sign up
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` and deploy

**URL**: `https://insurance-predictor-api.onrender.com`

---

## 📄 License

This project is licensed under the MIT License.

---

## 📞 Contact

- **Author**: Nora Delgado
- **Email**: noradelgadobusot@gmail.com
- **LinkedIn**: [Nora Delgado](https://www.linkedin.com/in/noradelgado)
- **Portfolio**: [noradelgado.github.io](https://noradelgado.github.io)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by Nora Delgado

</div>
