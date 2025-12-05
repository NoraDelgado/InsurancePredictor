# 🏥 Health Insurance Cost Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red?style=for-the-badge)
![TailwindCSS](https://img.shields.io/badge/Tailwind-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)

### Predict health insurance costs using Machine Learning

[🌐 **Live Demo**](https://noradelgado.github.io/InsurancePredictor/) • [📡 **API Docs**](https://insurance-predictor-api.onrender.com/docs) • [👩‍💻 **About Me**](https://noradelgado.github.io/)

</div>

---

## 🎯 Project Overview

A full-stack machine learning application that predicts individual health insurance costs based on demographic and health factors. This project demonstrates end-to-end ML development from data analysis to production deployment.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Machine Learning** | XGBoost model with 80% R² accuracy |
| ⚡ **Fast API** | RESTful API with FastAPI (sub-200ms responses) |
| 🎨 **Modern UI** | Responsive Next.js frontend with glass-morphism design |
| 🚀 **Production Deployed** | Live on GitHub Pages + Render.com |
| 📊 **Data-Driven** | Trained on 1,340 real insurance records |

---

## 🌐 Live Demo

<div align="center">

### **Try it now: [https://noradelgado.github.io/InsurancePredictor/](https://noradelgado.github.io/InsurancePredictor/)**

</div>

| Endpoint | URL |
|----------|-----|
| 🖥️ **Web App** | https://noradelgado.github.io/InsurancePredictor/ |
| 📡 **API** | https://insurance-predictor-api.onrender.com |
| 📖 **API Docs** | https://insurance-predictor-api.onrender.com/docs |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.11** - Core programming language
- **FastAPI** - Modern, fast web framework for APIs
- **XGBoost** - Gradient boosting ML algorithm
- **scikit-learn** - Data preprocessing & model evaluation
- **Pandas/NumPy** - Data manipulation
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **React Hook Form** - Form handling with validation

### Deployment
- **GitHub Pages** - Frontend hosting
- **Render.com** - Backend API hosting
- **GitHub Actions** - CI/CD pipeline

---

## 📊 Model Performance

The XGBoost model was trained on insurance data with the following results:

| Metric | Score |
|--------|-------|
| **R² Score** | 0.80 |
| **MAE** | $4,108 |
| **RMSE** | $5,727 |

### Feature Importance

```
Smoker            ████████████████████████████████  87.1%
Blood Pressure    ███                                4.1%
BMI               ██                                 2.5%
Region            █                                  1.6%
Children          █                                  1.3%
Gender            █                                  1.2%
Age               █                                  1.1%
Diabetic          █                                  1.0%
```

**Key Insight:** Smoking status is the dominant predictor, accounting for 87% of the prediction power. Smokers pay significantly higher insurance premiums.

---

## 🗂️ Project Structure

```
InsurancePredictor/
├── 📁 api/                    # FastAPI Backend
│   ├── main.py                # API endpoints & logic
│   └── schemas/               # Pydantic models
│
├── 📁 frontend/               # Next.js Frontend
│   └── src/
│       ├── app/               # Next.js pages
│       ├── components/        # React components
│       └── lib/               # API client & types
│
├── 📁 models/                 # Trained ML Models
│   ├── trained/               # Production model (.pkl)
│   └── artifacts/             # Scaler, encoders
│
├── 📁 data/raw/               # Training dataset
├── 📁 src/                    # ML source code
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm

### Installation

```bash
# Clone the repository
git clone https://github.com/NoraDelgado/InsurancePredictor.git
cd InsurancePredictor

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train the model
python train_model.py

# Start the API (Terminal 1)
uvicorn api.main:app --reload --port 8000

# Start the frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to use the app locally.

---

## 📡 API Reference

### Endpoints

#### Health Check
```http
GET /health
```
```json
{"status": "healthy", "model_loaded": true, "model_version": "1.0.0"}
```

#### Predict Insurance Cost
```http
POST /predict
Content-Type: application/json
```

**Request:**
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

| Field | Type | Valid Values |
|-------|------|--------------|
| age | number | 18-100 |
| gender | string | "male", "female" |
| bmi | number | 10.0-60.0 |
| bloodpressure | number | 60-200 |
| diabetic | string | "Yes", "No" |
| children | number | 0-10 |
| smoker | string | "Yes", "No" |
| region | string | "northeast", "northwest", "southeast", "southwest" |

---

## 📈 What I Learned

This project helped me develop skills in:

- **Machine Learning Pipeline**: Data cleaning, feature engineering, model training, and evaluation
- **API Development**: Building RESTful APIs with FastAPI and proper error handling
- **Frontend Development**: Creating responsive UIs with React/Next.js and TypeScript
- **DevOps**: Setting up CI/CD pipelines, containerization concepts, and cloud deployment
- **Full-Stack Integration**: Connecting frontend to backend with proper CORS handling

---

## 👩‍💻 About the Author

**Nora Delgado**

I'm a recent graduate with a Bachelor's degree in Information Science with a concentration in Data Science and Analytics. I'm passionate about solving puzzles, learning about science, and helping others grow.

- 🌐 **Portfolio**: [noradelgado.github.io](https://noradelgado.github.io/)
- 💼 **LinkedIn**: [linkedin.com/in/nora-delgado](https://www.linkedin.com/in/noradelgadobusot/) 
- 📧 **Email**: noradelgadobusot@gmail.com
- 📍 **Location**: Fort Myers, Florida

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

Made with ❤️ by [Nora Delgado](https://noradelgado.github.io/)

</div>
