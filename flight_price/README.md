# Flight Price Prediction (Regression)

This scenario focuses on predicting flight prices using a Random Forest regression model.

## Features
- **Experiment Tracking**: Integrated with **MLFlow** to log parameters and performance metrics (RMSE, R2).
- **APIs**: Served via **Flask**.
- **Containerization**: Packaged with **Docker**.
- **Orchestration**: **Kubernetes** manifests for deployment and **Apache Airflow** DAG for automated training.
- **CI/CD**: **Jenkinsfile** defining the build and deploy pipeline.

## Getting Started

1. **Train Model**: `python src/train.py`
2. **Run API**: `python src/app.py` (Port 5000)
3. **Build Docker**: `docker build -t flight-price-api .`
