# Voyage Analytics: Deployment Guide & Operational Runbook

This guide provides step-by-step instructions for setting up, training, containerizing, orchestrating, and deploying the **Voyage Analytics** platform.

---

## 1. Prerequisites & Environment Setup

### System Requirements:
- **Python:** 3.10 or 3.11
- **Docker Desktop:** Installed and running
- **Kubernetes CLI (`kubectl`):** Enabled in Docker Desktop, Minikube, or Kind
- **Git:** Version 2.30+

### Local Environment Initialization:
```powershell
# Clone the repository
git clone <your-repository-url>
cd "Voyage Analytics Integrating MLOps in Travel"

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install root dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Model Training & Asset Generation

Execute the training pipelines for all three machine learning systems:

### A. Flight Price Regression (with MLflow Tracking)
```powershell
# Trains XGBoost regression model with hyperparameter search and logs to MLflow
python flight_price/src/train.py
```
- **Output Artifact:** `flight_price/models/flight_price_model.pkl`
- **MLflow Tracking Directory:** `flight_price/mlruns/`

### B. Gender Classification Model
```powershell
# Trains enhanced Random Forest demographic classifier
python gender_classification/src/train.py
```
- **Output Artifact:** `gender_classification/models/gender_model_enhanced.pkl`

### C. Travel Recommendation Collaborative Model
```powershell
# Builds SVD Matrix Factorization collaborative filtering model
python travel_recommendation/src/train.py
```
- **Output Artifacts:**
  - `travel_recommendation/models/collab_model.pkl`
  - `travel_recommendation/models/hotel_metadata.pkl`

---

## 3. Running Services Locally

### A. Flight Price Flask REST API (Port 5000)
```powershell
python flight_price/src/app.py
```
- **Access UI:** `http://localhost:5000`
- **Health Check:** `http://localhost:5000/health`
- **Sample Prediction API Call:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "agency": "FlyingDrops",
    "date": "2024-10-15",
    "time": 2.5,
    "distance": 680.0
}'
```

### B. Gender Classification REST API (Port 5001)
```powershell
python gender_classification/src/app.py
```
- **Access UI:** `http://localhost:5001`
- **Health Check:** `http://localhost:5001/health`
- **Sample Classification API Call:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5001/classify" -Method Post -ContentType "application/json" -Body '{
    "name": "Roy Braun",
    "age": 25,
    "company": "4You"
}'
```

### C. Travel Recommendation Streamlit Dashboard (Port 8501)
```powershell
streamlit run travel_recommendation/src/app.py
```
- **Access Dashboard:** `http://localhost:8501`

### D. MLflow Experiment Tracking UI
```powershell
mlflow ui --backend-store-uri "file:///$(Get-Location)/flight_price/mlruns" --port 5050
```
- **Access MLflow Dashboard:** `http://localhost:5050`

---

## 4. Docker Containerization

### A. Build and Run Flight Price API Container:
```powershell
# Build Docker image
docker build -t flight-price-api:latest -f flight_price/Dockerfile .

# Run Docker container
docker run -d -p 5000:5000 --name flight-price-container flight-price-api:latest

# Verify health
curl http://localhost:5000/health
```

### B. Build and Run Travel Recommendation Dashboard Container:
```powershell
docker build -t travel-recommendation-app:latest -f travel_recommendation/Dockerfile .
docker run -d -p 8501:8501 --name travel-rec-container travel-recommendation-app:latest
```

---

## 5. Kubernetes Orchestration Runbook

Deploy the containerized Flight Price microservice onto a Kubernetes cluster:

```powershell
# 1. Apply Kubernetes manifests (Deployment and LoadBalancer Service)
kubectl apply -f flight_price/k8s/deployment.yaml

# 2. Verify Pod status
kubectl get pods -l app=flight-price-api

# 3. Verify Service and External Port
kubectl get svc flight-price-service

# 4. Check Deployment Rollout Status
kubectl rollout status deployment/flight-price-api

# 5. Port Forwarding (if testing locally on Minikube / Kind)
kubectl port-forward svc/flight-price-service 8080:80
```
- Test via forwarded port: `http://localhost:8080/health`

---

## 6. Apache Airflow DAG Orchestration

To run the automated data-driven training workflow in Apache Airflow:

1. **Configure Airflow Home & Copy DAG:**
   ```powershell
   export AIRFLOW_HOME=~/airflow
   mkdir -p $AIRFLOW_HOME/dags
   cp flight_price/airflow/dags/training_dag.py $AIRFLOW_HOME/dags/
   ```
2. **Ensure Input Data Directory:**
   ```powershell
   mkdir -p /opt/airflow/data
   cp dataset/flights.csv /opt/airflow/data/flights.csv
   ```
3. **Start Airflow Webserver & Scheduler:**
   ```powershell
   airflow standalone
   ```
4. **Trigger the DAG via CLI:**
   ```powershell
   airflow dags trigger flight_price_training_pipeline
   ```

---

## 7. Jenkins CI/CD Pipeline Setup

1. Open Jenkins Web Interface (`http://localhost:8080`).
2. Create a new **Pipeline** job: `Voyage-Flight-Price-CI-CD`.
3. Under **Pipeline Definition**, select **Pipeline script from SCM** (Git).
4. Set **Script Path** to `flight_price/Jenkinsfile`.
5. Click **Build Now** to verify the end-to-end automated testing, image building, and Kubernetes rolling deployment.
