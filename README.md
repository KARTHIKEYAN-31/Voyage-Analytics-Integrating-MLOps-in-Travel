# Voyager MLOps Project

Voyager is a comprehensive MLOps project focusing on predictive analytics for travel and user data. It covers regression, classification, and recommendation system scenarios, integrated with modern MLOps tools.


![Voyager](picture/flight_price_prediction.png)


## Project Structure

```text
Voyage/
├── dataset/                    # Shared dataset directory
├── flight_price/               # Scenario 1: Regression (Flight Price Prediction)
├── gender_classification/      # Scenario 2: Classification (User Gender)
└── travel_recommendation/      # Scenario 3: recommendation & Dashboard
```

## Scenarios Overview

### 1. Flight Price Prediction (`flight_price/`)
- **Type**: Regression
- **Objective**: Predict the price of a flight based on distance, time, and agency.
- **MLOps Integrations**: MLFlow, Docker, Kubernetes, Apache Airflow, Jenkins CI/CD.

### 2. Gender Classification (`gender_classification/`)
- **Type**: Classification
- **Objective**: Categorize a user's gender based on age and company.
- **MLOps Integrations**: Flask REST API, Docker.

### 3. Travel Recommendation (`travel_recommendation/`)
- **Type**: recommendation System
- **Objective**: Provide hotel suggestions and travel insights.
- **Features**: Interactive Streamlit Dashboard, Content-based recommendation Engine.

## Global Setup

1. **Clone the repository.**
2. **Install requirements**: Each scenario folder contains its own code. Ensure you have `pandas`, `scikit-learn`, `flask`, `streamlit`, and `joblib` installed.
3. **Explore Scenarios**: Navigate to each sub-folder and read the local `README.md` for specific execution instructions.
