import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost
import joblib
import os

# Set MLFlow experiment
# Robust path resolution for tracking URI
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir) # Voyage/flight_price
tracking_uri = "file:///" + os.path.join(project_dir, "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("flight_price")

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Drop ID columns
    df = df.drop(columns=['travelCode', 'userCode'])
    
    # Convert date to datetime and extract features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df = df.drop(columns=['date'])
    
    return df

def main():
    # Robust dataset path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir) # Voyage/flight_price
    
    # Assuming dataset is at Voyage/dataset/flights.csv
    # project_dir is Voyage/flight_price, parent is Voyage
    voyage_dir = os.path.dirname(project_dir)
    data_path = os.path.join(voyage_dir, 'dataset', 'flights.csv')
    
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Identify categorical and numerical columns
    categorical_cols = ['from', 'to', 'flightType', 'agency']
    numerical_cols = ['time', 'distance', 'month', 'day', 'weekday']
    
    # Create preprocessing pipeline
    # Enhancements: Added StandardScaler for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Define model with XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    
    # Expanded Hyperparameter Grid
    param_grid = {
        'regressor__n_estimators': [100, 200, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'regressor__min_child_weight': [1, 3, 5]
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting RandomizedSearchCV...")
    # Randomized Search with more iterations
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=20, # Increased iterations
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    with mlflow.start_run():
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        print(f"Best Params: {random_search.best_params_}")
        
        # Predict w/ best model
        y_pred = best_model.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
        
        # Log params and metrics
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        models_dir = os.path.join(project_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'flight_price_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
