import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def feature_engineering(df):
    # Extract name features
    df['name_len'] = df['name'].apply(lambda x: len(str(x)))
    df['name_start'] = df['name'].apply(lambda x: str(x)[0].lower())
    df['name_end'] = df['name'].apply(lambda x: str(x)[-1].lower())
    df['name_last2'] = df['name'].apply(lambda x: str(x)[-2:].lower())
    
    return df

def train_gender_model():
    print("Loading data...")
    df = pd.read_csv('dataset/users.csv')
    
    print("Feature Engineering...")
    df = feature_engineering(df)
    
    # Features
    # We will use age, company, and name derivatives
    X = df[['age', 'company', 'name_len', 'name_start', 'name_end', 'name_last2']]
    y = df['gender']
    
    # Preprocessing
    categorical_cols = ['company', 'name_start', 'name_end', 'name_last2']
    numerical_cols = ['age', 'name_len']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Model Pipeline
    # Changed to RandomForestClassifier as per notebook improvements
    from sklearn.ensemble import RandomForestClassifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # Params for tuning
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10]
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Starting Training (RandomizedSearch)...")
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Best Params: {search.best_params_}")
    print(f"Enhanced Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('gender_classification/models', exist_ok=True)
    joblib.dump(best_model, 'gender_classification/models/gender_model_enhanced.pkl')
    print("Saved enhanced model to gender_classification/models/gender_model_enhanced.pkl")

if __name__ == '__main__':
    train_gender_model()
