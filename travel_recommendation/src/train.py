import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def build_collaborative_model():
    print("Loading data...")
    df = pd.read_csv('dataset/hotels.csv')
    
    # User-Item Matrix
    # We count visits as implicit feedback strength, or just binary
    # Let's count visits
    user_hotel_matrix = df.groupby(['userCode', 'name']).size().unstack(fill_value=0)
    
    # Normalize by user (optional, but good practice)
    # For now, just raw counts
    
    # Matrix Factorization using SVD
    # Identify latent features
    n_components = min(20, user_hotel_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_reduced = svd.fit_transform(user_hotel_matrix)
    
    print(f"Explained Variance Ratio: {svd.explained_variance_ratio_.sum()}")
    
    # Reconstruct Matrix for Evaluation
    matrix_reconstructed = np.dot(matrix_reduced, svd.components_)
    
    # Calculate RMSE on non-zero elements (approximation of accuracy on observed data)
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    mask = user_hotel_matrix.values > 0
    rmse = sqrt(mean_squared_error(user_hotel_matrix.values[mask], matrix_reconstructed[mask]))
    print(f'Reconstruction RMSE on Observed Interactions: {rmse}')
    
    # We will save the SVD model and the original matrix columns/index to reconstruct
    
    model_data = {
        'svd_model': svd,
        'user_index': user_hotel_matrix.index,
        'hotel_columns': user_hotel_matrix.columns,
        'matrix_reduced': matrix_reduced,
        'rmse': rmse
    }
    
    os.makedirs('travel_recommendation/models', exist_ok=True)
    joblib.dump(model_data, 'travel_recommendation/models/collab_model.pkl')
    
    # Save simple hotel info for display
    hotel_info = df[['name', 'place', 'price']].drop_duplicates('name').set_index('name')
    joblib.dump(hotel_info, 'travel_recommendation/models/hotel_metadata.pkl')
    
    print("Collaborative Filtering model built and saved.")

if __name__ == '__main__':
    build_collaborative_model()
