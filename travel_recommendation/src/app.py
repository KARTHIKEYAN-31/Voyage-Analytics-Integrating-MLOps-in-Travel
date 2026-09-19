import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Voyager Travel Insights", layout="wide")

st.title("Voyager Travel Insights Dashboard (Enhanced)")

# Robust path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_dir, 'models', 'collab_model.pkl')
META_PATH = os.path.join(project_dir, 'models', 'hotel_metadata.pkl')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.getcwd(), 'travel_recommendation', 'models', 'collab_model.pkl')
    META_PATH = os.path.join(os.getcwd(), 'travel_recommendation', 'models', 'hotel_metadata.pkl')

@st.cache_data
def load_data():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return None, None
    model_data = joblib.load(MODEL_PATH)
    hotel_meta = joblib.load(META_PATH)
    return model_data, hotel_meta

model_data, hotel_meta = load_data()

if model_data is None:
    st.error("Model assets not found. Please train the recommendation engine first using 'python travel_recommendation/src/train.py'.")
else:
    # Sidebar
    st.sidebar.header("User Selection")
    
    # Reconstruct user list from the model data index
    user_index = model_data['user_index']
    user_codes = sorted(user_index.tolist())
    
    selected_user = st.sidebar.selectbox("Select User Code", user_codes)
    
    # Get User Data
    user_idx = user_index.get_loc(selected_user)
    
    # SVD components
    svd = model_data['svd_model']
    matrix_reduced = model_data['matrix_reduced']
    hotel_columns = model_data['hotel_columns']
    
    # Get user vector in latent space
    user_vector = matrix_reduced[user_idx]
    
    # Predict scores: User Vector * Matrix_V (components)
    # The SVD attribute components_ is V^T (n_components, n_features)
    predicted_scores = np.dot(user_vector, svd.components_)
    
    # Create a series for easier handling
    scores_series = pd.Series(predicted_scores, index=hotel_columns)
    scores_series = scores_series.sort_values(ascending=False)
    
    # Historical Data (Simulate from matrix, though matrix is 0/1 usually)
    # We can't easily reverse the matrix to 'list of names' without the original df 
    # but strictly from matrix, non-zero entries are visited.
    # To keep it simple, let's assume we want to recommend things not in the top visited (if we had history).
    # Since we don't have the explicit history list here easily without loading the huge CSV again,
    # let's just show Top Recommendations.
    
    st.header(f"Personalized Recommendations for User {selected_user}")
    
    st.write("These recommendations are generated using **Collaborative Filtering (SVD)** based on your travel patterns.")
    
    top_recommendations = scores_series.head(10).index.tolist()
    
    # Filter metadata for these hotels
    rec_df = hotel_meta.loc[top_recommendations]
    st.table(rec_df)
            
    # Insights Section
    st.divider()
    st.header("Global Travel Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution by Place")
        price_chart = hotel_meta.groupby('place')['price'].mean().sort_values()
        st.bar_chart(price_chart)
        
    with col2:
        st.subheader("Popular Destinations")
        count_chart = hotel_meta['place'].value_counts()
        st.bar_chart(count_chart)
