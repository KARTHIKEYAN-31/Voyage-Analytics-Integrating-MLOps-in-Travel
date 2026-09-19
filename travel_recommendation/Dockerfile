FROM python:3.11-slim

WORKDIR /app

# Install dependencies for Streamlit and data processing
RUN pip install streamlit pandas scikit-learn joblib matplotlib

# Copy source code and recommendation assets
COPY travel_recommendation/ ./travel_recommendation/
COPY dataset/ ./dataset/

# Expose Streamlit default port
EXPOSE 8501

# Streamlit healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "travel_recommendation/src/app.py", "--server.address", "0.0.0.0"]
