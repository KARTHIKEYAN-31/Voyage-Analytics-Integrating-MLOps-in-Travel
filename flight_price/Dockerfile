# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY dataset/ ./dataset/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/app.py

# Run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]
