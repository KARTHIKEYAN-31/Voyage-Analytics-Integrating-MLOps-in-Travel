FROM python:3.11-slim

WORKDIR /app

COPY gender_classification/requirements.txt ./requirements_local.txt
# Using global requirements or specific? Let's create specific.
RUN pip install pandas scikit-learn flask gunicorn joblib

COPY gender_classification/ ./gender_classification/
COPY dataset/ ./dataset/

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "gender_classification.src.app:app"]
