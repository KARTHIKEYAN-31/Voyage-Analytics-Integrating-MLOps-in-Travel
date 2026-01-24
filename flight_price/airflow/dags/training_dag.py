from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import subprocess
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_model():
    # Trigger the training script
    script_path = os.path.join(os.environ['AIRFLOW_HOME'], 'dags/scripts/train.py')
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
    print(result.stdout)

with DAG(
    'flight_price_training_pipeline',
    default_args=default_args,
    description='Pipeline for training flight price prediction model',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # 1. Wait for data file
    wait_for_data = FileSensor(
        task_id='wait_for_flights_data',
        filepath='/opt/airflow/data/flights.csv',
        poke_interval=30,
        timeout=600
    )

    # 2. Train Model
    train_task = PythonOperator(
        task_id='train_regression_model',
        python_callable=train_model
    )

    # Define task dependencies
    wait_for_data >> train_task
