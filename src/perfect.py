from prefect import flow, task
import subprocess
import sys

@task
def run_training():
    subprocess.run([sys.executable, "src/train_mlflow.py"], check=True)

@flow(name="Flipkart Sentiment MLflow Training Pipeline")
def sentiment_pipeline():
    run_training()

if __name__ == "__main__":
    sentiment_pipeline()
