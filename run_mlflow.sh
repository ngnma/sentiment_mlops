#!/bin/bash
# source venv/bin/activate  
mlflow server \
  --backend-store-uri sqlite:///mlflow_data/mlflow.db \
  --default-artifact-root ./mlflow_data/mlruns \
  --host 127.0.0.1 --port 5000