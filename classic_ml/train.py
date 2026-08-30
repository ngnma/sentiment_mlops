"""Train and save the classic sentiment-analysis pipeline.

Run from any directory with:
    /usr/bin/python3 classic_ml/train.py
"""

# libraries
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# MLFlow
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000") # Set the MLflow tracking server URI (sending all longs to this server)
mlflow.set_experiment("sentiment-classic") # Set the experiment name for organizing runs in MLflow


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "inputs"
MODEL_PATH = PROJECT_DIR / "model" / "classic_pipeline.joblib"

# Edit this configuration to change data columns, the hyperparameter search,
# or the cross-validation settings.
CONFIG = {
    "text_column": "text",
    "target_column": "sentiment",
    "vectorizer": {
    },
    "classifier": {
        "max_iter": 1_000,
    },
    # "param_grid": {
    #     # Core feature extraction
    #     "vectorizer__ngram_range": [(1, 1), (1, 2)],
    #     "vectorizer__min_df": [2, 5, 10],
    #     "vectorizer__max_df": [0.9, 1.0],
    #     "vectorizer__max_features": [3_000, 10_000],
    #     "vectorizer__binary": [False, True],
    #     # Preprocessing and tokenization
    #     "vectorizer__lowercase": [True, False],
    #     "vectorizer__stop_words": [None, "english"],
    #     "vectorizer__token_pattern": [r"(?u)\b\w\w+\b", r"(?u)\b\w+\b"],
    #     "vectorizer__strip_accents": [None, "unicode"],
    #     # Classifier
    #     "classifier__C": [0.001, 0.01, 0.1, 1],
    #     "classifier__penalty": ["l2"],
    #     "classifier__solver": ["liblinear"],
    #     "classifier__class_weight": ["balanced"],
    # },
    "param_grid": {
        # Core feature extraction
        "vectorizer__ngram_range": [(1, 4)],
        "vectorizer__min_df": [2],
        "vectorizer__max_df": [0.9],
        "vectorizer__max_features": [10_000],
        "vectorizer__binary": [False],
        # Preprocessing and tokenization
        "vectorizer__lowercase": [True],
        "vectorizer__stop_words": [None],
        "vectorizer__token_pattern": ['(?u)\\b\\w\\w+\\b'],
        "vectorizer__strip_accents": [None],
        # Classifier
        "classifier__C": [0.1],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["liblinear"],
        "classifier__class_weight": ["balanced"],
    },
    "grid_search": {
        "cv": 5,
        "scoring": "f1_macro",
        "n_jobs": -1,
    }
}


def train() -> Pipeline:
    """Fit the configured pipeline and save it as one joblib artifact."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")

    text_column = CONFIG["text_column"]
    target_column = CONFIG["target_column"]
    for dataframe in (train_df, valid_df):
        dataframe[text_column] = dataframe[text_column].fillna("").astype(str)

    with mlflow.start_run(): # wraps the whole training block so everything inside gets grouped into one logged run
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer(**CONFIG["vectorizer"])),
                ("classifier", LogisticRegression(**CONFIG["classifier"])),
            ]
        )
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=CONFIG["param_grid"],
            **CONFIG["grid_search"],
        )
        grid_search.fit(train_df[text_column], train_df[target_column])
        best_pipeline = grid_search.best_estimator_

        train_predictions = best_pipeline.predict(train_df[text_column])
        valid_predictions = best_pipeline.predict(valid_df[text_column])

        train_f1 = f1_score(train_df[target_column], train_predictions, average="macro")
        valid_f1 = f1_score(valid_df[target_column], valid_predictions, average="macro")
        valid_accuracy = accuracy_score(valid_df[target_column], valid_predictions)
        valid_precision = precision_score(valid_df[target_column], valid_predictions, average="macro")
        valid_recall = recall_score(valid_df[target_column], valid_predictions, average="macro")

        print(f"Training set has {len(train_df)} rows")
        print(f"Validation set has {len(valid_df)} rows")
        print("Best parameters:", grid_search.best_params_)
        print("Train F1-macro:", round(train_f1, 4))
        print("Validation F1-macro:", round(valid_f1, 4))

        # log the winning hyperparameters found by grid search
        mlflow.log_params(grid_search.best_params_) 

        # log validation metrics (these are what you'll compare across runs)
        mlflow.log_metric("train_f1_macro", train_f1)
        mlflow.log_metric("valid_f1_macro", valid_f1)
        mlflow.log_metric("valid_accuracy", valid_accuracy)
        mlflow.log_metric("valid_precision_macro", valid_precision)
        mlflow.log_metric("valid_recall_macro", valid_recall)

        # log the fitted pipeline as an MLflow-tracked artifact
        mlflow.sklearn.log_model(best_pipeline, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(best_pipeline, MODEL_PATH)
        print(f"Saved fitted pipeline to {MODEL_PATH}")

    return best_pipeline


if __name__ == "__main__":
    train()
