from joblib import load
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "classic_pipeline.joblib"

def test_model_loads():
    pipeline = load(MODEL_PATH)
    assert pipeline is not None

def test_prediction_shape():
    pipeline = load(MODEL_PATH)
    proba = pipeline.predict_proba(["I love this"])
    assert proba.shape[1] == 2  # two classes: positive/negative