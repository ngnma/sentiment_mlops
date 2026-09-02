from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from mangum import Mangum
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "classic_pipeline.joblib"
app = FastAPI()
pipeline = load(MODEL_PATH)  # loaded once, when the container starts


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    proba = pipeline.predict_proba([request.text])[0]
    label_index = proba.argmax()
    label = pipeline.classes_[label_index]
    score = float(proba[label_index])
    
    print(f"PREDICTION_LOG input={request.text!r} label={label} score={score:.4f}")

    return {"label": label, "score": score}


handler = Mangum(app)  # this is what Lambda actually calls
# Why mangum: this is the important detail. Lambda doesn't run a normal web server listening on a port — 
# it just executes a function every time it receives an "event." 
# Mangum is a small adapter that translates a Lambda event into a regular HTTP request FastAPI understands, 
# and translates FastAPI's response back into what Lambda expects. Without it, FastAPI can't run inside Lambda at all.