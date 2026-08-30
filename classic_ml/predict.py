"""Prediction interface for the classic sentiment model."""

from pathlib import Path
from typing import Dict, Union

from joblib import load


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "classic_pipeline.joblib"


def predict(text: str) -> Dict[str, Union[str, float]]:
    """Return the sentiment label and model confidence for one piece of text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not hasattr(predict, "_pipeline"):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model pipeline not found at {MODEL_PATH}. Run classic_ml/train.py first."
            )
        predict._pipeline = load(MODEL_PATH)

    pipeline = predict._pipeline
    label = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]

    return {"label": str(label), "score": float(probabilities.max())}

if __name__ == "__main__":
    # Example usage
    example_text = "It works as designs but is a total ripoff for a cheap charcoal filter."
    result = predict(example_text)
    print(f"Text: {example_text}")
    print(f"Predicted sentiment: {result['label']}, Confidence score: {result['score']:.4f}")