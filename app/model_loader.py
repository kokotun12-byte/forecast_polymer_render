import os
import sys
import joblib
import numpy as np

from tensorflow.keras.models import load_model

# NumPy compatibility shim for joblib/pickle artifacts
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core

if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts(lookback: int, n_features: int) -> dict:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    hybrid_artifacts = joblib.load(os.path.join(MODEL_DIR, "hybrid_artifacts.pkl"))
    history_y = joblib.load(os.path.join(MODEL_DIR, "history_y.pkl"))
    history_X = joblib.load(os.path.join(MODEL_DIR, "history_X.pkl"))
    feature_history = joblib.load(os.path.join(MODEL_DIR, "feature_history.pkl"))

    lstm_model = load_model(
        os.path.join(MODEL_DIR, "lstm_model.h5"),
        compile=False
    )

    return {
        "scaler": scaler,
        "hybrid_artifacts": hybrid_artifacts,
        "history_y": history_y,
        "history_X": history_X,
        "feature_history": feature_history,
        "lstm_model": lstm_model,
        "lookback": lookback,
        "n_features": n_features,
    }