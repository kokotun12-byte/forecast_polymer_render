import os
import sys
import joblib
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# ---- NumPy compatibility shim for older/newer pickle paths ----
# Some saved joblib/pickle artifacts reference numpy._core.*.
# In environments where that path does not exist, map it to numpy.core.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core

if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def build_lstm_model(lookback, n_features):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, n_features), return_sequences=False),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    
    # 🔥 IMPORTANT: build model before loading weights
    model.build(input_shape=(None, lookback, n_features))
    
    return model


def load_artifacts(lookback: int, n_features: int):
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    hybrid_artifacts = joblib.load(os.path.join(MODEL_DIR, "hybrid_artifacts.pkl"))
    history_y = joblib.load(os.path.join(MODEL_DIR, "history_y.pkl"))
    history_X = joblib.load(os.path.join(MODEL_DIR, "history_X.pkl"))
    feature_history = joblib.load(os.path.join(MODEL_DIR, "feature_history.pkl"))

    from tensorflow.keras.models import load_model

    lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    
    return {
        "scaler": scaler,
        "hybrid_artifacts": hybrid_artifacts,
        "history_y": history_y,
        "history_X": history_X,
        "feature_history": feature_history,
        "lstm_model": lstm_model,
    }