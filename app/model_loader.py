import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


def build_lstm_model(lookback, n_features):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, n_features), return_sequences=False),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    return model


def load_artifacts(lookback: int, n_features: int):
    base_path = "models"

    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    hybrid_artifacts = joblib.load(os.path.join(base_path, "hybrid_artifacts.pkl"))
    history_y = joblib.load(os.path.join(base_path, "history_y.pkl"))
    history_X = joblib.load(os.path.join(base_path, "history_X.pkl"))
    feature_history = joblib.load(os.path.join(base_path, "feature_history.pkl"))

    lstm_model = build_lstm_model(lookback, n_features)
    lstm_model.load_weights(os.path.join(base_path, "lstm_model.weights.h5"))

    return {
        "scaler": scaler,
        "hybrid_artifacts": hybrid_artifacts,
        "history_y": history_y,
        "history_X": history_X,
        "feature_history": feature_history,
        "lstm_model": lstm_model,
    }