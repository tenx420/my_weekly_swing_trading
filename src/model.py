# src/model.py

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

def train_model(df, model_path="xgb_model.pkl"):
    """
    Train a 3-class model (Bullish / Bearish / Consolidation).
    """
    # Example feature columns
    feature_cols = ["sma_5", "rsi_14", "return_1d", "volume"]  
    X = df[feature_cols]
    y = df["label"]  # This is 'Bullish'/'Bearish'/'Consolidation'

    # Map labels to numeric
    label_map = {"Bullish": 0, "Bearish": 1, "Consolidation": 2}
    y_encoded = y.map(label_map)

    # Time-based train/test split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, shuffle=False
    )

    # XGBoost multi-class classification
    model = XGBClassifier(eval_metric="mlogloss")  # omit use_label_encoder
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return model

def predict_next_week(model, df):
    """
    Predict next week's label (Bullish / Bearish / Consolidation)
    for the last row in df.
    """
    feature_cols = ["sma_5", "rsi_14", "return_1d", "volume"]
    label_map_inv = {0: "Bullish", 1: "Bearish", 2: "Consolidation"}

    latest_data = df.iloc[[-1]]  # last row as DataFrame
    X_latest = latest_data[feature_cols]

    y_pred = model.predict(X_latest)[0]  # numeric label
    return label_map_inv[y_pred]


def predict_with_confidence(model, X_latest, label_map_inv):
    """
    model: a trained 3-class classifier with .predict_proba
    X_latest: DataFrame or array with a single row of features
    label_map_inv: {0: "Bullish", 1: "Bearish", 2: "Consolidation"}
    
    Returns (predicted_label, confidence_label).
    E.g.: ("Bullish", "High-Confidence Bullish")
    """
    # model.predict_proba -> array of shape (1, 3) for 3 classes
    probs = model.predict_proba(X_latest)[0]  
    predicted_idx = np.argmax(probs)          # index of highest prob
    predicted_label = label_map_inv[predicted_idx]
    
    confidence = probs[predicted_idx]         # probability of predicted class

    # Simple thresholding
    # > 70% => High, > 50% => Moderate, else Low
    if confidence > 0.70:
        confidence_label = f"High-Confidence {predicted_label}"
    elif confidence > 0.50:
        confidence_label = f"Moderate-Confidence {predicted_label}"
    else:
        confidence_label = f"Low-Confidence {predicted_label}"

    return predicted_label, confidence_label
