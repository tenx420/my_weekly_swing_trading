# src/model.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from src.config import Config

def prepare_xy(df: pd.DataFrame):
    """
    Return X,y for training. 'label_id' is 0 for Bearish, 1 for Bullish.
    """
    df["label_id"] = df["label"].map(Config.LABEL_MAP)
    X = df[Config.FEATURE_COLS].values
    y = df["label_id"].values
    return X, y

def train_xgb_model(df: pd.DataFrame, verbose=True):
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=Config.XGB_PARAM_GRID,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=(1 if verbose else 0)
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    if verbose:
        print("Best Params:", grid.best_params_)
        y_pred = best_model.predict(X_test)
        print("Classification Report (Test Data):")
        print(classification_report(y_test, y_pred, target_names=Config.LABEL_MAP.keys()))

    joblib.dump(best_model, Config.MODEL_PATH)
    if verbose:
        print(f"Model saved to {Config.MODEL_PATH}")
    return best_model

def load_xgb_model():
    return joblib.load(Config.MODEL_PATH)

def predict_next_week(df: pd.DataFrame, model=None):
    """
    Predict if *next week* is Bullish or Bearish, using the last row of features.
    """
    if model is None:
        model = load_xgb_model()

    X, _ = prepare_xy(df)
    X_last = X[-1, :].reshape(1, -1)

    probs = model.predict_proba(X_last)[0]
    pred_idx = np.argmax(probs)
    pred_label = Config.INV_LABEL_MAP[pred_idx]

    confidence = probs[pred_idx]
    if confidence >= Config.CONF_HIGH:
        conf_level = "High"
    elif confidence >= Config.CONF_MEDIUM:
        conf_level = "Medium"
    else:
        conf_level = "Low"

    return {
        "prediction": pred_label,
        "confidence_level": conf_level,
        "probabilities": probs.tolist()
    }
