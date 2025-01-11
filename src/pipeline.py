# src/pipeline.py

import pandas as pd
from datetime import datetime
from src.config import Config
from src.data import fetch_daily_data, save_raw_data, daily_to_weekly
from src.features import add_technical_indicators, label_data_future_based, save_processed_data
from src.model import train_xgb_model, load_xgb_model, predict_next_week

def run_pipeline_for_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    train_new_model: bool = True
):
    """
    1) Fetch daily data for 'symbol' from start_date->end_date
    2) Resample daily->weekly
    3) Compute weekly indicators
    4) Label each bar with future-based approach
    5) Train or load model
    6) Predict next week
    Return a dict with final prediction details (no console printing).
    """

    # 1) Fetch daily data
    df_daily = fetch_daily_data(symbol, start_date, end_date)
    if df_daily.empty:
        return {
            "symbol": symbol,
            "message": "No daily data returned",
            "prediction": None
        }

    # Optional: save daily raw
    daily_filename = f"{symbol}_daily_raw.csv"
    save_raw_data(df_daily, daily_filename)

    # 2) daily->weekly
    df_weekly = daily_to_weekly(df_daily)
    if df_weekly.empty:
        return {
            "symbol": symbol,
            "message": "No weekly data after resampling",
            "prediction": None
        }
    weekly_filename = f"{symbol}_weekly_raw.csv"
    save_raw_data(df_weekly, weekly_filename)

    # 3) Add indicators
    df_feat = add_technical_indicators(df_weekly)

    # 4) Label (future-based)
    df_labeled = label_data_future_based(df_feat.copy())
    if df_labeled.empty:
        return {
            "symbol": symbol,
            "message": "No labeled rows",
            "prediction": None
        }
    save_processed_data(df_labeled, symbol)

    # 5) Train or load
    if train_new_model:
        model = train_xgb_model(df_labeled, verbose=False)  # silent
    else:
        model = load_xgb_model()

    # 6) Predict
    results = predict_next_week(df_labeled, model=model)
    # e.g. { 'prediction': 'Bearish', 'confidence_level': 'High', 'probabilities': [0.8, 0.2] }

    # last labeled bar date
    last_label_date = df_labeled["timestamp"].iloc[-1]
    next_week_start = last_label_date + pd.Timedelta(days=7)
    next_week_end   = next_week_start + pd.Timedelta(days=4)

    # Return a dictionary summarizing the final
    return {
        "symbol": symbol,
        "last_labeled_date": str(last_label_date.date()),
        "predicting_week_start": str(next_week_start.date()),
        "predicting_week_end": str(next_week_end.date()),
        "prediction": results["prediction"],
        "confidence_level": results["confidence_level"],
        "probabilities": results["probabilities"]
    }
