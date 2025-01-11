# src/pipeline.py

import pandas as pd
from src.config import Config
from src.data import fetch_daily_data, save_raw_data, daily_to_weekly
from src.features import add_technical_indicators, label_data_future_based, save_processed_data
from src.model import train_xgb_model, load_xgb_model, predict_next_week

def run_pipeline_for_symbol(symbol: str, start_date: str, end_date: str, train_new_model: bool = True):
    """
    Runs the entire pipeline (daily->weekly->indicators->label->train->predict) for a single symbol.
    Returns a dict with final prediction info or None if no data.
    """

    # 1) Fetch daily data
    df_daily = fetch_daily_data(symbol, start_date, end_date)
    if df_daily.empty:
        return None

    # 2) Save daily raw
    daily_filename = f"{symbol}_daily_raw.csv"
    save_raw_data(df_daily, daily_filename)

    # 3) Resample daily->weekly
    df_weekly = daily_to_weekly(df_daily)
    if df_weekly.empty:
        return None
    weekly_filename = f"{symbol}_weekly_raw.csv"
    save_raw_data(df_weekly, weekly_filename)

    # 4) Add weekly indicators
    df_feat = add_technical_indicators(df_weekly)

    # 5) Label with future-based approach
    df_labeled = label_data_future_based(df_feat.copy())
    if df_labeled.empty:
        return None
    save_processed_data(df_labeled, symbol)

    # 6) Train or load model
    if train_new_model:
        model = train_xgb_model(df_labeled, verbose=False)
    else:
        model = load_xgb_model()

    # 7) Predict
    results = predict_next_week(df_labeled, model=model)
    # e.g. { 'prediction': 'Bearish', 'confidence_level': 'High', 'probabilities': [0.8, 0.2] }

    # Let's also store last_labeled_date
    last_label_date = df_labeled["timestamp"].iloc[-1]

    # Return a dictionary with final details
    return {
        "symbol": symbol,
        "last_labeled_date": str(last_label_date.date()),
        "prediction": results["prediction"],
        "confidence_level": results["confidence_level"],
        "probabilities": results["probabilities"]
    }
