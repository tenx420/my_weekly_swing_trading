# src/pipeline.py

import os
import pickle
import pandas as pd
from datetime import datetime

from src.data import download_all_data
from src.features import build_weekly_features
from src.model import train_model, predict_with_confidence

# For convenience, define your label_map_inv (or store in config)
LABEL_MAP_INV = {0: "Bullish", 1: "Bearish", 2: "Consolidation"}

def run_full_pipeline():
    """
    - Download data for each ticker
    - Build weekly features (with 3-class label)
    - Train a multi-class model
    - Predict next week's label for each ticker WITH confidence
    - Save predictions to CSV
    """
    # 1) Download data
    download_all_data(output_dir="data/raw")

    # 2) Combine all tickers into a single DataFrame with weekly features
    all_weekly_df = []
    for file in os.listdir("data/raw"):
        if file.endswith(".csv"):
            path = os.path.join("data/raw", file)
            df_daily = pd.read_csv(path)
            df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
            
            # Build weekly features -> includes create_labels
            weekly_df = build_weekly_features(df_daily)
            
            # Parse ticker from filename, e.g. "AAPL_2023-01-01_to_2023-12-31.csv" => "AAPL"
            ticker_symbol = file.split("_")[0]
            weekly_df["ticker"] = ticker_symbol
            
            all_weekly_df.append(weekly_df)

    combined_df = pd.concat(all_weekly_df, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    combined_df.to_csv("data/processed/weekly_features.csv", index=False)
    print("Weekly features saved to data/processed/weekly_features.csv")

    # 3) Train the model on combined data
    model = train_model(combined_df, model_path="xgb_model.pkl")

    # 4) Load the saved model & generate predictions for each ticker
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    # The columns we used for training
    feature_cols = ["sma_5", "rsi_14", "return_1d", "volume"]
    
    predictions_list = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Predict next week label & confidence for each ticker
    for ticker in combined_df["ticker"].unique():
        df_ticker = combined_df[combined_df["ticker"] == ticker]
        # Get the last row's features
        latest_X = df_ticker[feature_cols].iloc[[-1]]  # DataFrame with 1 row

        pred_label, conf_label = predict_with_confidence(model, latest_X, LABEL_MAP_INV)

        predictions_list.append({
            "timestamp": now_str,
            "ticker": ticker,
            "predicted_label": pred_label,
            "confidence": conf_label  # e.g. "High-Confidence Bullish"
        })

    # 5) Save predictions to a CSV
    predictions_df = pd.DataFrame(predictions_list)
    predictions_path = "data/processed/next_week_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions (with confidence) saved to {predictions_path}")

