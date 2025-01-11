import os
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess

from src.data import download_all_data
from src.features import build_weekly_features
from src.model import train_model, predict_with_confidence
from src.config import TICKERS

# For convenience, define your label_map_inv (or store in config)
LABEL_MAP_INV = {0: "Bullish", 1: "Bearish", 2: "Consolidation"}

def plot_predictions(df, ticker, predictions, output_dir="charts"):
    """
    Plot the current week and next week predictions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["close"], label="Close Price")
    
    # Highlight the current week
    current_week = df.iloc[-5:]
    plt.plot(current_week["timestamp"], current_week["close"], label="Current Week", color="orange")
    
    # Add prediction for next week
    next_week_date = current_week["timestamp"].iloc[-1] + pd.Timedelta(days=7)
    plt.axvline(next_week_date, color="red", linestyle="--", label="Next Week Prediction")
    plt.text(next_week_date, df["close"].iloc[-1], predictions, color="red")
    
    plt.title(f"{ticker} - Current Week and Next Week Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f"{ticker}_prediction.png"))
    plt.close()

def run_full_pipeline():
    """
    - Download data for each ticker
    - Build weekly features (with 3-class label)
    - Train a multi-class model
    - Predict next week's label for each ticker WITH confidence
    - Save predictions to CSV
    - Plot current week and next week predictions
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
            ticker = file.split("_")[0]
            weekly_df["ticker"] = ticker
            
            all_weekly_df.append(weekly_df)
    
    combined_df = pd.concat(all_weekly_df, ignore_index=True)
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

        # Plot current week and next week predictions
        plot_predictions(df_ticker, ticker, conf_label)

    # 5) Save predictions to a CSV
    predictions_df = pd.DataFrame(predictions_list)
    predictions_path = "data/processed/next_week_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions (with confidence) saved to {predictions_path}")
