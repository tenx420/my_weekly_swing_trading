# src/features.py

import pandas as pd
import numpy as np
import pandas_ta as ta
from src.config import Config

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    On the weekly DataFrame, compute RSI(14), SMA(5), SMA(8).
    """
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_8"] = df["close"].rolling(8).mean()

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def label_data_future_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Future-based labeling:
    row i is Bullish if close[i+1] > close[i], else Bearish.
    """
    df["next_close"] = df["close"].shift(-1)

    labels = []
    for i in range(len(df)):
        if i == len(df) - 1:
            # last row has no "next week" => label is NaN
            labels.append(np.nan)
            continue
        curr_close = df.loc[i, "close"]
        nxt_close = df.loc[i, "next_close"]
        if nxt_close > curr_close:
            labels.append("Bullish")
        else:
            labels.append("Bearish")

    df["label"] = labels
    df.dropna(subset=["label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Optionally remove next_close if you don't need it in final data
    df.drop(columns=["next_close"], inplace=True)
    return df

def save_processed_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Save final weekly (with features + labels) to CSV in data/processed/.
    """
    import os
    os.makedirs(Config.DATA_PROCESSED_PATH, exist_ok=True)
    filename = f"{Config.DATA_PROCESSED_PATH}/{symbol}_weekly_processed.csv"
    df.to_csv(filename, index=False)
    print(f"Saved processed data to {filename}")

