# src/features.py
import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to the DataFrame (like moving averages, RSI, etc.).
    Assumes df has columns: [timestamp, open, high, low, close, volume].
    """
    df = df.copy()
    df['return_1d'] = df['close'].pct_change()
    
    # Example: 5-day simple moving average
    df['sma_5'] = df['close'].rolling(window=5).mean()
    
    # Example: 14-day RSI (very rough example)
    # For a better RSI, you might want a more precise calculation
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily data to weekly data (open of Monday, close of Friday, etc.).
    We'll label the weekly bar by the Monday open and Friday close.
    """
    df = df.set_index('timestamp')
    # Resample to weekly using 'W-FRI' frequency
    # open -> first, high -> max, low -> min, close -> last, volume -> sum
    weekly = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'return_1d': 'sum',  # sum of daily returns as an approximation
        'sma_5': 'last',
        'rsi_14': 'last'
    })
    weekly.reset_index(inplace=True)
    weekly.dropna(inplace=True)
    return weekly

import numpy as np

def create_labels(df, up_threshold=0.01, down_threshold=-0.01):
    """
    Create 3-class labels based on next week's percentage change.
    up_threshold: +1%
    down_threshold: -1%
    
    Classes:
      - Bullish:       next week's close >= current close * (1 + up_threshold)
      - Bearish:       next week's close <= current close * (1 + down_threshold)
      - Consolidation: otherwise
    """
    df = df.copy()

    # Shift close by -1 to get next week's close
    df["next_week_close"] = df["close"].shift(-1)

    # Calculate weekly return
    df["weekly_return"] = (df["next_week_close"] - df["close"]) / df["close"]

    # Label logic
    conditions = [
        (df["weekly_return"] >= up_threshold),
        (df["weekly_return"] <= down_threshold)
    ]
    choices = ["Bullish", "Bearish"]
    df["label"] = np.select(conditions, choices, default="Consolidation")

    # Drop the last row if it has no next_week_close
    df.dropna(subset=["next_week_close"], inplace=True)
    return df


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature engineering: apply technicals, resample to weekly, create labels.
    """
    daily_with_inds = compute_technical_indicators(df)
    weekly_df = resample_to_weekly(daily_with_inds)
    labeled_df = create_labels(weekly_df)
    return labeled_df
