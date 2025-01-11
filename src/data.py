# src/data.py

import os
import pandas as pd
from polygon import RESTClient
from src.config import Config

def fetch_daily_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLC data from Polygon. Returns [timestamp, open, high, low, close, volume].
    """
    client = RESTClient(Config.POLYGON_API_KEY)

    aggs = client.get_aggs(
        symbol,
        1,
        "day",
        start_date,
        end_date,
        limit=5000
    )

    records = []
    for bar in aggs:
        records.append({
            "timestamp": pd.to_datetime(bar.timestamp, unit='ms'),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save raw DataFrame to CSV in data/raw/.
    """
    os.makedirs(Config.DATA_RAW_PATH, exist_ok=True)
    path = os.path.join(Config.DATA_RAW_PATH, filename)
    df.to_csv(path, index=False)
    print(f"Saved raw data to {path}")

def daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily -> weekly bars (end Fri).
    Columns: [timestamp, open, high, low, close, volume].
    """
    if df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    df.set_index("timestamp", inplace=True)

    # Weekly aggregator on Friday
    df_weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    df_weekly.dropna(subset=["open", "high", "low", "close"], how="any", inplace=True)
    df_weekly.reset_index(inplace=True)
    return df_weekly



