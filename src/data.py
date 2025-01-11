import os
import pandas as pd
from polygon import RESTClient
from datetime import datetime
from src.config import TICKERS, POLYGON_API_KEY


# Example start date
START_DATE = "2024-01-01"
# Use current date as end date
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME = "day"

def fetch_historical_data(ticker: str) -> pd.DataFrame:
    """
    Fetch daily bar data for a given ticker from Polygon.
    Returns a DataFrame with columns: [timestamp, open, high, low, close, volume].
    """
    # Pass the hardcoded key directly to the RESTClient constructor
    client = RESTClient(api_key=POLYGON_API_KEY)
    
    all_bars = []
    bars = client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan=TIMEFRAME,  # e.g., "day"
        from_=START_DATE,
        to=END_DATE,
        limit=50000
    )

    for bar in bars:
        all_bars.append({
            "timestamp": pd.to_datetime(bar.timestamp, unit='ms'),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    df = pd.DataFrame(all_bars)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def download_all_data(output_dir: str):
    """
    Download historical data for all tickers and save as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in TICKERS:
        df = fetch_historical_data(ticker)
        csv_path = os.path.join(output_dir, f"{ticker}_{START_DATE}_to_{END_DATE}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {ticker} data to {csv_path}")