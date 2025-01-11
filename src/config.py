import os
from datetime import datetime
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

# Polygon API key
POLYGON_API_KEY = "POLYGON_API_KEY"

# OpenAI API key
OPENAI_API_KEY = ""

# Example ticker list
TICKERS = ["AAPL", "TSLA", "SPY", "HD", "MSFT", "AMZN", "GOOGL", "NFLX", "META", "NVDA", "QQQ", "IWM", "DIA", "GLD", "SLV", "BTC"]
# Example start/end dates for historical data
START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Other config constants
TIMEFRAME = "day"  # For daily bars from Polygon