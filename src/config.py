# src/config.py
import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

# Polygon API key
POLYGON_API_KEY = os.getenv("fDLhW6xHMAEyG9daMWLUQouqrpLV4TVd")

# OpenAI API key
OPENAI_API_KEY = os.getenv("")

# Example ticker list
TICKERS = ["AAPL", "TSLA", "SPY"]

# Example start/end dates for historical data
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Other config constants
TIMEFRAME = "day"  # For daily bars from Polygon
