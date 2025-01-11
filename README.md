# MY_WEEKLY_SWING_TRADING

This project demonstrates a weekly swing trading pipeline using Polygon data, RSI/SMA features, and an XGBoost multi-class model to classify next week's close as Bullish, Bearish, or Consolidation. It also outputs a confidence level (High, Medium, or Low) based on predicted probabilities.

## Prerequisites

- Python 3.8+
- Polygon.io API key
- Dependencies in `requirements.txt`

## Setup

1. Clone this repo
2. Create a virtual environment (optional)
3. `pip install -r requirements.txt`
4. Export your `POLYGON_API_KEY` or edit `src/config.py` directly.

## Usage

To run the entire pipeline for a given symbol and date range:

```bash
python run_pipeline.py AAPL 2022-01-01 2024-01-01 True
