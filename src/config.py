# src/config.py

class Config:
    POLYGON_API_KEY = ""

    DATA_RAW_PATH = "data/raw"
    DATA_PROCESSED_PATH = "data/processed"
    MODEL_PATH = "xgb_model.pkl"

    # Tickers we want to run
    TICKERS = ["AAPL", "MSFT", "TSLA"]  # or any list of symbols

    LABEL_MAP = {"Bearish": 0, "Bullish": 1}
    INV_LABEL_MAP = {0: "Bearish", 1: "Bullish"}

    FEATURE_COLS = [
        "open", "high", "low", "close", "volume",
        "rsi_14", "sma_5", "sma_8"
    ]

    XGB_PARAM_GRID = {
        "n_estimators": [100],
        "max_depth": [3],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "random_state": [42]
    }

    CONF_HIGH = 0.7
    CONF_MEDIUM = 0.5
