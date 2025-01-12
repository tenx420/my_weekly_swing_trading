#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Local imports (adjust to your file structure)
from src.config import Config
from src.data import fetch_daily_data, daily_to_weekly
from src.features import add_technical_indicators, label_data_future_based
from src.model import train_xgb_model, load_xgb_model, prepare_xy

def walk_forward_folds(full_df_daily, train_years=2, test_weeks=13):
    """
    Generator that yields (df_daily_train, df_daily_test) folds.
    """
    df_sorted = full_df_daily.sort_values("timestamp").reset_index(drop=True)
    min_date = df_sorted["timestamp"].iloc[0]
    max_date = df_sorted["timestamp"].iloc[-1]

    current_train_start = min_date
    current_train_end = min_date + pd.DateOffset(years=train_years)

    while True:
        test_end = current_train_end + pd.DateOffset(weeks=test_weeks)
        if test_end > max_date:
            break

        df_train = df_sorted[
            (df_sorted["timestamp"] >= current_train_start) & 
            (df_sorted["timestamp"] < current_train_end)
        ]
        df_test = df_sorted[
            (df_sorted["timestamp"] >= current_train_end) & 
            (df_sorted["timestamp"] < test_end)
        ]

        if df_train.empty or df_test.empty:
            break

        yield (df_train.copy(), df_test.copy())
        current_train_end = test_end

def prepare_weekly_labeled(df_daily):
    """
    1) daily -> weekly
    2) compute indicators
    3) label with future-based approach
    """
    df_weekly = daily_to_weekly(df_daily)
    if df_weekly.empty:
        return None
    
    df_feat = add_technical_indicators(df_weekly)
    df_labeled = label_data_future_based(df_feat.copy())
    
    if df_labeled.empty:
        return None
    
    return df_labeled

def backtest_symbols(symbols, start_date, end_date, train_years=2, test_weeks=13):
    """
    Returns a dictionary:
        results[symbol] = {
            'accuracy': float,
            'labels': [...],
            'preds':  [...],
        }
    """
    from sklearn.metrics import accuracy_score
    symbol_results = {}

    for symbol in symbols:
        print(f"\n=== Backtesting {symbol} ===")
        df_daily = fetch_daily_data(symbol, start_date, end_date)

        if df_daily.empty:
            print(f"No daily data for {symbol} in this date range.")
            symbol_results[symbol] = None
            continue

        folds = walk_forward_folds(df_daily, train_years, test_weeks)
        
        all_test_labels = []
        all_test_preds = []
        fold_index = 0

        for (df_train_daily, df_test_daily) in folds:
            fold_index += 1
            print(f"\n--- Fold {fold_index} ({symbol}) ---")
            print(f"Train: {df_train_daily['timestamp'].iloc[0].date()} -> "
                  f"{df_train_daily['timestamp'].iloc[-1].date()}, rows={len(df_train_daily)}")
            print(f"Test : {df_test_daily['timestamp'].iloc[0].date()} -> "
                  f"{df_test_daily['timestamp'].iloc[-1].date()}, rows={len(df_test_daily)}")

            df_train_labeled = prepare_weekly_labeled(df_train_daily)
            df_test_labeled  = prepare_weekly_labeled(df_test_daily)

            if (df_train_labeled is None) or (df_test_labeled is None):
                print("Empty weekly data in fold. Skipping.")
                continue

            model = train_xgb_model(df_train_labeled, verbose=False)
            X_test, y_test = prepare_xy(df_test_labeled)
            y_pred = model.predict(X_test)

            all_test_labels.extend(y_test.tolist())
            all_test_preds.extend(y_pred.tolist())

        # Calculate accuracy for this symbol
        if len(all_test_labels) > 0:
            accuracy = accuracy_score(all_test_labels, all_test_preds)
        else:
            accuracy = float('nan')

        print(f"\n>>> Final Accuracy for {symbol}: {accuracy:.4f}")
        symbol_results[symbol] = {
            'accuracy': accuracy,
            'labels': all_test_labels,
            'preds': all_test_preds,
        }

    # Print overall average accuracy, if needed
    valid_accuracies = [
        info['accuracy'] for info in symbol_results.values()
        if info is not None and not np.isnan(info['accuracy'])
    ]
    if len(valid_accuracies) > 0:
        mean_accuracy = np.mean(valid_accuracies)
        print(f"\nOverall average accuracy across {len(valid_accuracies)} symbols: {mean_accuracy:.4f}")

    return symbol_results

def main():
    TICKERS = [
        "GOOGL", "HD", "GLD", "META", "SLV", "AAPL", "AMZN",
        "BTC", "DIA", "IWM", "MSFT", "NFLX", "NVDA", "QQQ",
        "SPY", "TSLA", "DELL", "TSM", "RDDT", "DJT", "GME"
    ]
    
    start_date = "2015-01-01"
    end_date   = "2025-01-10"
    train_years = 2
    test_weeks  = 13

    # 1) Run the backtest
    results = backtest_symbols(TICKERS, start_date, end_date, train_years, test_weeks)

    # 2) Create a DataFrame with one row per symbol and its final accuracy
    accuracy_rows = []
    for symbol, data in results.items():
        if data is not None:
            accuracy = data["accuracy"]
            accuracy_rows.append({"Symbol": symbol, "Final Accuracy": accuracy})

    df_accuracies = pd.DataFrame(accuracy_rows)

    # 3) Save to CSV
    df_accuracies.to_csv("all_symbols_accuracies.csv", index=False)
    print("\nSaved final accuracies for all symbols to all_symbols_accuracies.csv")

if __name__ == "__main__":
    main()


