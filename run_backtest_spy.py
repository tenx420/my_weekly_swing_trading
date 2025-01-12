#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.config import Config
from src.data import fetch_daily_data, daily_to_weekly
from src.features import add_technical_indicators, label_data_future_based
from src.model import train_xgb_model, load_xgb_model

"""
Example walk-forward backtest for SPY:

1) Define folds by date ranges
2) For each fold:
   - Train on historical segment
   - Predict next segment
   - Compare predictions to actual next-week movement
3) Aggregate results (accuracy, etc.)

Usage:
    python run_backtest_spy.py
"""

def walk_forward_folds(full_df_daily, train_years=2, test_weeks=13):
    """
    Generator that yields (df_daily_train, df_daily_test) folds.

    - train_years: how many years to include in each training set. 
      Then we slide forward for each fold.
    - test_weeks: how many weeks to test in each fold.

    For example, if your daily data starts 2015-01-01, you might:
      - fold 1 trains on ~2015–2016 daily, tests on next 13 weeks
      - fold 2 trains on ~2015–2017 daily, tests on next 13 weeks
      etc.
    """
    # Sort by date
    df_sorted = full_df_daily.sort_values("timestamp")
    df_sorted.reset_index(drop=True, inplace=True)

    min_date = df_sorted["timestamp"].iloc[0]
    max_date = df_sorted["timestamp"].iloc[-1]

    # We'll do an expanding window approach:
    # 1) Start with train_years of data, test on next test_weeks,
    # 2) move forward test_weeks, expand train set, etc.
    current_train_start = min_date
    current_train_end = min_date + pd.DateOffset(years=train_years)

    while True:
        # test from current_train_end to test_end
        test_end = current_train_end + pd.DateOffset(weeks=test_weeks)

        # If test_end > max_date, break
        if test_end > max_date:
            break

        # Slice daily data
        df_train = df_sorted[(df_sorted["timestamp"] >= current_train_start) & 
                             (df_sorted["timestamp"] < current_train_end)]
        df_test  = df_sorted[(df_sorted["timestamp"] >= current_train_end) & 
                             (df_sorted["timestamp"] < test_end)]
        
        if df_train.empty or df_test.empty:
            break
        
        yield (df_train.copy(), df_test.copy())

        # move forward: expand train up to test_end
        current_train_end = test_end
        # next fold continues

def prepare_weekly_labeled(df_daily):
    """
    1) daily -> weekly
    2) compute indicators
    3) label with future-based approach
    Returns labeled weekly DataFrame.
    """
    from src.data import daily_to_weekly
    from src.features import add_technical_indicators, label_data_future_based

    df_weekly = daily_to_weekly(df_daily)
    if df_weekly.empty:
        return None
    df_feat = add_technical_indicators(df_weekly)
    df_labeled = label_data_future_based(df_feat.copy())
    if df_labeled.empty:
        return None
    return df_labeled

def backtest_spy():
    """
    Main walk-forward backtest function for SPY.
    We'll measure accuracy across folds as an example.
    """
    # 1) Fetch daily data for SPY
    symbol = "SPY"
    start_date = "2015-01-01"
    end_date   = "2025-01-10"
    df_daily = fetch_daily_data(symbol, start_date, end_date)
    if df_daily.empty:
        print("No daily data for SPY in this range.")
        return
    
    # 2) Walk-forward folds
    # For demonstration: 2-year train window, 13 weeks test
    # Adjust as needed
    folds = walk_forward_folds(df_daily, train_years=2, test_weeks=13)

    all_test_labels = []
    all_test_preds  = []

    fold_index = 0

    for (df_train_daily, df_test_daily) in folds:
        fold_index += 1
        print(f"\n=== Fold {fold_index} ===")
        print(f"Train: {df_train_daily['timestamp'].iloc[0].date()} -> {df_train_daily['timestamp'].iloc[-1].date()}, rows={len(df_train_daily)}")
        print(f"Test : {df_test_daily['timestamp'].iloc[0].date()} -> {df_test_daily['timestamp'].iloc[-1].date()}, rows={len(df_test_daily)}")

        # Prepare weekly labeled data for train
        df_train_labeled = prepare_weekly_labeled(df_train_daily)
        # Prepare weekly labeled data for test
        df_test_labeled  = prepare_weekly_labeled(df_test_daily)
        
        if (df_train_labeled is None) or (df_test_labeled is None):
            print("Empty weekly data in fold. Skipping.")
            continue
        
        # Train XGB on df_train_labeled
        from src.model import train_xgb_model, load_xgb_model, prepare_xy
        model = train_xgb_model(df_train_labeled, verbose=False)

        # We'll predict *all rows* of df_test_labeled to see how it performs
        # Instead of just the final row
        # For each weekly row in test, we compare predicted label to actual label
        X_test, y_test = prepare_xy(df_test_labeled)

        y_probs = model.predict_proba(X_test)  # shape: [n_test, 2]
        y_pred = model.predict(X_test)         # shape: [n_test,]

        # actual (0=Bearish, 1=Bullish)
        all_test_labels.extend(y_test.tolist())
        all_test_preds.extend(y_pred.tolist())

    # After all folds, compute final accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_test_labels, all_test_preds)
    print(f"\nWalk-Forward Accuracy across folds: {accuracy:.4f}")

if __name__ == "__main__":
    backtest_spy()
