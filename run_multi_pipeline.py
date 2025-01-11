#!/usr/bin/env python

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from src.config import Config
from src.pipeline import run_pipeline_for_symbol

"""
Usage:
    python run_multi_pipeline.py START_DATE END_DATE [TRAIN_FLAG]

Example:
    python run_multi_pipeline.py 2022-01-01 2025-01-10 True

- Tickers come from Config.TICKERS
- All tickers share the same start/end dates
- We'll store final predictions in a CSV, e.g. predictions_YYYYMMDD_HHMMSS.csv
- The predicted "next week" start/end is based on the console 'end_date'.
"""

def parse_bool(arg: str) -> bool:
    return arg.strip().lower() == "true"

def next_monday_after(date_str: str):
    """
    Given an 'end_date' (like '2025-01-10'), find the next Monday->Friday range.
    If 'end_date' is Friday, that Monday is +3 days, etc.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    dow = d.weekday()  # Monday=0, Tuesday=1, ..., Sunday=6
    # If dow=4 (Friday), next Monday = +3 days
    # If dow=0 (Monday), next Monday = +7 days
    # etc.
    days_until_monday = (7 - dow) if dow != 0 else 7
    monday_date = d + timedelta(days=days_until_monday)
    friday_date = monday_date + timedelta(days=4)
    return monday_date, friday_date

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_multi_pipeline.py START_DATE END_DATE [TRAIN_FLAG]")
        sys.exit(1)

    start_date = sys.argv[1]  # e.g. "2022-01-01"
    end_date   = sys.argv[2]  # e.g. "2025-01-10"

    train_flag = True
    if len(sys.argv) >= 4:
        train_flag = parse_bool(sys.argv[3])

    # 1) Figure out next Monday->Friday after end_date
    next_mon, next_fri = next_monday_after(end_date)

    # 2) We'll collect final predictions in a list of dicts
    predictions = []

    # 3) For each ticker in Config.TICKERS, call run_pipeline_for_symbol
    for symbol in Config.TICKERS:
        result = run_pipeline_for_symbol(symbol, start_date, end_date, train_flag)

        # Add the next Monday->Friday range to the result
        # (We override or add these fields to ensure consistent naming in the CSV)
        result["predicting_week_start"] = str(next_mon.date())
        result["predicting_week_end"]   = str(next_fri.date())

        predictions.append(result)

    # 4) Convert the list of results to a DataFrame
    df_preds = pd.DataFrame(predictions)

    # 5) Construct output filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"predictions_{timestamp_str}.csv"

    df_preds.to_csv(out_csv, index=False)
    print(f"[run_multi_pipeline.py] All predictions saved to {out_csv}")
