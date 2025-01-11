# run_multi_pipeline.py
#!/usr/bin/env python

import sys
import os
import pandas as pd
from datetime import datetime
from src.config import Config
from src.pipeline import run_pipeline_for_symbol

"""
Usage:
    python run_multi_pipeline.py START_DATE END_DATE [TRAIN_FLAG]

Example:
    python run_multi_pipeline.py 2022-01-01 2025-01-10 True

- Tickers are read from Config.TICKERS
- All tickers use the same start/end date
- We'll store final predictions in a CSV, e.g. predictions_YYYYMMDD_HHMMSS.csv
"""

def parse_bool(arg: str) -> bool:
    return arg.strip().lower() == "true"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_multi_pipeline.py START_DATE END_DATE [TRAIN_FLAG]")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]

    train_flag = True
    if len(sys.argv) >= 4:
        train_flag = parse_bool(sys.argv[3])

    # We'll collect predictions in a list of dicts
    predictions = []

    for symbol in Config.TICKERS:
        # Call pipeline for each symbol
        result = run_pipeline_for_symbol(symbol, start_date, end_date, train_flag)
        predictions.append(result)

    # Convert to DataFrame
    df_preds = pd.DataFrame(predictions)

    # Construct output filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"predictions_{timestamp_str}.csv"

    df_preds.to_csv(out_csv, index=False)
    print(f"[run_multi_pipeline.py] All predictions saved to {out_csv}")

