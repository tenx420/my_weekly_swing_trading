# run_pipeline.py
#!/usr/bin/env python

import sys
from src.pipeline import run_pipeline

"""
Usage:
    python run_pipeline.py SYMBOL START_DATE END_DATE [TRAIN_FLAG]

Example:
    python run_pipeline.py AAPL 2022-01-01 2025-01-10 True
"""

def parse_bool(arg: str) -> bool:
    return arg.strip().lower() == "true"

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_pipeline.py SYMBOL START_DATE END_DATE [TRAIN_FLAG]")
        sys.exit(1)

    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    train_flag = True
    if len(sys.argv) >= 5:
        train_flag = parse_bool(sys.argv[4])

    run_pipeline(symbol, start_date, end_date, train_flag)

