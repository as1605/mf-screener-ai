#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok Small Cap scorer — 12-month SIP + 12-month hold XIRR.

2025-26 reset: valuations cooled, FY27-28 earnings is the bull case, but
walk-forward IC says information ratio recently reversed. Prefer left-tail
SIP-hold XIRR, path consistency (xirr_std, hold_vol), ulcer / relative
drawdown and Sortino. Shrink 2023-24 launches; haircut mega-AUM.

Run: python src/algorithms/Small\\ Cap_Grok.py [--date YYYY-MM-DD] [--backtest]
"""

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grok_engine import run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Small Cap MF screener (Grok)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--backtest", action="store_true", help="Walk-forward feature IC vs realized 24m SIP-hold XIRR")
    args = p.parse_args()
    run("Small Cap", date=args.date, backtest=args.backtest)


if __name__ == "__main__":
    main()
