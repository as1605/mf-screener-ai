#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok Total Market scorer — 12-month SIP + 12-month hold XIRR.

Flexi / multi / focused / value / contra. Large caps are the cheap sleeve
into 2026-28. Score residual alpha, IR, 12+12 batting vs Nifty 500, and
capture-spread. New-fund 1.0 batting averages are shrunk. Theme/conviction
holdings features are gone — they overfit the last cycle.

Run: python src/algorithms/Total\\ Market_Grok.py [--date YYYY-MM-DD] [--backtest]
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
    p = argparse.ArgumentParser(description="Total Market MF screener (Grok)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--backtest", action="store_true", help="Walk-forward feature IC vs realized 24m SIP-hold XIRR")
    args = p.parse_args()
    run("Total Market", date=args.date, backtest=args.backtest)


if __name__ == "__main__":
    main()
