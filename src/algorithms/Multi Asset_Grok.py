#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok Multi Asset scorer — 12-month SIP + 12-month hold XIRR.

Gold still has a structural bid; silver already melted up in 2025.
Winning pack: return-adequacy (no 8% CAGR "safety" funds) + gold
participation + late-silver penalty + equity downside protection.
IR vs Nifty 500 is not used. Short metals-2025 products are shrunk.

Run: python src/algorithms/Multi\\ Asset_Grok.py [--date YYYY-MM-DD] [--backtest]
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
    p = argparse.ArgumentParser(description="Multi Asset MF screener (Grok)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--backtest", action="store_true", help="Walk-forward feature IC vs realized 24m SIP-hold XIRR")
    args = p.parse_args()
    run("Multi Asset", date=args.date, backtest=args.backtest)


if __name__ == "__main__":
    main()
