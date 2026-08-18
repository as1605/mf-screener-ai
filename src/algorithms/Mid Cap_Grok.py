#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok Mid Cap scorer — 12-month SIP + 12-month hold XIRR.

Style purity and information ratio have the strongest walk-forward IC.
Safety-first packs (ulcer / hold-dd) had a negative top-5 edge. shrink_k
is high so a 3-4 year launch cannot outrank a long Invesco/Edelweiss path.
AUM is a mild capacity haircut, not a rank floor.

Run: python src/algorithms/Mid\\ Cap_Grok.py [--date YYYY-MM-DD] [--backtest]
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
    p = argparse.ArgumentParser(description="Mid Cap MF screener (Grok)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--backtest", action="store_true", help="Walk-forward feature IC vs realized 24m SIP-hold XIRR")
    args = p.parse_args()
    run("Mid Cap", date=args.date, backtest=args.backtest)


if __name__ == "__main__":
    main()
