#!/usr/bin/env python3
"""
Compile sector results and optionally publish to Google Sheets.
Runs for every sector that has at least one results/{SECTOR}_{model}.csv file.
"""
import argparse
import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from compile_results import discover_sectors, compile_and_write
from publish_sheet import load_env, publish_sector, publish_xirr_and_ranks


def main():
    p = argparse.ArgumentParser(description="Compile sector CSVs; optionally publish to Google Sheets.")
    p.add_argument(
        "--no-sheet",
        action="store_true",
        help="Only compile results/*.csv → sector sheets on disk; do not update Google Sheet.",
    )
    p.add_argument(
        "--ranks",
        action="store_true",
        help="Also publish XIRR and rank-history worksheets to Google Sheets.",
    )
    args = p.parse_args()

    if not args.no_sheet:
        load_env()
    sectors = discover_sectors()
    if not sectors:
        print("No sector result files found (results/{SECTOR}_{model}.csv).", file=sys.stderr)
        sys.exit(1)
    for sector in sectors:
        compile_and_write(sector)
        if args.no_sheet:
            print(f"Compiled (no sheet): {sector}")
        else:
            publish_sector(sector)
            print(f"Compiled and published: {sector}")

    if not args.no_sheet and args.ranks:
        publish_xirr_and_ranks()
        print("Published XIRR and rank-history worksheets")


if __name__ == "__main__":
    main()
