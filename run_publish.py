#!/usr/bin/env python3
"""
Compile results from all model TSVs per sector and publish to Google Sheet.
Runs for every sector that has at least one results/{SECTOR}_{model}.csv file.
"""
import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from compile_results import discover_sectors, compile_and_write
from publish_sheet import load_env, publish_sector


def main():
    load_env()
    sectors = discover_sectors()
    if not sectors:
        print("No sector result files found (results/{SECTOR}_{model}.csv).", file=sys.stderr)
        sys.exit(1)
    for sector in sectors:
        compile_and_write(sector)
        publish_sector(sector)
        print(f"Compiled and published: {sector}")


if __name__ == "__main__":
    main()
