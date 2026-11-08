#!/usr/bin/env python3
"""
MF Point-to-Point Returns
Usage:
    python src/experiments/returns.py
    python src/experiments/returns.py --start 2026-10-01 --end 2026-11-06
"""

import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mf_data_provider import MfDataProvider

START = "2026-10-23"
END   = "2026-11-06"
RANKS_DIR = ROOT / "results" / "ranks"


def load_algo_ranks(target_date: str) -> pd.Series:
    """Return a Series of mfId -> mf-screener-ai_rank from the last ranking on or before target_date.
    
    If target_date is earlier than available rankings, returns an empty Series.
    """
    if not RANKS_DIR.exists():
        return pd.Series(dtype=float, name="mf-screener-ai_rank")

    frames = []
    for path in sorted(RANKS_DIR.glob("*.csv")):
        if path.name == "xirr.csv":
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        date_cols = sorted([c for c in df.columns if c not in ("mfId", "name")], reverse=True)
        eligible = [d for d in date_cols if d <= target_date]
        if not eligible:
            continue
        col = eligible[0]
        s = df[["mfId", col]].rename(columns={col: "mf-screener-ai_rank"}).dropna(subset=["mf-screener-ai_rank"])
        s["mfId"] = s["mfId"].astype(str).str.strip()
        frames.append(s)

    if not frames:
        return pd.Series(dtype=float, name="mf-screener-ai_rank")

    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index("mfId")["mf-screener-ai_rank"]


def point_return(chart: pd.DataFrame, start: str, end: str):
    df = chart.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["nav"]).sort_values("date")

    s = df[df["date"] <= start]
    e = df[df["date"] <= end]
    if s.empty or e.empty:
        return None

    nav_s = float(s.iloc[-1]["nav"])
    nav_e = float(e.iloc[-1]["nav"])
    if nav_s <= 0 or e.iloc[-1]["date"] <= s.iloc[-1]["date"]:
        return None

    return (nav_e - nav_s) / nav_s * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START)
    parser.add_argument("--end",   default=END)
    args = parser.parse_args()

    provider = MfDataProvider()
    funds = provider.list_all_mf()

    rows = []

    def fetch(f):
        mf_id = f["mfId"]
        try:
            chart = provider.get_mf_chart(mf_id, duration="5y")
            ret = point_return(chart, args.start, args.end)
            if ret is None:
                return None
            return {
                "mfId":       mf_id,
                "name":       f.get("name", mf_id),
                "category":   f.get("subsector", ""),
                "sector":     f.get("sector", ""),
                "aum_cr":     round(float(f["aum"]), 2) if pd.notnull(f.get("aum")) else None,
                f"return_{args.start}_to_{args.end}": round(ret, 2),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(fetch, f): f for _, f in funds.iterrows()}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)

    ret_col = f"return_{args.start}_to_{args.end}"
    result = pd.DataFrame(rows).sort_values(ret_col, ascending=False).reset_index(drop=True)
    result.index += 1
    result.index.name = "rank"

    ranks = load_algo_ranks(args.start)
    result["mf-screener-ai_rank"] = pd.to_numeric(result["mfId"].map(ranks), errors="coerce").astype("Int64")

    print(result.to_string())

    out = ROOT / "results" / "experiments" / f"returns_{args.start}_to_{args.end}.csv"
    try:
        result.to_csv(out)
        print(f"\nSaved to {out}")
    except PermissionError:
        pass


if __name__ == "__main__":
    main()
