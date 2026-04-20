#!/usr/bin/env python3
"""
Experiment: focused portfolio / return snapshot.

Writes ``results/experiments/focused.csv`` by default. Builds a CSV with:
  mfId, name, sector, subsector, aum, first_nav_date, top5, top10, top20, 1y, 3y, 5y

- first_nav_date: earliest date (YYYY-MM-DD) with NAV in the cached chart for that fund.

- top5 / top10 / top20: sum of portfolio weights (%) of the largest 5 / 10 / 20 holdings.
- 1y / 3y / 5y: trailing absolute total return (%) over ~1, 3, 5 calendar years.
  Use ``--returns-as-of YYYY-MM-DD`` to end each window on the last NAV on or before that
  date (default: latest NAV in the chart).

**Holdings vs returns:** top5 / top10 / top20 always use **current** holdings from cache,
not historical allocations as of ``--returns-as-of``.

Flow:
  1. ``MfDataProvider.prefetch_all_holdings()`` — parallel API fetch → ``portfolio/*.csv``
  2. Per fund: ``read_mf_chart`` / ``read_mf_holdings`` — disk only

Requires NAV CSVs under data/<date>/mf/ (run mf_data_provider fetch first).

Usage (from repo root):
  python3 src/experiments/focused.py
  python3 src/experiments/focused.py --date 2026-03-18 --returns-as-of 2024-12-31 --limit 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mf_data_provider import MfDataProvider  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _holding_weight_pct(h: Dict[str, Any]) -> Optional[float]:
    """Parse allocation weight as percentage of AUM (0–100).

    Tickertape ``currentAllocation`` uses ``latest`` as % of portfolio (e.g. 4.2 means 4.2%).
    """
    v = h.get("latest")
    if v is not None:
        try:
            x = float(v)
            if x > 0:
                return round(min(x, 100.0), 8)
        except (TypeError, ValueError):
            pass

    for key in (
        "nav",
        "allocation",
        "allocationPct",
        "weight",
        "holdingPct",
        "percent",
        "pct",
        "holding",
        "portfolioShare",
    ):
        v = h.get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if 0 < x <= 1.0001:
            return round(x * 100.0, 8)
        if 1.0 < x <= 100.0:
            return round(x, 8)
    return None


def concentration_pct(holdings: List[Dict[str, Any]]) -> tuple[float, float, float]:
    weights: List[float] = []
    for h in holdings:
        w = _holding_weight_pct(h)
        if w is not None and w > 0:
            weights.append(w)
    weights.sort(reverse=True)
    return (
        round(sum(weights[:5]), 2),
        round(sum(weights[:10]), 2),
        round(sum(weights[:20]), 2),
    )


def _align_as_of(nav_index: pd.Index, as_of: pd.Timestamp) -> pd.Timestamp:
    """Make as_of comparable to nav index (UTC if index is tz-aware)."""
    ts = pd.Timestamp(as_of)
    if getattr(nav_index, "tz", None) is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    return ts


def trailing_return_pct(
    chart: pd.DataFrame,
    years: float,
    as_of: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    """Absolute return (%) from NAV ~`years` before end to end (last NAV ≤ as_of, or latest)."""
    if chart is None or chart.empty or "nav" not in chart.columns:
        return None
    df = chart[["timestamp", "nav"]].dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["timestamp", "nav"]).sort_values("timestamp")
    if len(df) < 2:
        return None
    nav = df.set_index("timestamp")["nav"]
    if as_of is not None:
        cutoff = _align_as_of(nav.index, as_of)
        nav = nav[nav.index <= cutoff]
    if nav.empty or len(nav) < 2:
        return None
    end = float(nav.iloc[-1])
    end_ts = nav.index[-1]
    if end <= 0:
        return None
    start_ts = end_ts - pd.DateOffset(days=int(years * 365.25))
    past = nav[nav.index <= start_ts]
    if past.empty:
        return None
    start = float(past.iloc[-1])
    if start <= 0:
        return None
    return round((end / start - 1.0) * 100.0, 2)


def first_nav_date_str(chart: pd.DataFrame) -> str:
    """Earliest NAV date in chart (YYYY-MM-DD). Full series; not clipped by --returns-as-of."""
    if chart is None or chart.empty or "nav" not in chart.columns:
        return ""
    df = chart[["timestamp", "nav"]].dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["timestamp", "nav"]).sort_values("timestamp")
    if df.empty:
        return ""
    return pd.Timestamp(df["timestamp"].iloc[0]).strftime("%Y-%m-%d")


def main() -> None:
    p = argparse.ArgumentParser(description="Focused experiment: concentration + trailing returns")
    p.add_argument("--date", default=None, help="data/<date>/ (default: today)")
    p.add_argument("--limit", type=int, default=0, help="Max funds (0 = all)")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "experiments" / "focused.csv",
        help="Output CSV path",
    )
    p.add_argument("--refresh-holdings", action="store_true", help="Refetch all holdings (prefetch)")
    p.add_argument(
        "--returns-as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "End date for 1y/3y/5y returns: use last NAV on or before this date as the "
            "return endpoint (default: latest NAV in each fund's chart)."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Threads for holdings prefetch (default: provider MF_HOLDINGS_MAX_WORKERS)",
    )
    args = p.parse_args()

    logger.warning(
        "top5, top10, and top20 are based on **current** holdings from the API, "
        "not historical portfolio weights for any past date."
    )
    returns_as_of_ts: Optional[pd.Timestamp] = None
    if args.returns_as_of:
        try:
            returns_as_of_ts = pd.Timestamp(args.returns_as_of)
        except ValueError:
            logger.error("Invalid --returns-as-of date: %s", args.returns_as_of)
            sys.exit(1)
        logger.info(
            "1y / 3y / 5y returns end on the last NAV on or before %s",
            args.returns_as_of,
        )

    provider = MfDataProvider(base_dir=str(ROOT / "data"), date=args.date)
    df_list = provider.list_all_mf()
    if df_list.empty:
        logger.error("No ALL.csv / fund list. Run: python3 src/mf_data_provider.py")
        sys.exit(1)

    n = len(df_list)
    if args.limit and args.limit > 0:
        df_list = df_list.head(args.limit)
        n = len(df_list)

    mf_ids = [str(x) for x in df_list["mfId"].tolist()]
    provider.prefetch_all_holdings(
        max_workers=args.workers,
        force_refresh=args.refresh_holdings,
        mf_ids=mf_ids,
    )

    rows: List[Dict[str, Any]] = []
    for num, (_, rec) in enumerate(df_list.iterrows(), 1):
        mf_id = str(rec["mfId"])
        if num % 200 == 0 or num == 1:
            logger.info("Building rows %d / %d (%s)", num, n, mf_id)

        chart = provider.read_mf_chart(mf_id)
        holdings = provider.read_mf_holdings(mf_id)
        top5, top10, top20 = concentration_pct(holdings)

        r1 = trailing_return_pct(chart, 1.0, as_of=returns_as_of_ts)
        r3 = trailing_return_pct(chart, 3.0, as_of=returns_as_of_ts)
        r5 = trailing_return_pct(chart, 5.0, as_of=returns_as_of_ts)

        row: Dict[str, Any] = {
            "mfId": mf_id,
            "name": rec.get("name", ""),
            "sector": rec.get("sector", ""),
            "subsector": rec.get("subsector", ""),
            "aum": rec.get("aum", ""),
            "first_nav_date": first_nav_date_str(chart),
            "top5": top5 if holdings else "",
            "top10": top10 if holdings else "",
            "top20": top20 if holdings else "",
            "1y": r1 if r1 is not None else "",
            "3y": r3 if r3 is not None else "",
            "5y": r5 if r5 is not None else "",
        }
        if args.returns_as_of:
            row["returns_as_of"] = args.returns_as_of
        rows.append(row)

    out = pd.DataFrame(rows)
    base_cols = [
        "mfId",
        "name",
        "sector",
        "subsector",
        "aum",
        "first_nav_date",
        "top5",
        "top10",
        "top20",
    ]
    if args.returns_as_of:
        out = out[base_cols + ["returns_as_of", "1y", "3y", "5y"]]
    else:
        out = out[base_cols + ["1y", "3y", "5y"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Wrote %d rows → %s", len(out), args.output)


if __name__ == "__main__":
    main()
