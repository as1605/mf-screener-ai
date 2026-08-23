"""Create the weekly, per-algorithm XIRR scorecard."""

from __future__ import annotations

import math
import re
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from mf_data_provider import MfDataProvider


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RANKS_DIR = RESULTS_DIR / "ranks"
BRANCH_DATE_PREFIX = "date/"
INVESTMENT_PER_RUN = 1000.0
TOP_FUNDS = 4
DAY_COUNT = 365.2425
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_run_dates() -> list[str]:
    """Return the dated screener branches in chronological order."""
    branches = subprocess.run(
        ["git", "branch", "-r", "--format=%(refname:short)"], cwd=ROOT_DIR,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    prefix = f"origin/{BRANCH_DATE_PREFIX}"
    return sorted({b[len(prefix):] for b in branches if b.startswith(prefix) and DATE_RE.fullmatch(b[len(prefix):])})


def read_portfolio_rankings(date: str) -> dict[str, tuple[pd.DataFrame, str]]:
    """Load model and combined-sector rankings from one dated branch."""
    branch = f"origin/{BRANCH_DATE_PREFIX}{date}"
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", branch, "--", "results"], cwd=ROOT_DIR,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    rankings: dict[str, tuple[pd.DataFrame, str]] = {}
    for path in files:
        result_path = Path(path)
        algorithm = result_path.stem
        # Experiments also use result CSVs, but are not weekly screener portfolios.
        if result_path.parent != Path("results") or not path.endswith(".csv"):
            continue
        content = subprocess.run(
            ["git", "show", f"{branch}:{path}"], cwd=ROOT_DIR,
            check=True, capture_output=True, text=True,
        ).stdout
        df = pd.read_csv(StringIO(content))
        if "_" in algorithm and {"mfId", "rank"}.issubset(df.columns):
            rankings[algorithm] = (df, "rank")
        elif "_" not in algorithm and {"mfId", "total_rank"}.issubset(df.columns):
            rankings[algorithm] = (df, "total_rank")
    return rankings


def top_funds(results: pd.DataFrame, rank_column: str) -> list[str]:
    ranked = results[["mfId", rank_column]].copy()
    ranked[rank_column] = pd.to_numeric(ranked[rank_column], errors="coerce")
    ids = ranked.dropna().sort_values(rank_column).drop_duplicates("mfId")["mfId"].head(TOP_FUNDS).tolist()
    if len(ids) != TOP_FUNDS:
        raise ValueError(f"Expected {TOP_FUNDS} unique ranked funds, found {len(ids)}")
    return ids


def clean_chart(chart: pd.DataFrame) -> pd.Series:
    chart = chart[["timestamp", "nav"]].copy()
    chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True, errors="coerce")
    chart["nav"] = pd.to_numeric(chart["nav"], errors="coerce")
    chart = chart.dropna().loc[lambda frame: frame["nav"] > 0].sort_values("timestamp")
    return chart.drop_duplicates("timestamp", keep="last").set_index("timestamp")["nav"]


def nav_as_of(chart: pd.Series, date: str, mf_id: str) -> float:
    nav = chart.loc[chart.index <= pd.Timestamp(date, tz="UTC")]
    if nav.empty:
        raise ValueError(f"No NAV on or before {date} for {mf_id}")
    return float(nav.iloc[-1])


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """Solve XIRR with bisection; None means there is not enough information yet."""
    dates = np.array([(date - cashflows[0][0]).days / DAY_COUNT for date, _ in cashflows])
    amounts = np.array([amount for _, amount in cashflows], dtype=float)
    if dates.max() <= 0 or not (np.any(amounts < 0) and np.any(amounts > 0)):
        return None

    def npv(rate: float) -> float:
        return float(np.sum(amounts / np.power(1 + rate, dates)))

    low, high = -0.999999, 1.0
    f_low, f_high = npv(low), npv(high)
    while f_low * f_high > 0 and high < 1_000_000:
        high *= 2
        f_high = npv(high)
    if not math.isfinite(f_low) or not math.isfinite(f_high) or f_low * f_high > 0:
        return None
    for _ in range(200):
        middle = (low + high) / 2
        f_middle = npv(middle)
        if abs(f_middle) < 1e-8:
            return middle
        if f_low * f_middle <= 0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    return (low + high) / 2


def build_xirr_scores(dates: list[str] | None = None) -> pd.DataFrame:
    dates = dates or get_run_dates()
    run_results = {date: read_portfolio_rankings(date) for date in dates}
    fund_ids = sorted({
        mf_id
        for rankings in run_results.values()
        for results, rank_column in rankings.values()
        for mf_id in top_funds(results, rank_column)
    })

    # A dedicated cache avoids refetching NAVs on subsequent scorecard rebuilds.
    provider = MfDataProvider(base_dir=str(ROOT_DIR / "data"), date="xirr")
    charts = {mf_id: clean_chart(provider.fetch_mf_chart(mf_id, duration="max")) for mf_id in fund_ids}
    if any(chart.empty for chart in charts.values()):
        missing = [mf_id for mf_id, chart in charts.items() if chart.empty]
        raise ValueError(f"No NAV history returned for: {', '.join(missing)}")

    units: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    transactions: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for date in dates:
        for algorithm, (results, rank_column) in run_results[date].items():
            for mf_id in top_funds(results, rank_column):
                amount = INVESTMENT_PER_RUN / TOP_FUNDS
                units[algorithm][mf_id] += amount / nav_as_of(charts[mf_id], date, mf_id)
                transactions[algorithm].append((pd.Timestamp(date), -amount))

        row: dict[str, object] = {"date": date}
        for algorithm, holdings in units.items():
            terminal_value = sum(held_units * nav_as_of(charts[mf_id], date, mf_id) for mf_id, held_units in holdings.items())
            rate = xirr([*transactions[algorithm], (pd.Timestamp(date), terminal_value)])
            row[algorithm] = np.nan if rate is None else rate * 100
        rows.append(row)
    scores = pd.DataFrame(rows)
    return scores.reindex(columns=["date", *sorted(c for c in scores.columns if c != "date")])


def write_xirr_scores() -> Path:
    output = RANKS_DIR / "xirr.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    build_xirr_scores().to_csv(output, index=False, float_format="%.2f")
    return output


if __name__ == "__main__":
    print(f"Wrote XIRR scores to {write_xirr_scores()}")
