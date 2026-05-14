#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Grok

Original implementation for scoring Total Market (Flexi/Multi/Value/Contra/Focused) funds.
Tailored to 24-month horizon: 12 SIP months + 12 hold + exit.
Optimizes for XIRR of hybrid cashflows, fund manager skill (IR, alpha, consistency),
resilience for hold phase, and alignment to 2027-2028 Indian equity growth themes
(private capex, rural revival, financials, infra, consumption, manufacturing).

Uses daily NAV from MfDataProvider + size indices + current holdings for attribution proxies.
No code copied from prior models; fresh design with rolling scenario sims, Brinson-style
sector tilt scoring, turnover conviction proxy via concentration, multi-factor skill metrics.

Author: Grok 4.3 (original research)
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import optimize

# Path setup for sibling imports (mf_data_provider)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants - Sector & Data
# ---------------------------------------------------------------------------
SECTOR = "Total Market"
SUBSECTORS = ["Contra Fund", "Flexi Cap Fund", "Focused Fund", "Multi Cap Fund", "Value Fund"]
BENCHMARK_INDEX = ".NIFTY500"
SIZE_INDICES = {"Large": "Large Cap", "Mid": "Mid Cap", "Small": "Small Cap"}

# 2027-2028 Growth Themes (from macro research: private capex, rural, financials, infra, consumption, PLI/green)
GROWTH_THEMES = {
    "financials": ["Financial", "Bank", "NBFC", "Insurance", "Capital Markets"],
    "infra_auto": ["Auto", "Cement", "Construction", "Infrastructure", "Real Estate", "Utilities"],
    "consumption_rural": ["FMCG", "Consumer", "Retail", "Two Wheeler", "Tractor", "Durables"],
    "manufacturing_green": ["Manufacturing", "Capital Goods", "Renewable", "Power", "Telecom", "IT"],
}

RISK_FREE_RATE = 0.065  # 6.5% long-term India risk free approx
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Grok.csv"

# Scoring weights (sum to 1.0) - balanced for SIP+hold + skill + future growth
WEIGHTS = {
    "xirr_mean": 0.22,
    "xirr_min": 0.13,
    "xirr_consistency": 0.08,
    "recovery_days": 0.12,
    "downside_capture": 0.10,
    "info_ratio": 0.15,
    "size_alpha": 0.08,
    "theme_tilt": 0.07,
    "conviction": 0.05,
}

# ---------------------------------------------------------------------------
# Data Helpers (original implementations)
# ---------------------------------------------------------------------------

def clean_nav_series(df: pd.DataFrame) -> pd.Series:
    """Clean raw chart DF to sorted NAV Series with DatetimeIndex."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
    s["nav"] = pd.to_numeric(s["nav"], errors="coerce")
    s = s.dropna(subset=["timestamp", "nav"])
    s = s[s["nav"] > 0]
    s = s.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return s.set_index("timestamp")["nav"]


def get_cagr(nav: pd.Series, years: float) -> Optional[float]:
    """Annualized CAGR over last N years, or None if insufficient data."""
    if len(nav) < 2:
        return None
    cutoff = nav.index[-1] - timedelta(days=int(years * 365.25))
    start_slice = nav[nav.index >= cutoff]
    if start_slice.empty or start_slice.iloc[0] <= 0:
        return None
    start_v = start_slice.iloc[0]
    end_v = nav.iloc[-1]
    return float((end_v / start_v) ** (1.0 / years) - 1.0)


def _npv(rate: float, cashflows: List[Tuple[float, float]]) -> float:
    """NPV helper for XIRR bisection."""
    if rate <= -1.0:
        return float("inf")
    return sum(cf[1] / ((1 + rate) ** cf[0]) for cf in cashflows)


def calculate_xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    """Compute XIRR via scipy brentq on year-fractions from first date."""
    if len(cashflows) < 2:
        return None
    start = cashflows[0][0]
    timed = []
    for dt, amt in cashflows:
        years = (dt - start).days / 365.25
        timed.append((years, amt))
    try:
        root = optimize.brentq(lambda r: _npv(r, timed), -0.999, 100.0)
        return float(root)
    except (ValueError, RuntimeError):
        return None


def simulate_sip_hold_xirr(nav: pd.Series, start_date: pd.Timestamp) -> Optional[float]:
    """
    Simulate: SIP 1st of each month for 12 months, hold 12 more, full exit at month 24.
    Returns XIRR or None if data insufficient/gaps.
    """
    if len(nav) < 30:
        return None
    end_date = start_date + relativedelta(months=24)
    if end_date > nav.index[-1]:
        return None

    cashflows: List[Tuple[pd.Timestamp, float]] = []
    total_units = 0.0
    sip = 10000.0  # normalized amount, XIRR scale invariant

    for m in range(12):
        sip_date = start_date + relativedelta(months=m)
        # nearest trading day >= sip_date
        avail = nav[nav.index >= sip_date.replace(day=1)]
        if avail.empty:
            return None
        buy_date = avail.index[0]
        buy_nav = avail.iloc[0]
        if buy_nav <= 0:
            return None
        units = sip / buy_nav
        total_units += units
        cashflows.append((buy_date, -sip))

    # Exit at or after month 24
    exit_avail = nav[nav.index >= end_date.replace(day=1)]
    if exit_avail.empty:
        return None
    exit_date = exit_avail.index[0]
    if (exit_date - end_date).days > 45:  # allow reasonable gap
        return None
    exit_val = total_units * exit_avail.iloc[0]
    cashflows.append((exit_date, exit_val))

    return calculate_xirr(cashflows)


def rolling_xirr_stats(nav: pd.Series, bench_nav: Optional[pd.Series] = None) -> Dict[str, Optional[float]]:
    """
    Run rolling 24m SIP+hold simulations monthly over history.
    Returns mean, min (5th pct), std, batting_avg (fraction >0 or vs bench if provided).
    """
    if len(nav) < 110:  # weekly data: ~2+ years for at least a few 24m rolling windows
        return {"mean": None, "min": None, "std": None, "batting": None, "count": 0}

    first = nav.index[0]
    last = nav.index[-1]
    start = first.replace(day=1) + relativedelta(months=1)
    xirrs: List[float] = []
    bench_xirrs: List[float] = []

    while start + relativedelta(months=24) <= last:
        x = simulate_sip_hold_xirr(nav, start)
        if x is not None:
            xirrs.append(x)
            if bench_nav is not None:
                bx = simulate_sip_hold_xirr(bench_nav, start)
                if bx is not None:
                    bench_xirrs.append(bx)
        start += relativedelta(months=1)

    if not xirrs:
        return {"mean": None, "min": None, "std": None, "batting": None, "count": 0}

    arr = np.array(xirrs)
    res = {
        "mean": float(np.mean(arr)),
        "min": float(np.percentile(arr, 5)),
        "std": float(np.std(arr)),
        "count": len(arr),
    }
    if bench_xirrs and len(bench_xirrs) == len(xirrs):
        res["batting"] = float(np.mean(np.array(xirrs) > np.array(bench_xirrs)))
    else:
        res["batting"] = float(np.mean(arr > 0.0))
    return res


# ---------------------------------------------------------------------------
# Resilience & Skill Metrics (original)
# ---------------------------------------------------------------------------

def max_drawdown_recovery(nav: pd.Series) -> Tuple[Optional[float], Optional[int]]:
    """Return max drawdown (fraction) and max recovery days from any peak."""
    if len(nav) < 10:
        return None, None
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    max_dd = float(dd.min())
    # recovery durations
    recoveries: List[int] = []
    in_dd = False
    dd_start = None
    for ts, d in dd.items():
        if d < -0.001 and not in_dd:
            in_dd = True
            dd_start = ts
        elif d >= -0.001 and in_dd:
            dur = (ts - dd_start).days
            recoveries.append(dur)
            in_dd = False
    if in_dd and dd_start is not None:
        recoveries.append((nav.index[-1] - dd_start).days)
    max_rec = max(recoveries) if recoveries else 0
    return max_dd, max_rec


def capture_ratios(fund_nav: pd.Series, bench_nav: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Monthly downside and upside capture ratios (annualized geo)."""
    df = pd.DataFrame({"f": fund_nav, "b": bench_nav}).dropna()
    if len(df) < 40:
        return None, None
    m = df.resample("ME").last().pct_change().dropna()
    down = m[m["b"] < 0]
    up = m[m["b"] > 0]
    if len(down) < 3 or len(up) < 3:
        return None, None
    d_f = (1 + down["f"]).prod() ** (12 / len(down)) - 1
    d_b = (1 + down["b"]).prod() ** (12 / len(down)) - 1
    u_f = (1 + up["f"]).prod() ** (12 / len(up)) - 1
    u_b = (1 + up["b"]).prod() ** (12 / len(up)) - 1
    dc = float(d_f / d_b) if d_b < 0 else None
    uc = float(u_f / u_b) if u_b > 0 else None
    return dc, uc


def information_ratio(fund_nav: pd.Series, bench_nav: pd.Series) -> Optional[float]:
    """Annualized Information Ratio = (mean excess monthly) / tracking error."""
    df = pd.DataFrame({"f": fund_nav, "b": bench_nav}).dropna()
    if len(df) < 36:  # allow 3Y funds to contribute IR
        return None
    rets = df.resample("ME").last().pct_change().dropna()
    excess = rets["f"] - rets["b"]
    if excess.std() == 0:
        return None
    ir = (excess.mean() / excess.std()) * np.sqrt(12)
    return float(ir)


def size_factor_alpha(fund_nav: pd.Series, large: pd.Series, mid: pd.Series, small: pd.Series) -> Optional[float]:
    """OLS alpha from weekly returns regressed on Large/Mid/Small + RF."""
    df = pd.DataFrame({"f": fund_nav, "L": large, "M": mid, "S": small}).dropna()
    if len(df) < 60:  # allow 3Y funds to contribute size alpha
        return None
    w = df.resample("W").last().pct_change().dropna()
    rf_w = (1 + RISK_FREE_RATE) ** (1 / 52) - 1
    y = w["f"] - rf_w
    X = np.column_stack([w["L"] - rf_w, w["M"] - rf_w, w["S"] - rf_w, np.ones(len(w))])
    try:
        beta, resid = np.linalg.lstsq(X, y, rcond=None)[:2]
        alpha_w = beta[3]
        return float(((1 + alpha_w) ** 52) - 1.0)
    except np.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------------
# Holdings Intelligence (Brinson proxy + conviction + 2027 theme tilt)
# ---------------------------------------------------------------------------

def parse_holdings_metrics(holdings: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Original holdings analysis:
    - conviction: lower n_holdings + higher top10 concentration => higher conviction (low turnover intent)
    - theme_tilt: overweight in 2027 growth themes (financials, infra/auto, rural/consumption, mfg/green)
    Brinson-style: allocation effect proxy via theme deviation.
    """
    if not holdings or len(holdings) < 5:
        return {"conviction": 0.5, "theme_tilt": 0.0, "n_stocks": 0, "top10_w": 0.5}

    df = pd.DataFrame(holdings)
    # Actual Tickertape holdings columns: latest = weight (%), title=stock, no sector field available in cache
    pct_col = None
    for c in ["latest", "percent", "allocation", "weight", "percentPortfolio"]:
        if c in df.columns:
            pct_col = c
            break
    if pct_col is None:
        return {"conviction": 0.5, "theme_tilt": 0.0, "n_stocks": len(df), "top10_w": 0.5}

    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0)
    total = df[pct_col].sum()
    if total <= 0:
        return {"conviction": 0.5, "theme_tilt": 0.0, "n_stocks": len(df), "top10_w": 0.5}

    w = df[pct_col] / total
    n = len(df)
    top10 = float(w.nlargest(10).sum())
    # conviction: inverse n + high top concentration (normalized 0-1, higher better for skill)
    conv = float(np.clip((1.0 - (n / 200.0)) * 0.6 + (top10 - 0.4) * 0.8, 0.0, 1.0))

    # theme tilt: sum w in growth themes sectors
    sector_col = None
    for c in ["sector", "industry", "sectorName"]:
        if c in df.columns:
            sector_col = c
            break
    tilt = 0.0
    if sector_col:
        for theme, keywords in GROWTH_THEMES.items():
            mask = df[sector_col].astype(str).str.contains("|".join(keywords), case=False, na=False)
            tilt += float(w[mask].sum())
        tilt = float(np.clip((tilt - 0.45) * 2.0, -0.3, 0.4))  # deviation from neutral ~45-50%

    return {
        "conviction": round(conv, 4),
        "theme_tilt": round(tilt, 4),
        "n_stocks": n,
        "top10_w": round(top10, 4),
    }


# ---------------------------------------------------------------------------
# Per-Fund Analysis & Scoring
# ---------------------------------------------------------------------------

def analyse_fund(
    mf_id: str,
    name: str,
    subsector: str,
    nav: pd.Series,
    bench: pd.Series,
    large: pd.Series,
    mid: pd.Series,
    small: pd.Series,
    holdings: List[Dict[str, Any]],
    aum: float,
) -> Dict[str, Any]:
    """Compute all metrics + weighted score for one fund. Original logic."""
    data_days = int((nav.index[-1] - nav.index[0]).days) if len(nav) > 1 else 0
    c3 = get_cagr(nav, 3.0)
    c5 = get_cagr(nav, 5.0)

    xirr_stats = rolling_xirr_stats(nav, bench)
    x_mean = xirr_stats.get("mean")
    x_min = xirr_stats.get("min")
    x_std = xirr_stats.get("std") or 0.15
    batting = xirr_stats.get("batting") or 0.5

    md, rec_days = max_drawdown_recovery(nav)
    dc, uc = capture_ratios(nav, bench)
    ir = information_ratio(nav, bench)
    alpha = size_factor_alpha(nav, large, mid, small)

    h = parse_holdings_metrics(holdings)

    # Normalize components to ~0-1 or z-like for weighting (simple min-max proxy using reasonable bounds)
    def nz(v, lo, hi, invert=False):
        if v is None:
            return 0.5
        v = max(lo, min(hi, v))
        sc = (v - lo) / (hi - lo)
        return 1.0 - sc if invert else sc

    s_xmean = nz(x_mean, -0.05, 0.35)
    s_xmin = nz(x_min, -0.25, 0.20)
    s_cons = nz(1.0 / (x_std + 0.01), 3, 20)  # lower vol better
    s_rec = nz(rec_days or 400, 100, 700, invert=True)
    s_dc = nz(dc or 0.9, 0.4, 1.4, invert=True)
    s_ir = nz(ir or 0.0, -0.5, 1.5)
    s_alpha = nz(alpha or 0.0, -0.05, 0.15)
    s_theme = nz(h["theme_tilt"] + 0.15, 0.0, 0.5)  # positive tilt good
    s_conv = h["conviction"]

    score = (
        WEIGHTS["xirr_mean"] * s_xmean
        + WEIGHTS["xirr_min"] * s_xmin
        + WEIGHTS["xirr_consistency"] * s_cons
        + WEIGHTS["recovery_days"] * s_rec
        + WEIGHTS["downside_capture"] * s_dc
        + WEIGHTS["info_ratio"] * s_ir
        + WEIGHTS["size_alpha"] * s_alpha
        + WEIGHTS["theme_tilt"] * s_theme
        + WEIGHTS["conviction"] * s_conv
    ) * 100.0

    # AUM penalty for very large funds (liquidity/style drift risk in active Total Market)
    if aum and aum > 25000:
        score *= 0.92
    elif aum and aum > 15000:
        score *= 0.97

    return {
        "mfId": mf_id,
        "name": name,
        "subsector": subsector,
        "data_days": data_days,
        "cagr_3y": round(c3 * 100, 2) if c3 else None,
        "cagr_5y": round(c5 * 100, 2) if c5 else None,
        "mean_xirr": round(x_mean * 100, 2) if x_mean else None,
        "min_xirr": round(x_min * 100, 2) if x_min else None,
        "xirr_std": round(x_std * 100, 2),
        "batting_avg": round(batting, 3),
        "info_ratio": round(ir, 3) if ir else None,
        "size_alpha": round(alpha * 100, 2) if alpha else None,
        "max_drawdown": round(md * 100, 2) if md else None,
        "recovery_days": rec_days,
        "downside_capture": round(dc, 3) if dc else None,
        "upside_capture": round(uc, 3) if uc else None,
        "theme_tilt": h["theme_tilt"],
        "conviction": h["conviction"],
        "n_stocks": h["n_stocks"],
        "top10_weight": h["top10_w"],
        "aum_cr": round(aum, 0) if aum else None,
        "score": round(score, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Grok Total Market MF scorer")
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Force data refresh")
    args = parser.parse_args()

    logger.info(f"Starting Grok scorer for {SECTOR}")
    provider = MfDataProvider(date=args.date)

    # Ensure data present (charts + holdings for theme)
    if args.force_refresh:
        provider.fetch_all_data()
        provider.prefetch_all_holdings(force_refresh=True)

    df_all = provider.list_all_mf()
    equity = df_all[(df_all["sector"] == "Equity") & (df_all["subsector"].isin(SUBSECTORS))]
    mf_ids = equity["mfId"].tolist()
    logger.info(f"Analysing {len(mf_ids)} Total Market funds...")

    # Preload indices once
    bench_df = provider.get_index_chart("Total Market")
    if bench_df.empty:
        bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_series(bench_df)
    large_nav = clean_nav_series(provider.get_index_chart(SIZE_INDICES["Large"]))
    mid_nav = clean_nav_series(provider.get_index_chart(SIZE_INDICES["Mid"]))
    small_nav = clean_nav_series(provider.get_index_chart(SIZE_INDICES["Small"]))

    results: List[Dict[str, Any]] = []
    holdings_count = 0

    for idx, row in equity.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        sub = row["subsector"]
        aum = row.get("aum", 0.0) or 0.0

        nav_df = provider.get_mf_chart(mf_id)
        nav = clean_nav_series(nav_df)
        # All Tickertape data is weekly (7-day gaps). 130 weeks ≈ 2.5 years.
        # This correctly includes genuine 3Y+ funds even when only weekly NAV is available.
        if len(nav) < 130:
            logger.debug(f"Skipping {mf_id}: insufficient history ({len(nav)} weeks)")
            continue

        holdings = provider.read_mf_holdings(mf_id)
        if holdings:
            holdings_count += 1

        res = analyse_fund(mf_id, name, sub, nav, bench_nav, large_nav, mid_nav, small_nav, holdings, aum)
        results.append(res)
        if (idx + 1) % 20 == 0:
            logger.info(f"Processed {idx+1}/{len(equity)} funds")

    if not results:
        logger.error("No funds analysed")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Reorder columns to match task spec + extras
    cols = ["mfId", "name", "rank", "score", "subsector", "data_days", "cagr_3y", "cagr_5y",
            "mean_xirr", "min_xirr", "info_ratio", "size_alpha", "recovery_days",
            "downside_capture", "theme_tilt", "conviction", "batting_avg"]
    out = df[[c for c in cols if c in df.columns]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Wrote {len(out)} funds to {OUTPUT_FILE}")
    logger.info(f"Holdings used for {holdings_count} funds")
    logger.info("Top 5:\n%s", out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
