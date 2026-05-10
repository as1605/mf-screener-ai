#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Grok Small Cap Scoring Engine – Earnings Rebound & SIP Resilience
================================================================

Original forward-looking model for 1-year SIP XIRR in Indian Small Cap funds,
targeting 2027-2028 outperformance driven by earnings recovery, volatility
resilience and consistent compounding. Uses only NAV series + Small Cap index
(.NISM250). 8 custom features with median/MAD z-scores, research-informed
weights prioritizing rebound capture and drawdown recovery over raw momentum.

Target: high score variance, unique ranks, top funds show strong post-2025
rebound participation with controlled SIP pain.

Run: python src/algorithms/Small\ Cap_Grok.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GrokSmallCap")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider


def to_weekly(df: pd.DataFrame, date_col: str = "timestamp", val_col: str = "nav") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    weekly = df[val_col].resample("W-FRI").last().ffill(limit=2)
    return weekly.reset_index().rename(columns={val_col: "nav"})


def aligned_weekly_returns(fund_df: pd.DataFrame, bench_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    fw = to_weekly(fund_df)
    bw = to_weekly(bench_df)
    m = pd.merge(fw, bw, on="timestamp", suffixes=("_f", "_b"), how="inner").sort_values("timestamp")
    rf = m["nav_f"].pct_change().dropna()
    rb = m["nav_b"].pct_change().dropna()
    idx = rf.index.intersection(rb.index)
    return rf.loc[idx], rb.loc[idx]


def compute_cagr(nav: pd.Series, timestamps: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    ts = pd.to_datetime(timestamps)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    span_days = (ts.iloc[-1] - ts.iloc[0]).days
    years = span_days / 365.25
    if years < 1.0:
        return np.nan
    return (1 + total) ** (1 / years) - 1 if years > 0 else np.nan


def xirr(cashflows: List[float], dates: List[pd.Timestamp]) -> float:
    if len(cashflows) < 2:
        return np.nan
    def npv(r):
        return sum(cf / ((1 + r) ** ((d - dates[0]).days / 365.25)) for cf, d in zip(cashflows, dates))
    try:
        return brentq(npv, -0.99, 10.0)
    except:
        return np.nan


def simulate_monthly_sip_xirr(nav_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, invest: float = 1000.0) -> float:
    nav_df = nav_df.copy()
    nav_df["timestamp"] = pd.to_datetime(nav_df["timestamp"])
    nav_df = nav_df.set_index("timestamp").sort_index()
    dates = pd.date_range(start=start, end=end, freq="MS")
    cashflows, cf_dates, total_units = [], [], 0.0
    for d in dates:
        try:
            nav = nav_df.loc[:d, "nav"].iloc[-1]
            if pd.isna(nav) or nav <= 0:
                continue
            units = invest / nav
            cashflows.append(-invest)
            cf_dates.append(d)
            total_units += units
        except:
            continue
    if not cashflows:
        return np.nan
    final_nav = nav_df.loc[:end, "nav"].iloc[-1]
    if pd.isna(final_nav) or final_nav <= 0:
        return np.nan
    cashflows.append(total_units * final_nav)
    cf_dates.append(end)
    return xirr(cashflows, cf_dates)


def winsor(x: np.ndarray, p: float = 0.02) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [p*100, (1-p)*100])
    return np.clip(x, lo, hi)


# ---------------------------------------------------------------------------
# Original 8 features for Small Cap (earnings rebound + SIP resilience focus)
# ---------------------------------------------------------------------------
def rebound_capture(rf: np.ndarray, rb: np.ndarray, weeks: int = 26) -> float:
    if len(rf) < weeks + 4:
        return 0.0
    recent_rf = rf[-weeks:]
    recent_rb = rb[-weeks:]
    up_mask = recent_rb > 0
    if not np.any(up_mask):
        return 0.0
    return float(np.mean(recent_rf[up_mask]) - np.mean(recent_rb[up_mask]))


def sip_consistency(nav_df: pd.DataFrame, rolls: int = 8) -> Tuple[float, float]:
    nav_df = nav_df.copy()
    nav_df["timestamp"] = pd.to_datetime(nav_df["timestamp"])
    nav_df = nav_df.set_index("timestamp").sort_index()
    end = nav_df.index[-1]
    xirrs = []
    for k in range(rolls):
        s = end - pd.DateOffset(months=12 + k*3)
        if s < nav_df.index[0]:
            continue
        x = simulate_monthly_sip_xirr(nav_df.reset_index(), s, end)
        if not np.isnan(x):
            xirrs.append(x)
    if len(xirrs) < 3:
        return 0.0, 0.0
    arr = np.array(xirrs)
    return float(np.mean(arr)), float(1.0 / (1.0 + np.std(arr)))


def recovery_ratio(rf: np.ndarray, rb: np.ndarray, dd_thresh: float = 0.08) -> float:
    if len(rb) < 30:
        return 0.0
    cum_b = np.cumprod(1 + rb) - 1
    peak = np.maximum.accumulate(cum_b)
    dd = (cum_b - peak) / (peak + 1e-9)
    recs = []
    for i in range(len(dd) - 12):
        if dd[i] <= -dd_thresh:
            post = rf[i+1:i+13] - rb[i+1:i+13]
            recs.append(np.sum(post))
    return float(np.median(recs)) if recs else 0.0


def vol_adjusted_alpha(rf: np.ndarray, rb: np.ndarray) -> float:
    if len(rf) < 20:
        return 0.0
    ex = rf - rb
    return float(np.mean(ex) / (np.std(ex) + 1e-6))


def downside_resilience(rf: np.ndarray, rb: np.ndarray) -> float:
    mask = rb < 0
    if not np.any(mask):
        return 1.0
    fund_loss = np.mean(rf[mask])
    bench_loss = np.mean(rb[mask])
    return 1.0 - (fund_loss / bench_loss) if bench_loss != 0 else 0.0


def momentum_persistence(rf: np.ndarray) -> float:
    if len(rf) < 20:
        return 0.0
    return float(np.corrcoef(rf[:-1], rf[1:])[0, 1])


def drawdown_avoidance(rf: np.ndarray, rb: np.ndarray) -> float:
    def max_dd(r):
        c = np.cumprod(1 + r) - 1
        p = np.maximum.accumulate(c)
        return np.min(c - p)
    dd_f = max_dd(rf)
    dd_b = max_dd(rb)
    if dd_b == 0 or abs(dd_b) < 1e-6:
        return 1.0
    # Ratio of drawdown magnitudes (both negative). Lower ratio = fund avoided drawdown better.
    ratio = abs(dd_f) / abs(dd_b)
    # Convert to avoidance score: 1.0 = matched benchmark, >1 = better avoidance, <1 = worse
    avoidance = 1.0 / (ratio + 1e-6)
    return float(np.clip(avoidance, 0.4, 2.5))


def earnings_acceleration(nav_df: pd.DataFrame, short_w: int = 8, long_w: int = 26) -> float:
    nav_df = nav_df.copy()
    nav_df["timestamp"] = pd.to_datetime(nav_df["timestamp"])
    nav_df = nav_df.set_index("timestamp").sort_index()
    ret = nav_df["nav"].pct_change().dropna()
    if len(ret) < long_w + 4:
        return 0.0
    short = ret.rolling(short_w).mean().iloc[-1]
    long = ret.rolling(long_w).mean().iloc[-1]
    return float(short - long)


def score_fund(mf_id: str, name: str, aum: float, fund_chart: pd.DataFrame, bench_chart: pd.DataFrame) -> Dict:
    if fund_chart.empty or len(fund_chart) < 60:
        return None
    data_days = (pd.to_datetime(fund_chart["timestamp"]).max() - pd.to_datetime(fund_chart["timestamp"]).min()).days
    cagr3 = compute_cagr(fund_chart["nav"], fund_chart["timestamp"])
    cagr5 = compute_cagr(fund_chart["nav"], fund_chart["timestamp"]) if data_days > 365*4 else np.nan

    rf, rb = aligned_weekly_returns(fund_chart, bench_chart)
    rf = winsor(rf.values)
    rb = winsor(rb.values)

    rm = rebound_capture(rf, rb)
    mean_x, stab = sip_consistency(fund_chart)
    rec = recovery_ratio(rf, rb)
    va = vol_adjusted_alpha(rf, rb)
    dr = downside_resilience(rf, rb)
    mp = momentum_persistence(rf)
    da = drawdown_avoidance(rf, rb)
    ea = earnings_acceleration(fund_chart)

    aum = float(aum or 500)
    # Smaller AUM bonus for Small Cap: agility, lower impact cost, better micro-cap access
    if aum < 400:
        aum_factor = 1.07
    elif aum < 1500:
        aum_factor = 1.05
    elif aum < 4000:
        aum_factor = 1.02
    elif aum < 12000:
        aum_factor = 0.99
    else:
        aum_factor = 0.95

    return {
        "mfId": mf_id,
        "name": name,
        "data_days": int(data_days),
        "cagr_3y": round(cagr3 * 100, 2) if not np.isnan(cagr3) else "N/A (insufficient history)",
        "cagr_5y": round(cagr5 * 100, 2) if not np.isnan(cagr5) else "N/A (insufficient history)",
        "rebound_capture": round(rm, 4),
        "sip_stability": round(stab, 4),
        "recovery_ratio": round(rec, 4),
        "vol_alpha": round(va, 4),
        "downside_resilience": round(dr, 4),
        "momentum_persist": round(mp, 4),
        "dd_avoidance": round(da, 4),
        "earnings_accel": round(ea, 4),
        "aum_factor": round(aum_factor, 3)
    }


def main():
    provider = MfDataProvider()

    all_mf = provider.list_all_mf()
    small_ids = []
    for sector, subs in provider.list_mf_by_sector().items():
        for sub, ids in subs.items():
            if "Small" in sub and "Cap" in sub:
                small_ids.extend(ids)

    small_df = all_mf[all_mf["mfId"].isin(small_ids)].copy()
    logger.info(f"Scoring {len(small_df)} Small Cap funds")

    bench_chart = provider.get_index_chart("Small Cap")

    records = []
    for _, row in small_df.iterrows():
        chart = provider.get_mf_chart(row["mfId"])
        rec = score_fund(row["mfId"], row.get("name", row["mfId"]), row.get("aum", 0.0), chart, bench_chart)
        if rec:
            records.append(rec)

    if not records:
        logger.error("No funds scored")
        return

    df = pd.DataFrame(records)

    features = ["rebound_capture", "sip_stability", "recovery_ratio", "vol_alpha",
                "downside_resilience", "momentum_persist", "dd_avoidance", "earnings_accel"]
    # Theory weights: rebound & earnings first (research 2027-28 driver), then SIP consistency & recovery
    weights = np.array([0.18, 0.15, 0.14, 0.13, 0.12, 0.10, 0.10, 0.08])

    z = pd.DataFrame()
    for f in features:
        med = df[f].median()
        mad = (df[f] - med).abs().median() or 1.0
        z[f] = np.clip((df[f] - med) / (mad * 1.4826), -3.5, 3.5)

    raw = (z.values * weights).sum(axis=1)
    score = norm.cdf(raw / 1.15) * 100

    conf = np.where(df["data_days"] >= 900, 0.96, 0.68)
    final = np.clip(score * conf * df["aum_factor"].values, 18, 94)

    df["score"] = np.round(final, 2)
    df["rank"] = df["score"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values("rank")

    cols = ["mfId", "name", "rank", "score", "data_days", "cagr_3y", "cagr_5y"] + features + ["aum_factor"]
    out = df[[c for c in cols if c in df.columns]].copy()

    out_path = RESULTS_DIR / "Small Cap_Grok.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out)} funds to {out_path}")
    logger.info(f"Score range: {df['score'].min():.1f} – {df['score'].max():.1f} (std {df['score'].std():.1f}, unique {df['score'].nunique()})")
    logger.info(f"Top 5: {df.head(5)[['rank','name','score']].to_string(index=False)}")


if __name__ == "__main__":
    main()
