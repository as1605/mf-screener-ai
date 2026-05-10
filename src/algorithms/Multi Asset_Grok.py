#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Grok Multi-Asset Allocation Scoring Engine - Cycle Capture & Risk Balance
========================================================================

Original design for predicting 1-year forward monthly SIP performance in
Multi Asset Allocation funds. Emphasizes gold/silver rally capture,
equity participation with downside control, inferred allocation stability,
and resilience across commodity-equity regimes.

Uses weekly NAV alignment vs M_SBIGL (gold), M_ICPVF (silver, post-2022),
_NIFTY500. 9 custom features with robust median/MAD z-scores, theory-driven
weights favoring strategic cycle advantage over tactical noise.

Target: sensible spread, unique ranks, top funds = strong commodity beta
in upswings + equity rebound skill + low pain.

Run: python src/algorithms/Multi\ Asset_Grok.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GrokMultiAsset")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider


# ---------------------------------------------------------------------------
# Core Helpers (adapted for multi-benchmark)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 9 Original Multi-Asset Features (theory: cycle capture + stability)
# ---------------------------------------------------------------------------
def winsor(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [p*100, (1-p)*100])
    return np.clip(x, lo, hi)


def gold_up_excess(rf: np.ndarray, rg: np.ndarray) -> float:
    """Excess return on weeks gold rises (capture commodity rallies)."""
    mask = rg > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(rf[mask] - rg[mask]))


def equity_up_excess(rf: np.ndarray, re: np.ndarray) -> float:
    """Excess on equity up weeks (participation without lag)."""
    mask = re > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(rf[mask] - re[mask]))


def silver_beta_recent(rf: np.ndarray, rs: np.ndarray, min_weeks: int = 26) -> float:
    """Beta to silver in recent overlapping window (post-2022 data). Lower threshold because silver series is shorter."""
    if len(rf) < min_weeks or len(rs) < min_weeks:
        return 0.0
    # use last min_weeks
    n = min(len(rf), len(rs), min_weeks)
    rf_r, rs_r = rf[-n:], rs[-n:]
    if np.std(rs_r) < 1e-8:
        return 0.0
    cov = np.cov(rf_r, rs_r)[0, 1]
    var = np.var(rs_r)
    return float(cov / var) if var > 0 else 0.0


def downside_vs_equity(rf: np.ndarray, re: np.ndarray) -> float:
    """Downside capture vs equity (<1 = protection). Lower better."""
    mask = re < 0
    if not np.any(mask) or np.mean(re[mask]) == 0:
        return 1.0
    cap = np.mean(rf[mask]) / np.mean(re[mask])
    return float(np.clip(cap, 0.2, 2.0))


def rebound_elasticity_multi(rf: np.ndarray, re: np.ndarray, rg: np.ndarray, dd_thresh: float = 0.08, weeks: int = 8) -> float:
    """Median excess rebound after equity or gold drawdowns."""
    if len(rf) < weeks + 20:
        return 0.0
    cum_e = np.cumprod(1 + re) - 1
    peak_e = np.maximum.accumulate(cum_e)
    dd_e = (cum_e - peak_e) / (peak_e + 1e-9)
    rebounds = []
    for i in range(len(dd_e) - weeks):
        if dd_e[i] <= -dd_thresh:
            ex = np.sum(rf[i+1:i+1+weeks]) - np.sum(re[i+1:i+1+weeks])
            rebounds.append(ex / (-dd_e[i] + 1e-9))
    if rebounds:
        return float(np.median(rebounds))
    # fallback to gold dd
    cum_g = np.cumprod(1 + rg) - 1
    peak_g = np.maximum.accumulate(cum_g)
    dd_g = (cum_g - peak_g) / (peak_g + 1e-9)
    for i in range(len(dd_g) - weeks):
        if dd_g[i] <= -dd_thresh:
            ex = np.sum(rf[i+1:i+1+weeks]) - np.sum(rg[i+1:i+1+weeks])
            rebounds.append(ex / (-dd_g[i] + 1e-9))
    return float(np.median(rebounds)) if rebounds else 0.0


def max_drawdown_pain(nav_w: pd.Series) -> float:
    """Pain index: average depth of drawdowns (lower better)."""
    if len(nav_w) < 10:
        return 0.0
    cum = nav_w.values / nav_w.values[0] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    avg_pain = -np.mean(dd[dd < 0]) if np.any(dd < 0) else 0.0
    return float(np.clip(avg_pain, 0, 0.5))


def allocation_stability(rf: np.ndarray, re: np.ndarray, rg: np.ndarray, window: int = 26) -> float:
    """Low variance in rolling equity beta (stable strategic mix). Lower var better."""
    if len(rf) < window * 2:
        return 1.0
    betas = []
    for i in range(window, len(rf)):
        rfw = rf[i-window:i]
        rew = re[i-window:i]
        if np.std(rew) > 1e-8:
            b = np.cov(rfw, rew)[0, 1] / np.var(rew)
            betas.append(b)
    if len(betas) < 3:
        return 1.0
    return float(np.std(betas))


def tactical_noise_penalty(rf: np.ndarray, re: np.ndarray, rg: np.ndarray) -> float:
    """Penalty for erratic short-term deviations (high freq noise)."""
    if len(rf) < 30:
        return 0.0
    # simple: std of residual after 4w MA
    rf_ma = pd.Series(rf).rolling(4, min_periods=1).mean().values
    resid_std = np.std(rf - rf_ma)
    return float(np.clip(resid_std, 0, 0.05))


def regime_persistence(rf: np.ndarray, re: np.ndarray, rg: np.ndarray) -> float:
    """Consistency of outperformance across bull/bear commodity and equity regimes."""
    if len(rf) < 52:
        return 0.0
    # split into 4 rough regimes by signs of re, rg
    scores = []
    for sign_e, sign_g in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        mask = ((re > 0) == (sign_e > 0)) & ((rg > 0) == (sign_g > 0))
        if np.sum(mask) > 8:
            ex = np.mean(rf[mask] - 0.5*re[mask] - 0.3*rg[mask])  # rough blended
            scores.append(ex)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Scoring per fund
# ---------------------------------------------------------------------------
def score_fund(mf_id: str, name: str, aum: float,
               fund_chart: pd.DataFrame,
               gold_chart: pd.DataFrame,
               silver_chart: pd.DataFrame,
               nifty_chart: pd.DataFrame) -> Dict:
    rec: Dict = {
        "mfId": mf_id,
        "name": name,
        "aum": aum,
        "data_days": len(fund_chart) if not fund_chart.empty else 0,
    }

    if fund_chart.empty or len(fund_chart) < 50:
        for k in ["cagr_3y", "cagr_5y", "gold_up_excess", "equity_up_excess", "silver_beta_recent",
                  "downside_vs_equity", "rebound_elasticity_multi", "max_drawdown_pain",
                  "allocation_stability", "tactical_noise_penalty", "regime_persistence", "aum_penalty"]:
            rec[k] = np.nan
        return rec

    # CAGRs
    rec["cagr_3y"] = compute_cagr(fund_chart["nav"], fund_chart["timestamp"])
    rec["cagr_5y"] = compute_cagr(fund_chart["nav"], fund_chart["timestamp"])

    # Align returns vs 3 benchmarks
    rf, rg = aligned_weekly_returns(fund_chart, gold_chart)
    rf_e, re = aligned_weekly_returns(fund_chart, nifty_chart)
    # silver may be shorter
    rf_s, rs = aligned_weekly_returns(fund_chart, silver_chart) if not silver_chart.empty else (pd.Series(dtype=float), pd.Series(dtype=float))

    # ensure same length for rf by intersection
    common_idx = rf.index.intersection(rf_e.index)
    if len(common_idx) < 20:
        for k in ["gold_up_excess", "equity_up_excess", "silver_beta_recent",
                  "downside_vs_equity", "rebound_elasticity_multi", "max_drawdown_pain",
                  "allocation_stability", "tactical_noise_penalty", "regime_persistence"]:
            rec[k] = 0.0
        rec["aum_penalty"] = 0.7 if aum < 100 else 1.0
        return rec

    rf = rf.loc[common_idx].values
    rg = rg.loc[common_idx].values
    re = re.loc[common_idx].values
    if len(rs) > 0:
        rs_inter = rs.loc[rs.index.intersection(common_idx)]
        rs = rs_inter.values if len(rs_inter) >= 26 else np.zeros_like(rf)
    else:
        rs = np.zeros_like(rf)

    # weekly nav for pain
    fund_w = to_weekly(fund_chart)

    # 9 features
    rec["gold_up_excess"] = gold_up_excess(rf, rg)
    rec["equity_up_excess"] = equity_up_excess(rf, re)
    rec["silver_beta_recent"] = silver_beta_recent(rf, rs)
    rec["downside_vs_equity"] = downside_vs_equity(rf, re)
    rec["rebound_elasticity_multi"] = rebound_elasticity_multi(rf, re, rg)
    rec["max_drawdown_pain"] = max_drawdown_pain(fund_w["nav"])
    rec["allocation_stability"] = allocation_stability(rf, re, rg)
    rec["tactical_noise_penalty"] = tactical_noise_penalty(rf, re, rg)
    rec["regime_persistence"] = regime_persistence(rf, re, rg)

    # AUM penalty (small funds less reliable)
    rec["aum_penalty"] = 0.75 if aum < 200 else (0.9 if aum < 1000 else 1.0)

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    provider = MfDataProvider()

    all_mf = provider.list_all_mf()
    multi_ids = []
    for sector, subs in provider.list_mf_by_sector().items():
        for sub, ids in subs.items():
            if "Multi Asset" in sub:
                multi_ids.extend(ids)

    multi_df = all_mf[all_mf["mfId"].isin(multi_ids)].copy()
    logger.info(f"Scoring {len(multi_df)} Multi Asset Allocation funds")

    gold_chart = provider.get_index_chart("GBES")
    silver_chart = provider.get_mf_chart("M_ICPVF")  # silver ETF as MF (post-2022)
    nifty_chart = provider.get_index_chart(".NIFTY500")

    records = []
    for _, row in multi_df.iterrows():
        chart = provider.get_mf_chart(row["mfId"])
        if chart.empty or len(chart) < 50:
            continue
        rec = score_fund(row["mfId"], row.get("name", row["mfId"]), row.get("aum", 0.0),
                         chart, gold_chart, silver_chart, nifty_chart)
        records.append(rec)

    if not records:
        logger.error("No funds scored")
        return

    df = pd.DataFrame(records)

    # Features and weights (positive = higher better, negative = lower better)
    features = ["gold_up_excess", "equity_up_excess", "silver_beta_recent",
                "downside_vs_equity", "rebound_elasticity_multi", "max_drawdown_pain",
                "allocation_stability", "tactical_noise_penalty", "regime_persistence"]
    # negative weights for risk/tactical noise metrics
    weights = np.array([0.18, 0.15, 0.10, -0.14, 0.13, -0.10, -0.09, -0.08, 0.12])

    z = pd.DataFrame()
    for f in features:
        med = df[f].median()
        mad = (df[f] - med).abs().median() or 1.0
        z[f] = np.clip((df[f] - med) / (mad * 1.4826), -3, 3)

    raw = (z.values * weights).sum(axis=1)
    score = norm.cdf(raw / 1.15) * 100

    conf = np.where(df["data_days"] >= 365, 0.95, 0.70)
    final = np.clip(score * conf * df["aum_penalty"].values, 20, 90)

    df["score"] = np.round(final, 2)
    df["rank"] = df["score"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values("rank")

    cols = ["mfId", "name", "rank", "score", "data_days", "cagr_3y", "cagr_5y"] + features + ["aum_penalty"]
    out = df[[c for c in cols if c in df.columns]].copy()

    out_path = RESULTS_DIR / "Multi Asset_Grok.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out)} funds to {out_path}")
    logger.info(f"Score range: {df['score'].min():.1f} – {df['score'].max():.1f} (std {df['score'].std():.1f}, unique {df['score'].nunique()})")
    logger.info(f"Top 5: {df.head(5)[['rank','name','score']].to_string(index=False)}")


if __name__ == "__main__":
    main()
