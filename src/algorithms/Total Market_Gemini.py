#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Gemini

A custom scoring algorithm for Total Market mutual funds (Flexi Cap, Multi Cap, Value, Contra, Focused)
tailored to a specific 24-month investment horizon: 12 months of SIP followed by a 12-month hold.

Incorporates:
1. Terminal Exposure Risk (p20 rolling XIRR)
2. Asymmetry Score (Up-Beta / Down-Beta)
3. Bull Market Illusion (Confidence Penalty)
4. Debt-Hugger Penalty
5. Manager Edge Decay (IR Decay, Vol Expansion, Downside Decay)
6. Appraisal Ratio
7. Closet Indexer Penalty
8. Non-Linear Decision Tree Scoring

Sector  : Total Market
Author  : Gemini
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR = "Total Market"
SUBSECTORS = [
    "Contra Fund",
    "Flexi Cap Fund",
    "Focused Fund",
    "Multi Cap Fund",
    "Value Fund",
]
BENCHMARK_INDEX = "_NIFTY500"

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Gemini.csv"

# ===================================================================
# Data Cleaning & Basic Utils
# ===================================================================

def clean_nav_to_series(df: pd.DataFrame) -> pd.Series:
    """Convert raw chart DataFrame to a sorted, clean NAV Series."""
    if df.empty:
        return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["timestamp", "nav"])
    out = out[out["nav"] > 0]
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.set_index("timestamp")["nav"]

def align_series(fund_nav: pd.Series, bench_nav: pd.Series, freq='W-FRI'):
    """Align fund and benchmark NAVs, ffill missing daily, then resample to weekly."""
    df = pd.DataFrame({'fund': fund_nav, 'bench': bench_nav})
    df = df.ffill().dropna()
    if df.empty:
        return df
    weekly = df.resample(freq).last().dropna()
    return weekly

# ===================================================================
# XIRR & Simulation
# ===================================================================

def calculate_xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    if len(cashflows) < 2:
        return None
        
    start_date = cashflows[0][0]
    years = [(cf[0] - start_date).days / 365.25 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    
    def npv(r):
        if r <= -1.0: return float('inf')
        return sum(a / ((1 + r) ** y) for a, y in zip(amounts, years))
        
    low = -0.9999
    high = 100.0
    
    for _ in range(100):
        mid = (low + high) / 2.0
        try: npv_mid = npv(mid)
        except: return None
        if abs(npv_mid) < 1e-4: return mid
        if npv(low) * npv_mid < 0: high = mid
        else: low = mid
            
    return (low + high) / 2.0

def sim_24m(nav: pd.Series, start_date: pd.Timestamp) -> Optional[float]:
    end_date = start_date + relativedelta(months=24)
    if nav.index[-1] < end_date: return None
        
    cashflows = []
    total_units = 0.0
    
    for i in range(12):
        td = start_date + relativedelta(months=i)
        av = nav[nav.index >= td]
        if av.empty or (av.index[0] - td).days > 15: return None
        
        actual_date = av.index[0]
        units_bought = 1000.0 / av.iloc[0]
        total_units += units_bought
        cashflows.append((actual_date, -1000.0))
        
    av_end = nav[nav.index <= end_date]
    if av_end.empty or (end_date - av_end.index[-1]).days > 15: return None
        
    exit_date = av_end.index[-1]
    final_value = total_units * av_end.iloc[-1]
    cashflows.append((exit_date, final_value))
    
    return calculate_xirr(cashflows)

# ===================================================================
# Metric Calculators
# ===================================================================

def calc_downside_capture(rets: pd.DataFrame) -> float:
    down_mask = rets['bench'] < 0
    if down_mask.sum() == 0: return np.nan
    fund_down = (1 + rets.loc[down_mask, 'fund']).prod() ** (52 / down_mask.sum()) - 1
    bench_down = (1 + rets.loc[down_mask, 'bench']).prod() ** (52 / down_mask.sum()) - 1
    return fund_down / bench_down if bench_down < 0 else np.nan

def calc_ir(rets: pd.DataFrame) -> float:
    diff = rets['fund'] - rets['bench']
    te = diff.std() * np.sqrt(52)
    alpha = diff.mean() * 52
    return alpha / te if te > 0 else np.nan

def calc_asymmetry(rets: pd.DataFrame) -> float:
    if len(rets) < 10: return np.nan
    up_mask = rets['bench'] > 0
    down_mask = rets['bench'] < 0
    
    if up_mask.sum() < 3 or down_mask.sum() < 3: return np.nan
    
    # Up Beta
    cov_up = np.cov(rets.loc[up_mask, 'fund'], rets.loc[up_mask, 'bench'])[0, 1]
    var_up = np.var(rets.loc[up_mask, 'bench'])
    up_beta = cov_up / var_up if var_up > 0 else np.nan
    
    # Down Beta
    cov_down = np.cov(rets.loc[down_mask, 'fund'], rets.loc[down_mask, 'bench'])[0, 1]
    var_down = np.var(rets.loc[down_mask, 'bench'])
    down_beta = cov_down / var_down if var_down > 0 else np.nan
    
    if pd.notna(up_beta) and pd.notna(down_beta) and down_beta > 0:
        return up_beta / down_beta
    return np.nan

def calc_appraisal_and_r2(rets: pd.DataFrame) -> Tuple[float, float]:
    if len(rets) < 10: return np.nan, np.nan
    
    X = rets['bench'].values
    y = rets['fund'].values
    
    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    
    alpha_weekly = intercept
    r_squared = r_value ** 2
    
    # Calculate residuals
    residuals = y - (slope * X + intercept)
    resid_std = np.std(residuals)
    
    alpha_annual = alpha_weekly * 52
    idio_risk_annual = resid_std * np.sqrt(52)
    
    appraisal = alpha_annual / idio_risk_annual if idio_risk_annual > 0 else np.nan
    return appraisal, r_squared

# ===================================================================
# Main Analysis Function
# ===================================================================

def analyse_fund(
    mf_id: str, fund_nav: pd.Series, bench_nav: pd.Series, name: str, aum: float, subsector: str
) -> dict:
    
    res = {
        "mfId": mf_id, "name": name, "aum": round(aum, 2), "subsector": subsector,
        "data_days": len(fund_nav), "valid": False
    }

    if len(fund_nav) < 252: # Need at least 1 year
        return res

    # 1. Rolling 24m XIRR & Terminal Exposure Risk
    st = fund_nav.index[0].replace(day=1) + relativedelta(months=1)
    xirrs = []
    while st + relativedelta(months=24) <= fund_nav.index[-1]:
        x = sim_24m(fund_nav, st)
        if x is not None: xirrs.append(x)
        st += relativedelta(months=1)
        
    res["mean_xirr"] = np.mean(xirrs) if xirrs else np.nan
    res["p20_xirr"] = np.percentile(xirrs, 20) if xirrs else np.nan
    
    # 2. Daily Volatility (6m vs 3y)
    daily_rets = fund_nav.pct_change().dropna()
    def get_vol(period_days):
        end = daily_rets.index[-1]
        start = end - timedelta(days=period_days)
        period_rets = daily_rets[start:]
        return period_rets.std() * np.sqrt(252) if len(period_rets) > 10 else np.nan
        
    res["vol_6m"] = get_vol(180)
    res["vol_3y"] = get_vol(1095)
    
    # 3. Weekly relative metrics
    weekly = align_series(fund_nav, bench_nav)
    if len(weekly) < 10: return res
    
    rets = weekly.pct_change().dropna()
    end_dt = rets.index[-1]
    
    mask_6m = rets.index >= (end_dt - timedelta(days=180))
    mask_1y = rets.index >= (end_dt - timedelta(days=365))
    mask_3y = rets.index >= (end_dt - timedelta(days=1095))
    
    rets_6m = rets[mask_6m]
    rets_1y = rets[mask_1y]
    rets_3y = rets[mask_3y]
    
    res["ir_6m"] = calc_ir(rets_6m) if len(rets_6m) > 10 else np.nan
    res["ir_3y"] = calc_ir(rets_3y) if len(rets_3y) > 10 else np.nan
    
    res["d_cap_6m"] = calc_downside_capture(rets_6m) if len(rets_6m) > 10 else np.nan
    res["d_cap_3y"] = calc_downside_capture(rets_3y) if len(rets_3y) > 10 else np.nan
    
    res["asymmetry"] = calc_asymmetry(rets)
    
    app, r2 = calc_appraisal_and_r2(rets_1y)
    res["appraisal_1y"] = app
    res["r2_1y"] = r2
    
    # 4. Basic CAGR
    days = (fund_nav.index[-1] - fund_nav.index[0]).days
    res["cagr_3y"] = ((fund_nav.iloc[-1] / fund_nav[fund_nav.index >= (fund_nav.index[-1] - timedelta(days=1095))].iloc[0]) ** (365.25 / 1095) - 1) if days >= 1095 else np.nan
    res["cagr_5y"] = ((fund_nav.iloc[-1] / fund_nav[fund_nav.index >= (fund_nav.index[-1] - timedelta(days=1826))].iloc[0]) ** (365.25 / 1826) - 1) if days >= 1826 else np.nan

    res["valid"] = True
    return res

# ===================================================================
# Non-Linear Scoring
# ===================================================================

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Hard Filters & Penalties initialization
    df["penalty_mult"] = 1.0
    df["drop_flag"] = False
    
    # Stage 1: Hard Filters
    # Closet Indexer Penalty
    df.loc[df["r2_1y"] > 0.95, "penalty_mult"] *= 0.5
    
    # Debt-Hugger Penalty
    # If Mean XIRR < 8% (risk free 6.5% + marginal equity premium)
    df.loc[df["mean_xirr"] < 0.08, "penalty_mult"] *= 0.5
    
    # Bull Market Illusion / Confidence Penalty (< 4 years data)
    df.loc[df["data_days"] < 1000, "penalty_mult"] *= 0.7  # < ~4 years daily data
    
    # Stage 2: Edge Decay Multipliers
    # IR Decay
    decay_ir_mask = (df["ir_6m"] < df["ir_3y"] * 0.8) & (df["ir_3y"] > 0)
    df.loc[decay_ir_mask, "penalty_mult"] *= 0.8
    
    # Volatility Expansion
    decay_vol_mask = (df["vol_6m"] > df["vol_3y"] * 1.2) & (df["vol_3y"] > 0)
    df.loc[decay_vol_mask, "penalty_mult"] *= 0.8
    
    # Downside Capture Decay
    decay_dcap_mask = (df["d_cap_6m"] > df["d_cap_3y"] * 1.2) & (df["d_cap_3y"] > 0)
    df.loc[decay_dcap_mask, "penalty_mult"] *= 0.8
    
    # Base Rank Scoring (using non-linear percentiles)
    # 1. Terminal Exposure Risk (p20_xirr) -> 40% weight
    # 2. Mean XIRR -> 30% weight
    # 3. Asymmetry Score -> 30% weight
    
    score_p20 = df["p20_xirr"].rank(pct=True) * 40
    score_mean = df["mean_xirr"].rank(pct=True) * 30
    score_asym = df["asymmetry"].rank(pct=True) * 30
    
    df["base_score"] = score_p20.fillna(0) + score_mean.fillna(0) + score_asym.fillna(0)
    
    # Stage 3: Alpha & Asymmetry Boosts
    # Top quartile Appraisal Ratio gets +15 points
    app_75 = df["appraisal_1y"].quantile(0.75)
    if pd.notna(app_75):
        df.loc[df["appraisal_1y"] > app_75, "base_score"] += 15
        
    # Top quartile Asymmetry gets +15 points
    asym_75 = df["asymmetry"].quantile(0.75)
    if pd.notna(asym_75):
        df.loc[df["asymmetry"] > asym_75, "base_score"] += 15
        
    # Final Score
    df["score"] = (df["base_score"] * df["penalty_mult"]).round(2)
    
    return df

# ===================================================================
# Formatting Helpers
# ===================================================================

def _pct(v): return f"{v*100:.2f}" if pd.notna(v) else ""
def _num(v): return f"{v:.2f}" if pd.notna(v) else ""

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 80)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM — GEMINI")
    print(f"  Benchmark : {BENCHMARK_INDEX}")
    print("=" * 80)

    provider = MfDataProvider(date=date)

    logger.info("Loading Index Data...")
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_to_series(bench_df)

    df_all = provider.list_all_mf()
    sector_df = df_all[df_all["subsector"].isin(SUBSECTORS)].copy()
    print(f"  Total Funds in Sector: {len(sector_df)}")

    results = []
    logger.info("Analyzing funds...")
    
    for idx, row in sector_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        
        try:
            chart = provider.get_mf_chart(mf_id, duration="5y")
            fund_nav = clean_nav_to_series(chart)
            
            res = analyse_fund(mf_id, fund_nav, bench_nav, name, row.get("aum", 0), row["subsector"])
            if res.get("valid"):
                results.append(res)
            
        except Exception as e:
            logger.error(f"Error analyzing {mf_id}: {e}")

    if not results:
        print("No results generated.")
        return

    df_results = pd.DataFrame(results)
    df_scored = compute_composite_score(df_results)

    out_cols = [
        "mfId", "name", "rank", "score", "data_days", "subsector", "aum",
        "cagr_3y", "cagr_5y", "mean_xirr", "p20_xirr",
        "asymmetry", "appraisal_1y", "r2_1y",
        "ir_3y", "ir_6m", "vol_3y", "vol_6m", "d_cap_3y", "d_cap_6m",
        "penalty_mult"
    ]
    for col in out_cols:
        if col not in df_scored.columns:
            df_scored[col] = np.nan
            
    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min")
    df_scored = df_scored.sort_values("rank")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_df = df_scored[out_cols].copy()
    
    pct_cols = ["cagr_3y", "cagr_5y", "mean_xirr", "p20_xirr", "vol_3y", "vol_6m"]
    num_cols = ["asymmetry", "appraisal_1y", "r2_1y", "ir_3y", "ir_6m", "d_cap_3y", "d_cap_6m", "penalty_mult", "score"]
    
    for col in pct_cols:
        export_df[col] = export_df[col].apply(_pct)
    for col in num_cols:
        export_df[col] = export_df[col].apply(_num)
        
    export_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\nTop 20 Funds:")
    print(export_df[["name", "rank", "score", "mean_xirr", "p20_xirr", "asymmetry", "penalty_mult"]].head(20).to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Total Market MF screener (Gemini)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    main(date=p.parse_args().date)
