#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm - Gemini Model

A quantitative scoring model for Indian Small Cap mutual funds, optimized for
predicting performance over a 24-month horizon (12m SIP + 12m hold) using daily data.

Key differentiators from standard models:
1.  **24-Month Scenario Simulation (SIP + Hold)**: Explicitly tests median returns 
    and downside risk for the exact required investment journey.
2.  **Daily Data Resolution**: Leverages 5y daily NAV data for precise beta 
    asymmetry and downside capture profiling against NIFTY Smallcap 250.
3.  **Liquidity / AUM Constraints**: Explicit penalty for funds with > 10,000 Cr AUM, 
    as small cap illiquidity severely drags performance. Optimal AUM is rewarded.
4.  **Asymmetric Downside Risk**: Punishes funds that fall faster than the index 
    on down days.

Author : Gemini
Sector : Small Cap Fund
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR = "Small Cap"
SUBSECTOR = "Small Cap Fund"
BENCHMARK_INDEX = "Small Cap"           # .NISM250
RISK_FREE_RATE = 0.065                  # ~6.5%

TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_6M = 126
MIN_DAYS_1Y = 252
MIN_DAYS_3Y = 756
MIN_DAYS_5Y = 1260

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Gemini.csv"


# ===================================================================
# Advanced Metrics
# ===================================================================

def daily_returns(nav_series: pd.Series) -> pd.Series:
    """Compute simple daily returns."""
    return nav_series.pct_change().dropna()

def annualised_return(nav_series: pd.Series, days: int) -> Optional[float]:
    """CAGR over the last *days*."""
    if len(nav_series) < days + 1:
        return None
    start = nav_series.iloc[-(days + 1)]
    end = nav_series.iloc[-1]
    if start <= 0:
        return None
    years = days / TRADING_DAYS_PER_YEAR
    return (end / start) ** (1 / years) - 1

def max_drawdown(nav_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    return dd.min()

def calculate_capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series, days: int) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Upside and Downside Capture Ratios over the last N days."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < days:
        return None, None
        
    aligned = aligned.iloc[-days:]
    fund_r = aligned.iloc[:, 0]
    bench_r = aligned.iloc[:, 1]
    
    up_days = bench_r > 0
    down_days = bench_r < 0
    
    up_capture = None
    if up_days.sum() > 0:
        bench_up_ret = bench_r[up_days].mean()
        if bench_up_ret > 0:
            up_capture = fund_r[up_days].mean() / bench_up_ret
            
    down_capture = None
    if down_days.sum() > 0:
        bench_down_ret = bench_r[down_days].mean()
        if bench_down_ret < 0:
            down_capture = fund_r[down_days].mean() / bench_down_ret
            
    return up_capture, down_capture

def calculate_asymmetric_beta(fund_ret: pd.Series, bench_ret: pd.Series, days: int = 756) -> Tuple[Optional[float], Optional[float]]:
    """Calculate upside beta and downside beta over the last N days."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < days:
        return None, None
    aligned = aligned.iloc[-days:]
    fund_r = aligned.iloc[:, 0]
    bench_r = aligned.iloc[:, 1]
    
    up_mask = bench_r > 0
    down_mask = bench_r < 0
    
    beta_up = None
    if up_mask.sum() > 10:
        up_cov = np.cov(fund_r[up_mask], bench_r[up_mask])[0, 1]
        up_var = np.var(bench_r[up_mask])
        if up_var > 0:
            beta_up = up_cov / up_var
            
    beta_down = None
    if down_mask.sum() > 10:
        down_cov = np.cov(fund_r[down_mask], bench_r[down_mask])[0, 1]
        down_var = np.var(bench_r[down_mask])
        if down_var > 0:
            beta_down = down_cov / down_var
            
    return beta_up, beta_down

def simulate_24m_sip_hold(nav_series: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    """
    Simulate rolling 24-month investment journey: 12m SIP followed by 12m hold.
    Returns: (median_return, p20_return, number_of_windows)
    """
    # Resample to monthly end dates for the simulation
    monthly_nav = nav_series.resample('ME').last().dropna()
    if len(monthly_nav) < 24:
        return None, None, 0
        
    sim_returns = []
    
    for i in range(len(monthly_nav) - 24):
        # 12 SIP installments
        sip_dates = monthly_nav.index[i : i+12]
        sip_navs = monthly_nav.loc[sip_dates]
        
        # Hold until month 24
        hold_end_date = monthly_nav.index[i+24]
        hold_end_nav = monthly_nav.loc[hold_end_date]
        
        # Calculate return
        total_invested = 12000
        units_accumulated = (1000 / sip_navs).sum()
        final_value = units_accumulated * hold_end_nav
        ret = (final_value - total_invested) / total_invested
        sim_returns.append(ret)
        
    if not sim_returns:
        return None, None, 0
        
    median_ret = float(np.median(sim_returns))
    p20_ret = float(np.percentile(sim_returns, 20))
    return median_ret, p20_ret, len(sim_returns)

def calculate_liquidity_score(aum: float) -> float:
    """
    Score AUM based on Small Cap liquidity constraints.
    Optimal: 1000 - 5000 Cr.
    Penalty: < 500 Cr (too small/survival risk) or > 10000 Cr (severe liquidity drag).
    """
    if pd.isna(aum) or aum <= 0:
        return 0.5
    if aum < 500:
        return 0.4
    if 500 <= aum <= 6000:
        return 1.0
    if 6000 < aum <= 10000:
        return 0.8
    if 10000 < aum <= 15000:
        return 0.5
    if aum > 15000:
        return 0.2
    return 0.5

def information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Information Ratio: Active Return / Tracking Error"""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 252:
        return None
    
    active_return = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if tracking_error == 0:
        return None
        
    ann_active_return = active_return.mean() * TRADING_DAYS_PER_YEAR
    return ann_active_return / tracking_error

# ===================================================================
# Analysis Pipeline
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    name: str,
    aum: float,
) -> dict:
    """Compute all metrics for a single fund."""
    
    n = len(fund_nav)
    result = {"mfId": mf_id, "name": name, "aum": round(aum, 2)}
    
    # Returns
    rets = daily_returns(fund_nav)
    bench_rets = daily_returns(bench_nav)
    
    # Basic Metrics
    result["cagr_3y"] = annualised_return(fund_nav, MIN_DAYS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_DAYS_5Y)
    
    # 24-month Horizon Sim
    median_24m, p20_24m, n_windows = simulate_24m_sip_hold(fund_nav)
    result["sip_hold_median_ret"] = median_24m
    result["sip_hold_p20_ret"] = p20_24m
    result["sim_windows"] = n_windows
    
    # Risk & Asymmetry
    beta_up, beta_down = calculate_asymmetric_beta(rets, bench_rets, days=MIN_DAYS_3Y)
    result["beta_up_3y"] = beta_up
    result["beta_down_3y"] = beta_down
    
    if beta_up is not None and beta_down is not None and beta_down > 0:
        result["asymmetry_score"] = beta_up / beta_down
    else:
        result["asymmetry_score"] = None
        
    result["info_ratio"] = information_ratio(rets, bench_rets)
    
    _, down_cap_1y = calculate_capture_ratios(rets, bench_rets, MIN_DAYS_1Y)
    result["down_cap_1y"] = down_cap_1y
    
    result["max_dd_1y"] = max_drawdown(fund_nav.iloc[-MIN_DAYS_1Y:]) if n >= MIN_DAYS_1Y else None
    
    # Liquidity Profile
    result["liquidity_score"] = calculate_liquidity_score(aum)
    
    # Data Quality
    result["data_days"] = n
    result["has_3y"] = n >= MIN_DAYS_3Y

    return result

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank values to 0-100 percentile."""
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite score optimized for 24-month SIP+Hold in Small Caps.
    """
    
    score_components = {
        "sip_hold_median_ret":  (True,  0.25),  # Consistency of end outcomes
        "sip_hold_p20_ret":     (True,  0.20),  # Downside floor for end outcomes
        "asymmetry_score":      (True,  0.20),  # Upside beta vs downside beta (reward > risk)
        "down_cap_1y":          (False, 0.15),  # Capital protection in recent year
        "info_ratio":           (True,  0.10),  # Active management skill vs index
        "max_dd_1y":            (True,  0.10),  # Max DD (closer to 0 is better, so True because it's negative)
    }
    
    df = df.copy()
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)
    
    for col, (higher_better, weight) in score_components.items():
        if col not in df.columns:
            continue
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        contribution = pctl * weight
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += contribution[mask]
        applied_weight[mask] += weight
        
    df["score"] = np.where(
        applied_weight > 0,
        df["raw_score"] / applied_weight,
        0
    )
    
    # Apply Liquidity Multiplier
    df["score"] = df["score"] * df["liquidity_score"]
    
    # 1. Bull Market Illusion Penalty
    # Heavily penalize unseasoned funds without a 4-year track record
    penalty = pd.Series(1.0, index=df.index)
    has_4y = df["data_days"] >= 1008
    has_3y = df["data_days"] >= 756
    penalty[~has_3y] = 0.30  # 70% penalty if < 3Y (likely just a bull market wonder)
    penalty[has_3y & ~has_4y] = 0.70  # 30% penalty if < 4Y
    df["score"] = df["score"] * penalty
    
    # 2. Debt-Hugger Penalty
    # Prevent selecting defensive funds by ensuring a minimum cumulative equity premium floor.
    # We expect a minimum 24m median return of 24% (roughly 12% annualized) for Small Caps.
    debt_hugger_mask = df["sip_hold_median_ret"] < 0.24
    df.loc[debt_hugger_mask, "score"] = df.loc[debt_hugger_mask, "score"] * 0.20
    
    df["score"] = df["score"].round(2)
    
    return df

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 70)
    print(f"  SMALL CAP MUTUAL FUND SCORING - GEMINI MODEL (Daily 5y)")
    print(f"  Target: 24M Horizon (12m SIP + 12m Hold)")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print("=" * 70)

    provider = MfDataProvider(date=date)
    
    # Load Benchmark (5y daily data)
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"], utc=True)
    bench_df = bench_df.sort_values("timestamp").reset_index(drop=True)
    bench_nav = bench_df.set_index("timestamp")["nav"]
    bench_nav = bench_nav.resample("D").ffill().dropna()
    
    # Load Funds
    df_all = provider.list_all_mf()
    small_cap_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Found {len(small_cap_df)} Small Cap funds")
    
    results = []
    for _, row in small_cap_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = row.get("aum", 0) or 0
        
        try:
            # Upgrade data resolution to '5y'
            chart = provider.get_mf_chart(mf_id, duration='5y')
            if len(chart) < MIN_DAYS_1Y: 
                continue
                
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            chart = chart.sort_values("timestamp").reset_index(drop=True)
            fund_nav = chart.set_index("timestamp")["nav"]
            fund_nav = fund_nav.resample("D").ffill().dropna()
            
            metrics = analyse_fund(mf_id, fund_nav, bench_nav, name, aum)
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error {mf_id}: {e}")
            continue
            
    if not results:
        print("No funds analyzed.")
        return

    df_results = pd.DataFrame(results)
    df_scored = compute_composite_score(df_results)
    
    # Rank
    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min").astype(int)
    df_scored = df_scored.sort_values("rank")
    
    # Format Output
    fmt = lambda v: f"{v:.4f}" if pd.notna(v) else ""
    pct = lambda v: f"{v*100:.2f}" if pd.notna(v) else ""
    
    output = pd.DataFrame()
    output["mfId"] = df_scored["mfId"]
    output["name"] = df_scored["name"]
    output["rank"] = df_scored["rank"]
    output["score"] = df_scored["score"]
    output["data_days"] = df_scored["data_days"]
    output["cagr_3y"] = df_scored["cagr_3y"].apply(pct)
    output["cagr_5y"] = df_scored["cagr_5y"].apply(pct)
    output["sip_hold_median_ret"] = df_scored["sip_hold_median_ret"].apply(pct)
    output["sip_hold_p20_ret"] = df_scored["sip_hold_p20_ret"].apply(pct)
    output["asymmetry_score"] = df_scored["asymmetry_score"].apply(fmt)
    output["beta_up_3y"] = df_scored["beta_up_3y"].apply(fmt)
    output["beta_down_3y"] = df_scored["beta_down_3y"].apply(fmt)
    output["down_cap_1y"] = df_scored["down_cap_1y"].apply(fmt)
    output["max_dd_1y"] = df_scored["max_dd_1y"].apply(pct)
    output["liquidity_score"] = df_scored["liquidity_score"].apply(fmt)
    output["info_ratio"] = df_scored["info_ratio"].apply(fmt)
    output["aum"] = df_scored["aum"]
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    
    # Display Top 10
    print("\n" + "=" * 70)
    print("  TOP 10 FUNDS (Gemini Model)")
    print("=" * 70)
    print(output.head(10).to_string(index=False))
    print("\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Small Cap MF screener (Gemini)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
