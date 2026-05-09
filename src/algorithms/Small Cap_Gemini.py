#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm - Gemini Model

A quantitative scoring model for Indian Small Cap mutual funds, optimized for
predicting performance over the next 1 year.

Key differentiators from standard models:
1.  **Omega Ratio**: Captures non-normal return distributions common in small caps.
2.  **Hurst Exponent**: Identifies funds with persistent trending behavior (momentum).
3.  **Upside Potential Ratio**: Focuses on the asymmetry of returns (we want upside > downside).
4.  **Momentum Bias**: Higher weight on recent performance (3M/6M) for 1-year horizon.
5.  **Volatility Regime**: Penalizes funds with increasing recent volatility.

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
WEEKS_PER_YEAR = 52
MIN_WEEKS_1Y = 50
MIN_WEEKS_3Y = 150
MIN_WEEKS_5Y = 250

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Gemini.csv"


# ===================================================================
# Advanced Metrics
# ===================================================================

def weekly_returns(nav_series: pd.Series) -> pd.Series:
    """Compute simple weekly returns."""
    return nav_series.pct_change().dropna()

def annualised_return(nav_series: pd.Series, weeks: int) -> Optional[float]:
    """CAGR over the last *weeks*."""
    if len(nav_series) < weeks + 1:
        return None
    start = nav_series.iloc[-(weeks + 1)]
    end = nav_series.iloc[-1]
    if start <= 0:
        return None
    years = weeks / WEEKS_PER_YEAR
    return (end / start) ** (1 / years) - 1

def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation."""
    return returns.std() * np.sqrt(WEEKS_PER_YEAR)

def max_drawdown(nav_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    return dd.min()

def sharpe_ratio(cagr: float, vol: float, rf: float = RISK_FREE_RATE) -> Optional[float]:
    if vol == 0 or vol is None or cagr is None:
        return None
    return (cagr - rf) / vol

def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> Optional[float]:
    """Sortino ratio calculated from weekly returns series."""
    if len(returns) < 20:
        return None
    
    # Convert annual RF to weekly
    rf_weekly = (1 + rf) ** (1/WEEKS_PER_YEAR) - 1
    
    excess_returns = returns - rf_weekly
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return None
        
    downside_dev = np.sqrt((downside_returns**2).mean()) * np.sqrt(WEEKS_PER_YEAR)
    
    if downside_dev == 0:
        return None
        
    # Annualize mean excess return
    mean_excess = excess_returns.mean() * WEEKS_PER_YEAR
    return mean_excess / downside_dev

def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> Optional[float]:
    """
    Omega Ratio: Probability weighted ratio of gains vs losses for a threshold return.
    Captures all higher moments (skewness, kurtosis).
    """
    if len(returns) < 20:
        return None
        
    # Convert annual threshold to weekly
    thresh_weekly = (1 + threshold) ** (1/WEEKS_PER_YEAR) - 1
    
    excess = returns - thresh_weekly
    positive = excess[excess > 0].sum()
    negative = abs(excess[excess < 0].sum())
    
    if negative == 0:
        return 10.0 # Cap at a high value
        
    return positive / negative

def upside_potential_ratio(returns: pd.Series, mar: float = 0.0) -> Optional[float]:
    """Upside Potential Ratio: Upside deviation / Downside deviation."""
    if len(returns) < 20:
        return None
        
    mar_weekly = (1 + mar) ** (1/WEEKS_PER_YEAR) - 1
    excess = returns - mar_weekly
    
    upside = excess[excess > 0]
    downside = excess[excess < 0]
    
    if len(downside) == 0:
        return 5.0 # Cap
        
    up_dev = np.sqrt((upside**2).mean())
    down_dev = np.sqrt((downside**2).mean())
    
    if down_dev == 0:
        return 5.0
        
    return up_dev / down_dev

def hurst_exponent(time_series: pd.Series) -> Optional[float]:
    """
    Hurst Exponent to measure long-term memory of time series.
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending (Persistent)
    """
    try:
        lags = range(2, 20)
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return None

def compute_alpha_beta(fund_ret: pd.Series, bench_ret: pd.Series):
    """Jensen's alpha and beta."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None, None
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    
    cov = np.cov(x, y)[0, 1]
    var_x = np.var(x, ddof=1)
    
    beta = cov / var_x if var_x > 0 else 0
    
    # Annualized Alpha
    rf_weekly = (1 + RISK_FREE_RATE) ** (1/WEEKS_PER_YEAR) - 1
    alpha_weekly = np.mean(y) - rf_weekly - beta * (np.mean(x) - rf_weekly)
    alpha = alpha_weekly * WEEKS_PER_YEAR
    
    return alpha, beta

def information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Information Ratio: Active Return / Tracking Error"""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None
    
    active_return = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_return.std() * np.sqrt(WEEKS_PER_YEAR)
    
    if tracking_error == 0:
        return None
        
    ann_active_return = active_return.mean() * WEEKS_PER_YEAR
    return ann_active_return / tracking_error

def volatility_contraction(returns: pd.Series) -> Optional[float]:
    """Ratio of short-term volatility (3M) to long-term volatility (1Y). Lower is better."""
    if len(returns) < WEEKS_PER_YEAR:
        return None
        
    vol_1y = returns.iloc[-WEEKS_PER_YEAR:].std() * np.sqrt(WEEKS_PER_YEAR)
    vol_3m = returns.iloc[-13:].std() * np.sqrt(WEEKS_PER_YEAR)
    
    if vol_1y == 0:
        return None
        
    return vol_3m / vol_1y

def calculate_capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series, weeks: int) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Upside and Downside Capture Ratios over the last N weeks."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < weeks:
        return None, None
        
    aligned = aligned.iloc[-weeks:]
    fund_r = aligned.iloc[:, 0]
    bench_r = aligned.iloc[:, 1]
    
    up_months = bench_r > 0
    down_months = bench_r < 0
    
    up_capture = None
    if up_months.sum() > 0:
        bench_up_ret = bench_r[up_months].mean()
        if bench_up_ret > 0:
            up_capture = fund_r[up_months].mean() / bench_up_ret
            
    down_capture = None
    if down_months.sum() > 0:
        bench_down_ret = bench_r[down_months].mean()
        if bench_down_ret < 0:
            down_capture = fund_r[down_months].mean() / bench_down_ret
            
    return up_capture, down_capture

def calculate_rolling_sip_alpha(fund_nav: pd.Series, bench_nav: pd.Series, months: int = 12) -> pd.Series:
    """Calculate rolling 1Y SIP alpha (fund SIP return - benchmark SIP return)."""
    # Resample to monthly
    fund_monthly = fund_nav.resample('MS').first()
    bench_monthly = bench_nav.resample('MS').first()
    
    aligned = pd.concat([fund_monthly, bench_monthly], axis=1, join="inner").dropna()
    if len(aligned) < months + 1:
        return pd.Series(dtype=float)
        
    alphas = {}
    for i in range(len(aligned) - months):
        f_invest = aligned.iloc[i : i+months, 0]
        f_final = aligned.iloc[i+months, 0]
        
        b_invest = aligned.iloc[i : i+months, 1]
        b_final = aligned.iloc[i+months, 1]
        
        # Fund SIP Return
        f_units = 1000 / f_invest
        f_val = f_units.sum() * f_final
        f_ret = (f_val - (1000 * months)) / (1000 * months)
        
        # Benchmark SIP Return
        b_units = 1000 / b_invest
        b_val = b_units.sum() * b_final
        b_ret = (b_val - (1000 * months)) / (1000 * months)
        
        alphas[aligned.index[i+months]] = f_ret - b_ret
        
    return pd.Series(alphas)

def calculate_trend_slope(series: pd.Series) -> Optional[float]:
    """Calculate the linear trend slope of a series."""
    if len(series) < 3:
        return None
    x = np.arange(len(series))
    y = series.values
    slope, _ = np.polyfit(x, y, 1)
    return slope

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
    rets = weekly_returns(fund_nav)
    bench_rets = weekly_returns(bench_nav)
    
    # Basic Metrics
    result["cagr_3y"] = annualised_return(fund_nav, MIN_WEEKS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_WEEKS_5Y)
    
    # Advanced Risk-Adjusted
    result["omega"] = omega_ratio(rets, threshold=RISK_FREE_RATE)
    result["info_ratio"] = information_ratio(rets, bench_rets)
    result["vol_contraction"] = volatility_contraction(rets)
    
    # Capture Ratios (6M Upside, 1Y Downside)
    up_cap_6m, _ = calculate_capture_ratios(rets, bench_rets, 26)
    _, down_cap_1y = calculate_capture_ratios(rets, bench_rets, 52)
    result["up_cap_6m"] = up_cap_6m
    result["down_cap_1y"] = down_cap_1y
    
    # Rolling SIP Alpha Trend
    rolling_alphas = calculate_rolling_sip_alpha(fund_nav, bench_nav, months=12)
    if len(rolling_alphas) >= 12:
        # Trend over the last 12 rolling periods (approx 1 year of rolling data)
        recent_alphas = rolling_alphas.iloc[-12:]
        result["sip_alpha_trend"] = calculate_trend_slope(recent_alphas)
    else:
        result["sip_alpha_trend"] = None
    
    # Trend Persistence
    result["hurst"] = hurst_exponent(fund_nav.values)
    
    # Data Quality
    result["data_weeks"] = n
    result["data_days"] = (fund_nav.index.max() - fund_nav.index.min()).days + 1
    result["has_3y"] = n >= MIN_WEEKS_3Y

    return result

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank values to 0-100 percentile."""
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite score optimized for 1-year forward returns based on predictive metrics.
    
    Weights:
    - Information Ratio: 25% (Consistency of manager skill)
    - Rolling SIP Alpha Trend: 20% (Momentum of manager skill)
    - Recent Upside Capture (6M): 15% (Participation in rebound)
    - 1Y Downside Capture: 15% (Capital protection during consolidation - Lower is better)
    - Omega Ratio: 15% (Overall risk-adjusted return profile)
    - Volatility Contraction: 10% (Stabilization indicator - Lower is better)
    """
    
    score_components = {
        "info_ratio":           (True,  0.25),
        "sip_alpha_trend":      (True,  0.20),
        "up_cap_6m":            (True,  0.15),
        "down_cap_1y":          (False, 0.15),
        "omega":                (True,  0.15),
        "vol_contraction":      (False, 0.10),
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
    
    # Penalty for short history
    penalty = pd.Series(1.0, index=df.index)
    short_track = ~df["has_3y"]
    penalty[short_track] = 0.90 # 10% penalty
    df["score"] = df["score"] * penalty
    
    df["score"] = df["score"].round(2)
    
    return df

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 70)
    print(f"  SMALL CAP MUTUAL FUND SCORING - GEMINI MODEL")
    print(f"  Target: Maximize 1-Year Returns")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print("=" * 70)

    provider = MfDataProvider(date=date)
    
    # Load Benchmark
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"], utc=True)
    bench_df = bench_df.sort_values("timestamp").reset_index(drop=True)
    bench_nav = bench_df.set_index("timestamp")["nav"]
    
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
            chart = provider.get_mf_chart(mf_id)
            if len(chart) < 50: # Need at least ~1 year data
                continue
                
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            chart = chart.sort_values("timestamp").reset_index(drop=True)
            fund_nav = chart.set_index("timestamp")["nav"]
            
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
    fmt = lambda v: f"{v:.2f}" if pd.notna(v) else ""
    
    output = pd.DataFrame()
    output["mfId"] = df_scored["mfId"]
    output["name"] = df_scored["name"]
    output["rank"] = df_scored["rank"]
    output["score"] = df_scored["score"]
    output["data_days"] = df_scored["data_days"]
    output["cagr_3y"] = df_scored["cagr_3y"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["cagr_5y"] = df_scored["cagr_5y"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["info_ratio"] = df_scored["info_ratio"].apply(fmt)
    output["sip_alpha_trend"] = df_scored["sip_alpha_trend"].apply(fmt)
    output["up_cap_6m"] = df_scored["up_cap_6m"].apply(fmt)
    output["down_cap_1y"] = df_scored["down_cap_1y"].apply(fmt)
    output["vol_contraction"] = df_scored["vol_contraction"].apply(fmt)
    output["omega"] = df_scored["omega"].apply(fmt)
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
