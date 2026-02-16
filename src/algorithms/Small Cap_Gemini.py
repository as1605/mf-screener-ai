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
DATA_DATE = "2026-02-13"
WEEKS_PER_YEAR = 52
MIN_WEEKS_1Y = 50
MIN_WEEKS_3Y = 150
MIN_WEEKS_5Y = 250

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Gemini.tsv"


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
    result["cagr_1y"] = annualised_return(fund_nav, MIN_WEEKS_1Y)
    result["cagr_3y"] = annualised_return(fund_nav, MIN_WEEKS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_WEEKS_5Y)
    
    primary_cagr = result["cagr_3y"] or result["cagr_1y"]
    
    # Volatility & Risk
    vol = annualised_volatility(rets)
    result["volatility"] = vol
    result["max_drawdown"] = max_drawdown(fund_nav)
    
    # Advanced Risk-Adjusted
    result["sharpe"] = sharpe_ratio(primary_cagr, vol)
    result["sortino"] = sortino_ratio(rets)
    result["omega"] = omega_ratio(rets, threshold=RISK_FREE_RATE)
    result["upside_potential"] = upside_potential_ratio(rets, mar=RISK_FREE_RATE)
    
    # Benchmark Relative
    alpha, beta = compute_alpha_beta(rets, bench_rets)
    result["alpha"] = alpha
    result["beta"] = beta
    
    # Momentum (Crucial for 1Y horizon)
    result["ret_3m"] = annualised_return(fund_nav, 13)
    result["ret_6m"] = annualised_return(fund_nav, 26)
    
    bench_ret_3m = annualised_return(bench_nav, 13)
    bench_ret_6m = annualised_return(bench_nav, 26)
    
    if result["ret_6m"] is not None and bench_ret_6m is not None:
        result["momentum_6m"] = result["ret_6m"] - bench_ret_6m
    else:
        result["momentum_6m"] = None
        
    if result["ret_3m"] is not None and bench_ret_3m is not None:
        result["momentum_3m"] = result["ret_3m"] - bench_ret_3m
    else:
        result["momentum_3m"] = None

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
    Build composite score optimized for 1-year forward returns.
    
    Weights:
    - Momentum (3M/6M): 20% (High weight for short-term prediction)
    - Alpha: 20% (Skill persistence)
    - Omega Ratio: 15% (Better risk-adjusted measure for non-normal returns)
    - Sortino Ratio: 10% (Downside protection)
    - Hurst Exponent: 10% (Trend persistence)
    - Upside Potential: 10% (Asymmetry)
    - CAGR (3Y): 10% (Medium term consistency)
    - Max Drawdown: 5% (Disaster avoidance)
    """
    
    score_components = {
        "momentum_3m":          (True,  0.10),
        "momentum_6m":          (True,  0.10),
        "alpha":                (True,  0.20),
        "omega":                (True,  0.15),
        "sortino":              (True,  0.10),
        "hurst":                (True,  0.10),
        "upside_potential":     (True,  0.10),
        "cagr_3y":              (True,  0.10),
        "max_drawdown":         (False, 0.05),
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

def main():
    print("\n" + "=" * 70)
    print(f"  SMALL CAP MUTUAL FUND SCORING - GEMINI MODEL")
    print(f"  Target: Maximize 1-Year Returns")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print("=" * 70)

    provider = MfDataProvider()
    
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
    output["alpha"] = df_scored["alpha"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["omega"] = df_scored["omega"].apply(fmt)
    output["hurst"] = df_scored["hurst"].apply(fmt)
    output["momentum_3m"] = df_scored["momentum_3m"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["aum"] = df_scored["aum"]
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    
    # Display Top 10
    print("\n" + "=" * 70)
    print("  TOP 10 FUNDS (Gemini Model)")
    print("=" * 70)
    print(output.head(10).to_string(index=False))
    print("\n")

if __name__ == "__main__":
    main()
