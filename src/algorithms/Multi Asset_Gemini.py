#!/usr/bin/env python3
"""
Multi Asset Allocation Fund Scoring Algorithm - Gemini Model

A quantitative scoring model for Indian Multi Asset Allocation mutual funds,
optimized for predicting performance over the next 1 year.

Key differentiators:
1. Cycle Agility Score: Measures how well the fund increases equity exposure during
   bull markets and reduces it during bear markets (Equity Up-Beta - Equity Down-Beta).
2. Precious Metals Upside Capture: Measures the fund's ability to capture rallies
   in Gold and Silver.
3. Sortino Ratio (3Y): Focuses purely on downside risk.
4. SIP Return Stability: Evaluates the consistency of 1-year SIP returns over time.

Author : Gemini
Sector : Multi Asset
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

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
SECTOR = "Multi Asset"
SUBSECTOR = "Multi Asset Allocation Fund"
EQUITY_INDEX = "Total Market"           # .NIFTY500
GOLD_MF_ID = "M_SBIGL"
SILVER_MF_ID = "M_ICPVF"
RISK_FREE_RATE = 0.065                  # ~6.5%
TRADING_DAYS_PER_YEAR = 252
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

def compute_up_down_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Up-Market and Down-Market Beta."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None, None
        
    f_r = aligned.iloc[:, 0]
    b_r = aligned.iloc[:, 1]
    
    up_days = b_r > 0
    down_days = b_r < 0
    
    up_beta = None
    if up_days.sum() > 5:
        cov = np.cov(b_r[up_days], f_r[up_days])[0, 1]
        var = np.var(b_r[up_days], ddof=1)
        if var > 0:
            up_beta = cov / var
            
    down_beta = None
    if down_days.sum() > 5:
        cov = np.cov(b_r[down_days], f_r[down_days])[0, 1]
        var = np.var(b_r[down_days], ddof=1)
        if var > 0:
            down_beta = cov / var
            
    return up_beta, down_beta

def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> Optional[float]:
    """Sortino ratio calculated from daily returns series."""
    if len(returns) < 20:
        return None
    
    # Convert annual RF to daily
    rf_daily = (1 + rf) ** (1/TRADING_DAYS_PER_YEAR) - 1
    
    excess_returns = returns - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return None
        
    downside_dev = np.sqrt((downside_returns**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if downside_dev == 0:
        return None
        
    # Annualize mean excess return
    mean_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    return mean_excess / downside_dev

def calculate_rolling_sip_returns(fund_nav: pd.Series, months: int = 12) -> pd.Series:
    """Calculate rolling 1Y SIP returns for the fund."""
    # Resample to monthly
    fund_monthly = fund_nav.resample('MS').first()
    
    if len(fund_monthly) < months + 1:
        return pd.Series(dtype=float)
        
    returns = {}
    for i in range(len(fund_monthly) - months):
        f_invest = fund_monthly.iloc[i : i+months]
        f_final = fund_monthly.iloc[i+months]
        
        # Fund SIP Return
        f_units = 1000 / f_invest
        f_val = f_units.sum() * f_final
        f_ret = (f_val - (1000 * months)) / (1000 * months)
        
        returns[fund_monthly.index[i+months]] = f_ret
        
    return pd.Series(returns)

def calculate_sip_stability(fund_nav: pd.Series) -> Optional[float]:
    """Calculate SIP stability over the last 3 years (rolling 1Y SIPs)."""
    # Need at least 2 years of data to have 1 year of rolling 1Y SIPs
    if len(fund_nav) < MIN_DAYS_1Y * 2:
        return None
        
    # Get last 3 years of NAV
    recent_nav = fund_nav.iloc[-MIN_DAYS_3Y:]
    rolling_sips = calculate_rolling_sip_returns(recent_nav, months=12)
    
    if len(rolling_sips) < 12:
        return None
        
    mean_sip = rolling_sips.mean()
    std_sip = rolling_sips.std()
    
    if std_sip == 0:
        return None
        
    return mean_sip / std_sip

# ===================================================================
# Analysis Pipeline
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    equity_nav: pd.Series,
    gold_nav: pd.Series,
    silver_nav: pd.Series,
    name: str,
    aum: float,
) -> dict:
    """Compute all metrics for a single fund."""
    
    n = len(fund_nav)
    result = {"mfId": mf_id, "name": name, "aum": round(aum, 2)}
    
    # Returns
    fund_rets = daily_returns(fund_nav)
    equity_rets = daily_returns(equity_nav)
    gold_rets = daily_returns(gold_nav)
    silver_rets = daily_returns(silver_nav)
    
    # Basic Metrics
    result["cagr_3y"] = annualised_return(fund_nav, MIN_DAYS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_DAYS_5Y)
    
    # 1. Cycle Agility Score
    eq_up_beta, eq_down_beta = compute_up_down_beta(fund_rets, equity_rets)
    if eq_up_beta is not None and eq_down_beta is not None:
        result["agility_score"] = eq_up_beta - eq_down_beta
    else:
        result["agility_score"] = None
        
    # 2. Precious Metals Upside Capture
    gold_up_beta, _ = compute_up_down_beta(fund_rets, gold_rets)
    silver_up_beta, _ = compute_up_down_beta(fund_rets, silver_rets)
    
    pm_betas = []
    if gold_up_beta is not None:
        pm_betas.append(gold_up_beta)
    if silver_up_beta is not None:
        pm_betas.append(silver_up_beta)
        
    if pm_betas:
        result["pm_up_beta"] = np.mean(pm_betas)
    else:
        result["pm_up_beta"] = None
        
    # 3. Long-Term Risk-Adjusted Growth (Sortino)
    if len(fund_rets) >= MIN_DAYS_1Y:
        recent_rets = fund_rets.iloc[-MIN_DAYS_3Y:] if len(fund_rets) >= MIN_DAYS_3Y else fund_rets
        result["sortino_3y"] = sortino_ratio(recent_rets)
    else:
        result["sortino_3y"] = None
    
    # 4. SIP Return Stability
    result["sip_stability"] = calculate_sip_stability(fund_nav)
    
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
    Build composite score optimized for 1-year forward returns based on predictive metrics.
    
    Weights:
    - Cycle Agility Score: 30%
    - Precious Metals Upside Capture: 30%
    - Sortino Ratio (3Y): 20%
    - SIP Stability: 20%
    """
    
    score_components = {
        "agility_score":        (True,  0.30),
        "pm_up_beta":           (True,  0.30),
        "sortino_3y":           (True,  0.20),
        "sip_stability":        (True,  0.20),
    }
    
    df = df.copy()
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)
    
    for col, (higher_better, weight) in score_components.items():
        if col not in df.columns:
            continue
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        
        # Funds with short history might have None/NaN for some metrics.
        # Instead of just ignoring it (which scales up their other noisy metrics),
        # give them a below-average percentile (40th) for the missing metric.
        pctl = pctl.fillna(40.0)
        
        contribution = pctl * weight
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += contribution[mask]
        applied_weight[mask] += weight
        
    df["score"] = np.where(
        applied_weight > 0,
        df["raw_score"] / applied_weight,
        0
    )
    
    # Linear penalty based on data_days history
    # 756 days (3y) or more -> penalty = 1.0 (no penalty)
    # 252 days (1y) -> penalty = ~0.80
    # < 50 days -> penalty = ~0.72
    penalty = np.clip((df["data_days"] / MIN_DAYS_3Y) * 0.3 + 0.7, 0.5, 1.0)
    df["score"] = df["score"] * penalty
    
    df["score"] = df["score"].round(2)
    
    return df

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 70)
    print(f"  MULTI ASSET MUTUAL FUND SCORING - GEMINI MODEL")
    print(f"  Target: Maximize 1-Year Returns via Cycle Management")
    print("=" * 70)

    provider = MfDataProvider(date=date)
    
    # Load Equity Benchmark
    indices = provider.list_indices()
    equity_idx_id = indices.get(EQUITY_INDEX)
    if not equity_idx_id:
        logger.error(f"Equity index {EQUITY_INDEX} not found.")
        return
        
    equity_df = provider.get_index_chart(equity_idx_id)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
    equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)
    equity_nav = equity_df.set_index("timestamp")["nav"]
    
    # Load Gold MF
    gold_df = provider.get_mf_chart(GOLD_MF_ID)
    gold_df["timestamp"] = pd.to_datetime(gold_df["timestamp"], utc=True)
    gold_df = gold_df.sort_values("timestamp").reset_index(drop=True)
    gold_nav = gold_df.set_index("timestamp")["nav"]
    
    # Load Silver MF
    silver_df = provider.get_mf_chart(SILVER_MF_ID)
    silver_df["timestamp"] = pd.to_datetime(silver_df["timestamp"], utc=True)
    silver_df = silver_df.sort_values("timestamp").reset_index(drop=True)
    silver_nav = silver_df.set_index("timestamp")["nav"]
    
    # Load Funds
    df_all = provider.list_all_mf()
    maa_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Found {len(maa_df)} Multi Asset Allocation funds")
    
    results = []
    for _, row in maa_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = row.get("aum", 0) or 0
        
        try:
            chart = provider.get_mf_chart(mf_id)
            if len(chart) < 50: # Need at least some data
                continue
                
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            chart = chart.sort_values("timestamp").reset_index(drop=True)
            fund_nav = chart.set_index("timestamp")["nav"]
            
            metrics = analyse_fund(mf_id, fund_nav, equity_nav, gold_nav, silver_nav, name, aum)
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
    
    output = pd.DataFrame()
    output["mfId"] = df_scored["mfId"]
    output["name"] = df_scored["name"]
    output["rank"] = df_scored["rank"]
    output["score"] = df_scored["score"]
    output["data_days"] = df_scored["data_days"]
    output["cagr_3y"] = df_scored["cagr_3y"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["cagr_5y"] = df_scored["cagr_5y"].apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
    output["agility_score"] = df_scored["agility_score"].apply(fmt)
    output["pm_up_beta"] = df_scored["pm_up_beta"].apply(fmt)
    output["sortino_3y"] = df_scored["sortino_3y"].apply(fmt)
    output["sip_stability"] = df_scored["sip_stability"].apply(fmt)
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
    p = argparse.ArgumentParser(description="Multi Asset MF screener (Gemini)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
