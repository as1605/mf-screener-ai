#!/usr/bin/env python3
"""
Multi Asset Allocation Fund Scoring Algorithm - Gemini Model

A quantitative scoring model for Indian Multi Asset Allocation mutual funds,
optimized for a 24-month horizon (12-month SIP + 12-month hold).

Incorporates cross-sector best practices and non-linear institutional scoring:
1. 24-Month Scenario Optimization (Sortino & Max Drawdown).
2. Asymmetry Score (Up-Beta / Down-Beta).
3. Manager Edge Decay (Forward-looking 6m vs 3y decay metrics).
4. Appraisal Ratio (Alpha / Idiosyncratic Risk) & Closet Indexer Penalty.
5. Decision-Tree Scoring (Hard Filters -> Decay Multipliers -> Alpha Boosts).

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
import scipy.stats as stats
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
TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_6M = 126
MIN_DAYS_1Y = 252
MIN_DAYS_3Y = 756
MIN_DAYS_4Y = 1008
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

def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from peak."""
    if len(returns) == 0: return 0.0
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    return drawdown.min()

def calculate_24m_scenario_sortino(fund_nav: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the mean return and Sortino ratio for a rolling 24-month scenario
    (12m SIP + 12m Hold).
    """
    fund_monthly = fund_nav.resample('MS').first()
    months_total = 24
    if len(fund_monthly) < months_total + 1:
        return None, None
        
    MAR = 0.10 # 10% cumulative hurdle rate over 24m (approx 5% annualized)
    
    returns = []
    for i in range(len(fund_monthly) - months_total):
        f_invest = fund_monthly.iloc[i : i+12]
        f_final = fund_monthly.iloc[i+months_total]
        f_units = 1000 / f_invest
        f_val = f_units.sum() * f_final
        f_ret = (f_val - (1000 * 12)) / (1000 * 12)
        returns.append(f_ret)
        
    if not returns: return None, None
    returns = np.array(returns)
    mean_ret = returns.mean()
    
    excess_returns = returns - MAR
    downside = excess_returns[excess_returns < 0]
    
    if len(downside) == 0:
        sortino = mean_ret / (returns.std() + 1e-6)
    else:
        downside_std = np.sqrt(np.mean(downside**2))
        sortino = (mean_ret - MAR) / downside_std
        
    return mean_ret, sortino

def compute_up_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Calculate Up-Market Beta (e.g. against Gold)."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None
    f_r = aligned.iloc[:, 0]
    b_r = aligned.iloc[:, 1]
    up_days = b_r > 0
    if up_days.sum() > 5:
        cov = np.cov(b_r[up_days], f_r[up_days])[0, 1]
        var = np.var(b_r[up_days], ddof=1)
        if var > 0:
            return cov / var
    return None

def compute_up_down_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Up-Market and Down-Market Beta for Asymmetry Score."""
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

def compute_information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Calculate Information Ratio."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = excess.std()
    if te > 0:
        return (excess.mean() / te) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return None

def compute_appraisal_and_r2(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Calculate 1Y Appraisal Ratio and R-Squared."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 50:
        return None, None
        
    # Take last 1Y (252 days)
    aligned = aligned.iloc[-MIN_DAYS_1Y:]
    if len(aligned) < 50:
        return None, None
        
    f_r = aligned.iloc[:, 0]
    b_r = aligned.iloc[:, 1]
    
    # R-Squared
    corr = f_r.corr(b_r)
    r_squared = corr ** 2 if pd.notna(corr) else None
    
    # Appraisal Ratio
    cov = np.cov(b_r, f_r)[0, 1]
    var = np.var(b_r, ddof=1)
    if var > 0:
        beta = cov / var
        # Annualized Returns
        f_ann = f_r.mean() * TRADING_DAYS_PER_YEAR
        b_ann = b_r.mean() * TRADING_DAYS_PER_YEAR
        alpha = f_ann - beta * b_ann
        
        # Idiosyncratic Risk
        residuals = f_r - beta * b_r
        idio_risk = residuals.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        appraisal = alpha / idio_risk if idio_risk > 0 else None
    else:
        appraisal = None
        
    return appraisal, r_squared

# ===================================================================
# Analysis Pipeline
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    equity_nav: pd.Series,
    gold_nav: pd.Series,
    name: str,
    aum: float,
) -> dict:
    """Compute all metrics for a single fund."""
    
    n = len(fund_nav)
    result = {"mfId": mf_id, "name": name, "aum": round(aum, 2), "data_days": n}
    
    fund_rets = daily_returns(fund_nav)
    equity_rets = daily_returns(equity_nav)
    gold_rets = daily_returns(gold_nav)
    
    # 24m Scenario
    mean_24m, sortino_24m = calculate_24m_scenario_sortino(fund_nav)
    result["mean_24m"] = mean_24m
    result["sortino_24m"] = sortino_24m
    
    # Max Drawdown
    result["max_drawdown"] = max_drawdown(fund_rets)
    
    # Asymmetry Score (Up-Beta / Down-Beta) against Equity Index
    up_beta, down_beta = compute_up_down_beta(fund_rets, equity_rets)
    if up_beta is not None and down_beta is not None and down_beta > 0:
        result["asymmetry_score"] = up_beta / down_beta
    else:
        result["asymmetry_score"] = None
        
    # Gold Upside Beta
    result["gold_up_beta"] = compute_up_beta(fund_rets, gold_rets)
    
    # Volatility
    result["volatility"] = fund_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Appraisal Ratio & R-Squared
    appraisal, r2 = compute_appraisal_and_r2(fund_rets, equity_rets)
    result["appraisal_ratio"] = appraisal
    result["r_squared_1y"] = r2
    
    # Manager Edge Decay Metrics
    if n >= MIN_DAYS_3Y:
        f_ret_3y = fund_rets.iloc[-MIN_DAYS_3Y:]
        aligned_3y = pd.concat([f_ret_3y, equity_rets], axis=1, join="inner").dropna()
        if not aligned_3y.empty:
            result["ir_3y"] = compute_information_ratio(aligned_3y.iloc[:, 0], aligned_3y.iloc[:, 1])
            result["vol_3y"] = aligned_3y.iloc[:, 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            _, d_beta_3y = compute_up_down_beta(aligned_3y.iloc[:, 0], aligned_3y.iloc[:, 1])
            result["downside_cap_3y"] = d_beta_3y
        else:
            result["ir_3y"] = result["vol_3y"] = result["downside_cap_3y"] = None
            
        f_ret_6m = fund_rets.iloc[-MIN_DAYS_6M:]
        aligned_6m = pd.concat([f_ret_6m, equity_rets], axis=1, join="inner").dropna()
        if not aligned_6m.empty:
            result["ir_6m"] = compute_information_ratio(aligned_6m.iloc[:, 0], aligned_6m.iloc[:, 1])
            result["vol_6m"] = aligned_6m.iloc[:, 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            _, d_beta_6m = compute_up_down_beta(aligned_6m.iloc[:, 0], aligned_6m.iloc[:, 1])
            result["downside_cap_6m"] = d_beta_6m
        else:
            result["ir_6m"] = result["vol_6m"] = result["downside_cap_6m"] = None
    else:
        result["ir_3y"] = result["vol_3y"] = result["downside_cap_3y"] = None
        result["ir_6m"] = result["vol_6m"] = result["downside_cap_6m"] = None
        
    # Basic Metrics
    result["cagr_3y"] = annualised_return(fund_nav, MIN_DAYS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_DAYS_5Y)

    return result

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank values to 0-100 percentile."""
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100

def compute_decision_tree_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a multi-stage non-linear score:
    1. Base Score (Sortino, MDD, Mean_24m, Gold_Beta)
    2. Hard Filters (Closet Indexer, AUM Bloat, Debt-Hugger, Bull Market Illusion)
    3. Edge Decay Multipliers (IR, Vol, Downside Capture)
    4. Alpha & Asymmetry Boosts (Top Quartile Appraisal & Asymmetry)
    """
    df = df.copy()
    
    # -------------------------------------------------------------------
    # 0. Base Core Score (Linear Combination of foundational safety metrics)
    # -------------------------------------------------------------------
    core_components = {
        "sortino_24m":  (True,  0.40),
        "max_drawdown": (True,  0.30),
        "mean_24m":     (True,  0.20),
        "gold_up_beta": (True,  0.10),
    }
    
    df["raw_core"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)
    
    for col, (higher_better, weight) in core_components.items():
        if col not in df.columns: continue
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        pctl = pctl.fillna(40.0)
        df["raw_core"] += pctl * weight
        applied_weight += weight
        
    df["score"] = np.where(applied_weight > 0, df["raw_core"] / applied_weight, 0)
    
    # -------------------------------------------------------------------
    # Stage 1: Hard Filters (Massive Penalties)
    # -------------------------------------------------------------------
    # a. Closet Indexer Penalty (R-Squared > 0.95)
    r2 = df["r_squared_1y"].fillna(0)
    filter_r2 = np.where(r2 > 0.95, 0.4, 1.0) # 60% penalty
    
    # b. AUM Bloat Penalty (> 30,000 Cr)
    aum = df["aum"].fillna(0)
    filter_aum = np.where(aum > 30000, 0.7, 1.0)
    filter_aum = np.where(aum > 50000, 0.5, filter_aum)
    
    # c. Debt-Hugger Penalty (mean_24m < 13%)
    mean_ret = df["mean_24m"].fillna(0)
    filter_debt = np.where(mean_ret < 0.13, 0.5, 1.0)
    
    # d. Bull Market Illusion (< 4 years data)
    filter_seasoning = np.where(df["data_days"] < MIN_DAYS_4Y, 0.6, 1.0)
    
    stage1_multiplier = filter_r2 * filter_aum * filter_debt * filter_seasoning
    df["score"] = df["score"] * stage1_multiplier
    
    # -------------------------------------------------------------------
    # Stage 2: Manager Edge Decay Multipliers
    # -------------------------------------------------------------------
    for idx, row in df.iterrows():
        decay_mult = 1.0
        # IR Decay
        ir_3y, ir_6m = row.get("ir_3y"), row.get("ir_6m")
        if pd.notna(ir_3y) and pd.notna(ir_6m) and ir_3y > 0:
            if ir_6m < (ir_3y * 0.8): decay_mult *= 0.8
                
        # Volatility Expansion
        vol_3y, vol_6m = row.get("vol_3y"), row.get("vol_6m")
        if pd.notna(vol_3y) and pd.notna(vol_6m) and vol_3y > 0:
            if vol_6m > (vol_3y * 1.2): decay_mult *= 0.8
                
        # Downside Capture Decay
        dc_3y, dc_6m = row.get("downside_cap_3y"), row.get("downside_cap_6m")
        if pd.notna(dc_3y) and pd.notna(dc_6m) and dc_3y > 0:
            if dc_6m > (dc_3y * 1.2): decay_mult *= 0.8
            
        df.at[idx, "score"] *= decay_mult

    # -------------------------------------------------------------------
    # Stage 3: Alpha & Asymmetry Boosts (Top Quartile)
    # -------------------------------------------------------------------
    # We apply a 20% score boost for funds in the top quartile of Appraisal Ratio,
    # and a 20% boost for top quartile Asymmetry Score.
    
    appr_pctl = percentile_rank(df["appraisal_ratio"])
    asym_pctl = percentile_rank(df["asymmetry_score"])
    
    boost_appr = np.where(appr_pctl >= 75.0, 1.2, 1.0)
    boost_asym = np.where(asym_pctl >= 75.0, 1.2, 1.0)
    
    df["score"] = df["score"] * boost_appr * boost_asym
    
    # Normalize back to roughly a 0-100 scale logically (some might exceed 100, we can clip)
    df["score"] = np.clip(df["score"], 0, 100).round(2)
    
    return df

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 70)
    print(f"  MULTI ASSET MUTUAL FUND SCORING - GEMINI MODEL")
    print(f"  Target: Non-Linear Decision Tree for 24m Horizon")
    print("=" * 70)

    provider = MfDataProvider(date=date)
    
    indices = provider.list_indices()
    equity_idx_id = indices.get(EQUITY_INDEX)
    if not equity_idx_id:
        logger.error(f"Equity index {EQUITY_INDEX} not found.")
        return
        
    equity_df = provider.get_index_chart(equity_idx_id)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
    equity_nav = equity_df.sort_values("timestamp").set_index("timestamp")["nav"]
    
    gold_df = provider.get_mf_chart(GOLD_MF_ID, duration='5y')
    gold_df["timestamp"] = pd.to_datetime(gold_df["timestamp"], utc=True)
    gold_nav = gold_df.sort_values("timestamp").set_index("timestamp")["nav"]
    
    df_all = provider.list_all_mf()
    maa_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Found {len(maa_df)} Multi Asset Allocation funds")
    
    results = []
    for _, row in maa_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = row.get("aum", 0) or 0
        
        try:
            chart = provider.get_mf_chart(mf_id, duration='5y')
            if len(chart) < 252:
                continue
                
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            fund_nav = chart.sort_values("timestamp").set_index("timestamp")["nav"]
            
            metrics = analyse_fund(mf_id, fund_nav, equity_nav, gold_nav, name, aum)
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error {mf_id}: {e}")
            continue
            
    if not results:
        print("No funds analyzed.")
        return

    df_results = pd.DataFrame(results)
    df_scored = compute_decision_tree_score(df_results)
    
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
    output["mean_24m"] = df_scored["mean_24m"].apply(fmt)
    output["sortino_24m"] = df_scored["sortino_24m"].apply(fmt)
    output["max_drawdown"] = df_scored["max_drawdown"].apply(fmt)
    output["appraisal_ratio"] = df_scored["appraisal_ratio"].apply(fmt)
    output["r_squared_1y"] = df_scored["r_squared_1y"].apply(fmt)
    output["asymmetry_score"] = df_scored["asymmetry_score"].apply(fmt)
    output["gold_up_beta"] = df_scored["gold_up_beta"].apply(fmt)
    output["ir_3y"] = df_scored["ir_3y"].apply(fmt)
    output["ir_6m"] = df_scored["ir_6m"].apply(fmt)
    output["downside_cap_3y"] = df_scored["downside_cap_3y"].apply(fmt)
    output["downside_cap_6m"] = df_scored["downside_cap_6m"].apply(fmt)
    output["aum"] = df_scored["aum"]
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    
    # Display Top 10
    print("\n" + "=" * 70)
    print("  TOP 10 FUNDS (Gemini Model - Decision Tree Multi Asset)")
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
