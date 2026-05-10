#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Gemini

A custom scoring algorithm for Total Market mutual funds (Flexi Cap, Multi Cap, Value, Contra, Focused)
tailored to a specific 24-month investment horizon: 12 months of SIP followed by a 12-month hold.

Metrics:
1. Mean Scenario XIRR (25%)
2. Size-Adjusted Alpha (15%)
3. Minimum Scenario XIRR (20%)
4. Drawdown Recovery Time (20%)
5. Downside Capture Ratio (20%)

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
import scipy.optimize
from dateutil.relativedelta import relativedelta

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
SECTOR = "Total Market"
SUBSECTORS = [
    "Contra Fund",
    "Flexi Cap Fund",
    "Focused Fund",
    "Multi Cap Fund",
    "Value Fund",
]
BENCHMARK_INDEX = "_NIFTY500"
FACTOR_INDICES = {
    "Large": "Large Cap",
    "Mid": "Mid Cap",
    "Small": "Small Cap",
}

RISK_FREE_RATE = 0.065
WEEKS_PER_YEAR = 52

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

def annualised_return(nav: pd.Series, years: float) -> Optional[float]:
    days = int(years * 365)
    if len(nav) < 2 or (nav.index[-1] - nav.index[0]).days < days:
        return None
    
    start_date = nav.index[-1] - timedelta(days=days)
    # Find nearest date >= start_date
    start_navs = nav[nav.index >= start_date]
    if start_navs.empty:
        return None
    start_val = start_navs.iloc[0]
    end_val = nav.iloc[-1]
    
    if start_val <= 0:
        return None
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


# ===================================================================
# XIRR & Simulation
# ===================================================================

def calculate_xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    """
    Calculate the Internal Rate of Return (IRR) for a schedule of cash flows using bisection.
    """
    if len(cashflows) < 2:
        return None
        
    start_date = cashflows[0][0]
    years = [(cf[0] - start_date).days / 365.25 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    
    def npv(r):
        if r <= -1.0:
            return float('inf')
        return sum(a / ((1 + r) ** y) for a, y in zip(amounts, years))
        
    # Bisection method
    low = -0.9999
    high = 100.0 # 10000%
    
    # Check if a root exists in the interval
    try:
        npv_low = npv(low)
        npv_high = npv(high)
    except:
        return None
        
    if npv_low * npv_high > 0:
        # If signs are the same, try to expand the upper bound
        high = 1000.0
        try:
            npv_high = npv(high)
        except:
            return None
        if npv_low * npv_high > 0:
            return None
            
    for _ in range(100):
        mid = (low + high) / 2.0
        try:
            npv_mid = npv(mid)
        except:
            return None
            
        if abs(npv_mid) < 1e-4:
            return mid
            
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
            
    return (low + high) / 2.0

def simulate_24m_scenario(nav: pd.Series, start_date: pd.Timestamp) -> Optional[float]:
    """
    Simulate 12 months SIP + 12 months hold starting from start_date.
    Returns the XIRR of the scenario.
    """
    end_date = start_date + relativedelta(months=24)
    if nav.index[-1] < end_date:
        return None
        
    cashflows = []
    total_units = 0.0
    sip_amount = 1000.0
    
    # 12 months SIP
    for i in range(12):
        target_date = start_date + relativedelta(months=i)
        # Find nearest available NAV date on or after target_date
        available_navs = nav[nav.index >= target_date]
        if available_navs.empty:
            return None
        
        actual_date = available_navs.index[0]
        # If the gap is too large (e.g., > 10 days), simulation is invalid
        if (actual_date - target_date).days > 10:
            return None
            
        current_nav = available_navs.iloc[0]
        units_bought = sip_amount / current_nav
        total_units += units_bought
        
        cashflows.append((actual_date, -sip_amount))
        
    # Month 24 exit
    available_navs = nav[nav.index <= end_date]
    if available_navs.empty:
        return None
        
    exit_date = available_navs.index[-1]
    # If the gap is too large, simulation is invalid
    if (end_date - exit_date).days > 10:
        return None
        
    exit_nav = available_navs.iloc[-1]
    final_value = total_units * exit_nav
    
    cashflows.append((exit_date, final_value))
    
    return calculate_xirr(cashflows)

def run_rolling_simulations(nav: pd.Series) -> List[float]:
    """
    Run the 24-month scenario on a rolling monthly basis.
    """
    if len(nav) < 2:
        return []
        
    first_date = nav.index[0]
    last_date = nav.index[-1]
    
    # Need at least 24 months of data
    if last_date < first_date + relativedelta(months=24):
        return []
        
    xirrs = []
    
    # Start on the 1st of the next month after first_date
    current_start = first_date.replace(day=1)
    if current_start < first_date:
        current_start += relativedelta(months=1)
        
    while current_start + relativedelta(months=24) <= last_date:
        xirr_val = simulate_24m_scenario(nav, current_start)
        if xirr_val is not None:
            xirrs.append(xirr_val)
        current_start += relativedelta(months=1)
        
    return xirrs

# ===================================================================
# Resilience & Factor Metrics
# ===================================================================

def drawdown_recovery_time(nav: pd.Series) -> Optional[int]:
    """
    Calculate the maximum number of days taken to recover from a peak-to-trough drawdown.
    """
    if len(nav) < 10:
        return None
        
    # Calculate running max
    running_max = nav.cummax()
    
    # Find drawdowns
    drawdown = (nav / running_max) - 1.0
    
    # Find periods where we are in a drawdown
    in_drawdown = drawdown < 0
    
    if not in_drawdown.any():
        return 0
        
    # Calculate duration of each drawdown
    max_duration = 0
    current_duration = 0
    drawdown_start = None
    
    for date, is_down in in_drawdown.items():
        if is_down:
            if drawdown_start is None:
                drawdown_start = date
        else:
            if drawdown_start is not None:
                duration = (date - drawdown_start).days
                max_duration = max(max_duration, duration)
                drawdown_start = None
                
    # Check if currently in a drawdown
    if drawdown_start is not None:
        duration = (nav.index[-1] - drawdown_start).days
        max_duration = max(max_duration, duration)
        
    return max_duration

def downside_capture_ratio(fund_nav: pd.Series, bench_nav: pd.Series) -> Optional[float]:
    """
    Calculate the downside capture ratio (monthly).
    """
    # Align series
    df = pd.DataFrame({'fund': fund_nav, 'bench': bench_nav}).dropna()
    if len(df) < 20:
        return None
        
    # Resample to monthly
    monthly = df.resample("ME").last()
    rets = monthly.pct_change().dropna()
    
    # Filter for months where benchmark was down
    down_months = rets[rets['bench'] < 0]
    
    if len(down_months) < 3:
        return None
        
    # Calculate annualized return during down months
    # Geometric mean of down months
    fund_down_ret = (1 + down_months['fund']).prod() ** (12 / len(down_months)) - 1
    bench_down_ret = (1 + down_months['bench']).prod() ** (12 / len(down_months)) - 1
    
    if bench_down_ret >= 0:
        return None
        
    return float(fund_down_ret / bench_down_ret)

def size_adjusted_alpha(
    fund_nav: pd.Series, 
    large_nav: pd.Series, 
    mid_nav: pd.Series, 
    small_nav: pd.Series
) -> Optional[float]:
    """
    Multi-factor regression against Large, Mid, and Small cap indices.
    Returns the annualized alpha.
    """
    # Align all series
    df = pd.DataFrame({
        'fund': fund_nav,
        'large': large_nav,
        'mid': mid_nav,
        'small': small_nav
    }).dropna()
    
    if len(df) < 50:
        return None
        
    # Resample to weekly to reduce noise but keep enough data points
    weekly = df.resample('W').last()
    rets = weekly.pct_change().dropna()
    
    rf_weekly = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    
    Y = rets['fund'] - rf_weekly
    X = rets[['large', 'mid', 'small']] - rf_weekly
    
    # Add constant for alpha
    X_vals = X.values
    ones = np.ones((X_vals.shape[0], 1))
    X_with_const = np.hstack((ones, X_vals))
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_with_const, Y.values, rcond=None)
        alpha_weekly = coeffs[0]
        return float(alpha_weekly * WEEKS_PER_YEAR)
    except Exception:
        return None

# ===================================================================
# Main Analysis Function
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    large_nav: pd.Series,
    mid_nav: pd.Series,
    small_nav: pd.Series,
    name: str,
    aum: float,
    subsector: str,
) -> dict:
    
    result = {
        "mfId": mf_id,
        "name": name,
        "aum": round(aum, 2),
        "subsector": subsector,
    }

    if fund_nav.empty:
        return result
        
    first_ts = fund_nav.index.min()
    last_ts = fund_nav.index.max()
    result["data_days"] = int((last_ts - first_ts).days) + 1
    
    if result["data_days"] < 365 * 2: # Need at least 2 years for one simulation
        return result

    # Basic Returns
    result["cagr_3y"] = annualised_return(fund_nav, 3.0)
    result["cagr_5y"] = annualised_return(fund_nav, 5.0)

    # Rolling Simulations
    fund_xirrs = run_rolling_simulations(fund_nav)
    bench_xirrs = run_rolling_simulations(bench_nav)
    
    if fund_xirrs:
        result["mean_scenario_xirr"] = float(np.mean(fund_xirrs))
        result["min_scenario_xirr"] = float(np.min(fund_xirrs))
        
        # Win Rate vs Index
        if bench_xirrs and len(fund_xirrs) == len(bench_xirrs):
            wins = sum(1 for f, b in zip(fund_xirrs, bench_xirrs) if f > b)
            result["win_rate_vs_index"] = float(wins / len(fund_xirrs))
    
    # Resilience Metrics
    result["drawdown_recovery_days"] = drawdown_recovery_time(fund_nav)
    result["downside_capture"] = downside_capture_ratio(fund_nav, bench_nav)
    
    # Size-Adjusted Alpha
    result["size_adjusted_alpha"] = size_adjusted_alpha(fund_nav, large_nav, mid_nav, small_nav)

    return result

# ===================================================================
# Scoring & Composite
# ===================================================================

SCORE_WEIGHTS = {
    # Metric: (Higher is Better, Weight)
    "mean_scenario_xirr": (True, 0.25),
    "size_adjusted_alpha": (True, 0.15),
    "min_scenario_xirr": (True, 0.20),
    "drawdown_recovery_days": (False, 0.20),
    "downside_capture": (False, 0.20),
}

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)

    for col, (higher_better, weight) in SCORE_WEIGHTS.items():
        if col not in df.columns:
            continue
        
        if df[col].notna().sum() < 5:
            continue

        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += (pctl * weight)[mask]
        applied_weight[mask] += weight

    df["score"] = np.where(
        applied_weight > 0.5,
        df["raw_score"] / applied_weight,
        0,
    )
    
    # Confidence adjustment based on history length
    def _confidence(days: int) -> float:
        if days < 3 * 365: return 0.70
        if days < 4 * 365: return 0.85
        return 1.00

    df["confidence"] = df["data_days"].apply(_confidence)
    df["score"] = (df["score"] * df["confidence"]).round(2)
    
    return df

# ===================================================================
# Formatting Helpers
# ===================================================================

def _pct(v): return f"{v*100:.2f}" if pd.notna(v) else ""
def _num(v): return f"{v:.2f}" if pd.notna(v) else ""
def _int(v): return f"{int(v)}" if pd.notna(v) else ""

# ===================================================================
# Main
# ===================================================================

def main(date: Optional[str] = None):
    print("\n" + "=" * 80)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM — GEMINI")
    print(f"  Benchmark : {BENCHMARK_INDEX}")
    print("=" * 80)

    provider = MfDataProvider(date=date)

    # --- Load Indices ---
    logger.info("Loading Index Data...")
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_to_series(bench_df)
    
    large_nav = clean_nav_to_series(provider.get_index_chart(FACTOR_INDICES["Large"]))
    mid_nav = clean_nav_to_series(provider.get_index_chart(FACTOR_INDICES["Mid"]))
    small_nav = clean_nav_to_series(provider.get_index_chart(FACTOR_INDICES["Small"]))

    # --- Load Funds ---
    df_all = provider.list_all_mf()
    sector_df = df_all[df_all["subsector"].isin(SUBSECTORS)].copy()
    print(f"  Total Funds in Sector: {len(sector_df)}")

    results = []
    
    logger.info("Analyzing funds...")
    
    for idx, row in sector_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        
        try:
            chart = provider.get_mf_chart(mf_id)
            fund_nav = clean_nav_to_series(chart)
            
            res = analyse_fund(
                mf_id, 
                fund_nav, 
                bench_nav,
                large_nav,
                mid_nav,
                small_nav,
                name, 
                row.get("aum", 0), 
                row["subsector"]
            )
            results.append(res)
            
        except Exception as e:
            logger.error(f"Error analyzing {mf_id}: {e}")

    if not results:
        print("No results generated.")
        return

    df_results = pd.DataFrame(results)
    df_scored = compute_composite_score(df_results)

    out_cols = [
        "mfId", "name", "rank", "score", "subsector", "data_days",
        "cagr_3y", "cagr_5y",
        "mean_scenario_xirr", "min_scenario_xirr", "win_rate_vs_index",
        "size_adjusted_alpha", "drawdown_recovery_days", "downside_capture",
        "confidence",
    ]
    for col in out_cols:
        if col not in df_scored.columns:
            df_scored[col] = np.nan
    
    # Rank
    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min")
    df_scored = df_scored.sort_values("rank")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_cols = [c for c in out_cols if c in df_scored.columns]
    export_df = df_scored[final_cols].copy()
    
    # Format specifics
    for col in ["cagr_3y", "cagr_5y", "mean_scenario_xirr", "min_scenario_xirr", "win_rate_vs_index", "size_adjusted_alpha"]:
        if col in export_df: export_df[col] = export_df[col].apply(_pct)
    if "downside_capture" in export_df: export_df["downside_capture"] = export_df["downside_capture"].apply(_num)
    if "drawdown_recovery_days" in export_df: export_df["drawdown_recovery_days"] = export_df["drawdown_recovery_days"].apply(_int)
    
    export_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Top 20 Display
    print("\nTop 20 Funds:")
    print(export_df.head(20).to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Total Market MF screener (Gemini)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
