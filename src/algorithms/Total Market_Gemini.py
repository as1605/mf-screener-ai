#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Gemini

A dynamic multi-factor model for scoring Indian Total Market mutual funds
(Contra, Flexi Cap, Focused, Multi Cap, Value).

This model differentiates itself by using a Multi-Factor Regression approach to isolate
'Pure Alpha' from market-cap exposure, recognizing that Total Market funds often
generate returns simply by tilting towards Mid/Small caps.

Key Components:
1. Multi-Factor Alpha (30%):
   - Decomposes returns into Large, Mid, and Small cap factors.
   - Rewards funds that generate alpha *after* accounting for their cap bias.
   - Penalizes 'fake alpha' coming solely from high beta to small caps.

2. Downside Resistance (20%):
   - Uses Ulcer Index (depth + duration) and Max Drawdown.
   - Focuses on the 'pain' felt by investors during holding periods.

3. Consistency of Skill (20%):
   - Rolling Information Ratio (1-year rolling windows).
   - Rewards funds that consistently deliver risk-adjusted excess returns,
     rather than those with one lucky year.

4. Upside Potential (15%):
   - Omega Ratio: Probability-weighted ratio of gains vs losses.
   - Captures the non-normal distribution of returns better than Sharpe.

5. Momentum (15%):
   - Recent performance trend using EWMA to identify funds currently in sync with the market.

Sector  : Total Market
Author  : Gemini
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

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
BENCHMARK_INDEX = "Total Market" # .NIFTY500
FACTOR_INDICES = {
    "Large": "Large Cap",  # .NSEI
    "Mid": "Mid Cap",      # .NIMI150
    "Small": "Small Cap",  # .NISM250
}

RISK_FREE_RATE = 0.065
WEEKS_PER_YEAR = 52

# Time Horizons
LB_1Y = 52
LB_3Y = 156
LB_5Y = 260

MIN_WEEKS_FOR_ANALYSIS = 50
MIN_WEEKS_3Y = 150

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

def weekly_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()

def annualised_return(nav: pd.Series, weeks: int) -> Optional[float]:
    if len(nav) < weeks + 1:
        return None
    start = nav.iloc[-(weeks + 1)]
    end = nav.iloc[-1]
    if start <= 0:
        return None
    years = weeks / WEEKS_PER_YEAR
    return float((end / start) ** (1.0 / years) - 1.0)

def annualised_volatility(rets: pd.Series) -> float:
    if len(rets) < 8:
        return np.nan
    return float(rets.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))

# ===================================================================
# Advanced Metric Calculations
# ===================================================================

def multi_factor_alpha(
    fund_ret: pd.Series, 
    large_ret: pd.Series, 
    mid_ret: pd.Series, 
    small_ret: pd.Series
) -> Dict[str, Optional[float]]:
    """
    Performs a multi-factor regression:
    R_fund - Rf = alpha + b1(R_large - Rf) + b2(R_mid - Rf) + b3(R_small - Rf)
    
    Returns alpha (annualised), betas, and R-squared.
    """
    # Align all series
    data = pd.DataFrame({
        'fund': fund_ret,
        'large': large_ret,
        'mid': mid_ret,
        'small': small_ret
    }).dropna()

    if len(data) < LB_1Y: # Need at least 1 year of data
        return {'mf_alpha': None, 'mf_r2': None}

    rf_weekly = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    
    Y = data['fund'] - rf_weekly
    X = data[['large', 'mid', 'small']] - rf_weekly
    X = sm_add_constant(X) # Add intercept for alpha

    try:
        # Using numpy lstsq for speed/simplicity over statsmodels
        # X matrix: [1, large, mid, small]
        coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        
        alpha_weekly = coeffs[0]
        beta_large = coeffs[1]
        beta_mid = coeffs[2]
        beta_small = coeffs[3]
        
        # Calculate R-squared
        y_mean = np.mean(Y)
        ss_tot = np.sum((Y - y_mean)**2)
        ss_res = np.sum((Y - (X @ coeffs))**2) if residuals.size == 0 else residuals[0]
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        return {
            'mf_alpha': float(alpha_weekly * WEEKS_PER_YEAR),
            'mf_beta_large': float(beta_large),
            'mf_beta_mid': float(beta_mid),
            'mf_beta_small': float(beta_small),
            'mf_r2': float(r2)
        }
    except Exception:
        return {'mf_alpha': None, 'mf_r2': None}

def sm_add_constant(X):
    """Helper to add constant column for intercept manually"""
    x_vals = X.values
    ones = np.ones((x_vals.shape[0], 1))
    return np.hstack((ones, x_vals))

def ulcer_index_calc(nav: pd.Series) -> float:
    """
    Ulcer Index: measures the depth and duration of drawdowns.
    """
    if len(nav) < 10:
        return np.nan
    
    # Calculate percentage drawdown series
    dd_pct = (nav / nav.cummax() - 1.0) * 100.0
    # Root Mean Squared Drawdown
    return float(np.sqrt(np.mean(dd_pct ** 2)))

def max_drawdown_calc(nav: pd.Series) -> float:
    if len(nav) < 10:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())

def rolling_information_ratio(
    fund_ret: pd.Series, 
    bench_ret: pd.Series, 
    window: int = LB_1Y, 
    step: int = 4
) -> Optional[float]:
    """
    Calculates the stability of the Information Ratio.
    Returns the mean IR penalised by its volatility.
    """
    aligned = pd.concat([fund_ret, bench_ret], axis=1).dropna()
    if len(aligned) < window + 10:
        return None
    
    f_r = aligned.iloc[:, 0]
    b_r = aligned.iloc[:, 1]
    
    irs = []
    
    for i in range(0, len(aligned) - window, step):
        sub_f = f_r.iloc[i : i + window]
        sub_b = b_r.iloc[i : i + window]
        
        excess = sub_f - sub_b
        te = excess.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        
        if te > 1e-6:
            ir = (excess.mean() * WEEKS_PER_YEAR) / te
            irs.append(ir)
            
    if not irs:
        return None
        
    mean_ir = np.mean(irs)
    std_ir = np.std(irs)
    
    # We want high mean IR and low std IR
    # Score = Mean / (1 + Std)
    return float(mean_ir / (1 + std_ir))

def omega_ratio(rets: pd.Series, threshold_annual: float = RISK_FREE_RATE) -> Optional[float]:
    """
    Omega Ratio: Probability weighted ratio of gains vs losses.
    Threshold is the risk-free rate converted to weekly.
    """
    if len(rets) < 20:
        return None
        
    threshold_weekly = (1 + threshold_annual) ** (1 / WEEKS_PER_YEAR) - 1
    excess = rets - threshold_weekly
    
    pos = excess[excess > 0].sum()
    neg = abs(excess[excess < 0].sum())
    
    if neg < 1e-12:
        return 10.0 # Capped max score for no losses
        
    return float(pos / neg)

def momentum_ewma(nav: pd.Series, span_weeks: int = 26) -> Optional[float]:
    """
    Exponentially Weighted Moving Average of returns.
    Proxy for recent trend strength.
    """
    if len(nav) < span_weeks:
        return None
    
    rets = weekly_returns(nav)
    ewma = rets.ewm(span=span_weeks, adjust=False).mean().iloc[-1]
    return float(ewma * WEEKS_PER_YEAR)

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

    n_weeks = len(fund_nav)
    result["data_weeks"] = n_weeks
    
    if n_weeks < MIN_WEEKS_FOR_ANALYSIS:
        return result

    # Basic Returns
    result["cagr_1y"] = annualised_return(fund_nav, LB_1Y)
    result["cagr_3y"] = annualised_return(fund_nav, LB_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, LB_5Y)
    
    primary_cagr = result["cagr_3y"] or result["cagr_5y"] or result["cagr_1y"]
    if primary_cagr is None:
        return result

    fund_rets = weekly_returns(fund_nav)
    bench_rets = weekly_returns(bench_nav)
    
    # 1. Multi-Factor Alpha
    large_rets = weekly_returns(large_nav)
    mid_rets = weekly_returns(mid_nav)
    small_rets = weekly_returns(small_nav)
    
    mf_res = multi_factor_alpha(fund_rets, large_rets, mid_rets, small_rets)
    result.update(mf_res)

    # 2. Downside Resistance
    result["ulcer_index"] = ulcer_index_calc(fund_nav)
    result["max_drawdown"] = max_drawdown_calc(fund_nav)
    
    # 3. Consistency
    result["rolling_ir"] = rolling_information_ratio(fund_rets, bench_rets)
    
    # 4. Upside Potential
    result["omega_ratio"] = omega_ratio(fund_rets)
    
    # 5. Momentum
    result["momentum_ewma"] = momentum_ewma(fund_nav, span_weeks=26)
    
    # Metadata
    first_ts = fund_nav.index.min()
    last_ts = fund_nav.index.max()
    result["data_days"] = int((last_ts - first_ts).days) + 1

    return result

# ===================================================================
# Scoring & Composite
# ===================================================================

SCORE_WEIGHTS = {
    # Metric: (Higher is Better, Weight)
    
    # Alpha (30%)
    "mf_alpha": (True, 0.30),
    
    # Downside (20%)
    "ulcer_index": (False, 0.12),
    "max_drawdown": (False, 0.08),
    
    # Consistency (20%)
    "rolling_ir": (True, 0.20),
    
    # Upside (15%)
    "omega_ratio": (True, 0.15),
    
    # Momentum (15%)
    "momentum_ewma": (True, 0.15),
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
        
        # Check if column has enough valid data
        if df[col].notna().sum() < 5:
            continue

        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += (pctl * weight)[mask]
        applied_weight[mask] += weight

    df["score"] = np.where(
        applied_weight > 0.5, # Must have at least 50% of weight metrics
        df["raw_score"] / applied_weight,
        0,
    )
    
    # Confidence adjustment based on history length
    def _confidence(days: int) -> float:
        if days < 365: return 0.60
        if days < 2 * 365: return 0.75
        if days < 3 * 365: return 0.90
        return 1.00

    df["confidence"] = df["data_days"].apply(_confidence)
    df["score"] = (df["score"] * df["confidence"]).round(2)
    
    return df

# ===================================================================
# Formatting Helpers
# ===================================================================

def _pct(v): return f"{v*100:.2f}" if pd.notna(v) else ""
def _num(v): return f"{v:.2f}" if pd.notna(v) else ""

# ===================================================================
# Main
# ===================================================================

def main():
    print("\n" + "=" * 80)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM — GEMINI")
    print(f"  Benchmark : {BENCHMARK_INDEX}")
    print("=" * 80)

    provider = MfDataProvider()

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
            
            if len(fund_nav) < MIN_WEEKS_FOR_ANALYSIS:
                continue

            # Align NAVs for this fund's timeframe
            common_idx = fund_nav.index.intersection(bench_nav.index)
            if len(common_idx) < MIN_WEEKS_FOR_ANALYSIS:
                continue
                
            res = analyse_fund(
                mf_id, 
                fund_nav.loc[common_idx], 
                bench_nav.loc[common_idx],
                large_nav.loc[large_nav.index.intersection(common_idx)],
                mid_nav.loc[mid_nav.index.intersection(common_idx)],
                small_nav.loc[small_nav.index.intersection(common_idx)],
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
    
    # Rank
    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min")
    df_scored = df_scored.sort_values("rank")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select and rename columns for output
    out_cols = [
        "mfId", "name", "rank", "score", "subsector", "data_days", 
        "cagr_3y", "cagr_5y", 
        "mf_alpha", "mf_r2", "mf_beta_large", "mf_beta_mid", "mf_beta_small",
        "ulcer_index", "max_drawdown", "rolling_ir", "omega_ratio", "momentum_ewma", "confidence"
    ]
    
    # Ensure columns exist
    final_cols = [c for c in out_cols if c in df_scored.columns]
    
    # Format for CSV
    export_df = df_scored[final_cols].copy()
    
    # Format specifics
    if "cagr_3y" in export_df: export_df["cagr_3y"] = export_df["cagr_3y"].apply(_pct)
    if "cagr_5y" in export_df: export_df["cagr_5y"] = export_df["cagr_5y"].apply(_pct)
    if "mf_alpha" in export_df: export_df["mf_alpha"] = export_df["mf_alpha"].apply(_pct)
    if "ulcer_index" in export_df: export_df["ulcer_index"] = export_df["ulcer_index"].apply(_num)
    if "max_drawdown" in export_df: export_df["max_drawdown"] = export_df["max_drawdown"].apply(_pct)
    if "omega_ratio" in export_df: export_df["omega_ratio"] = export_df["omega_ratio"].apply(_num)
    if "rolling_ir" in export_df: export_df["rolling_ir"] = export_df["rolling_ir"].apply(_num)
    
    export_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Top 20 Display
    print("\nTop 20 Funds:")
    print(export_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
