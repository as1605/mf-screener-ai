#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm

A multi-factor quantitative scoring model for Indian Small Cap mutual funds.
Designed to identify funds most likely to deliver superior risk-adjusted returns
over the next 1 year by analysing NAV history, benchmark-relative behaviour,
drawdown resilience, return consistency and recent momentum.

Methodology
-----------
1.  Compute per-fund metrics across multiple time horizons (1Y / 3Y / 5Y):
    - CAGR, annualised volatility, Sharpe, Sortino, Calmar ratios
    - Jensen's Alpha, Beta, Information Ratio vs Nifty SmallCap 250
    - Up / Down capture ratios (market-swing analysis)
    - Rolling-return consistency (% of rolling 1Y windows beating benchmark)
    - Maximum drawdown and recovery analysis
    - Short-term momentum (3M / 6M relative returns)

2.  Normalise each metric to a 0-100 percentile rank within the peer group.

3.  Combine into a weighted composite score emphasising:
    - Alpha generation & risk-adjusted returns  (40 %)
    - Downside protection & drawdown resilience  (20 %)
    - Consistency & benchmark beating frequency   (15 %)
    - Momentum & recent relative performance      (10 %)
    - Return magnitude (CAGR blend)               (15 %)

4.  Apply a track-record penalty for funds with < 3 Y data (reduced confidence).

Author : Claude
Sector : Small Cap Fund
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Path setup – allow running from repo root or from src/algorithms/
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
BENCHMARK_INDEX = "Small Cap"           # resolved to .NISM250 by provider
RISK_FREE_RATE = 0.065                  # annualised (Indian T-bill proxy ~6.5 %)
DATA_DATE = "2026-02-13"                # date folder containing cached data
WEEKS_PER_YEAR = 52
MIN_WEEKS_1Y = 50
MIN_WEEKS_3Y = 150
MIN_WEEKS_5Y = 250

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Claude.tsv"


# ===================================================================
# Helper functions
# ===================================================================

def weekly_returns(nav_series: pd.Series) -> pd.Series:
    """Compute simple weekly returns from a NAV series."""
    return nav_series.pct_change().dropna()


def annualised_return(nav_series: pd.Series, weeks: int) -> Optional[float]:
    """CAGR over the last *weeks* data points (weekly data)."""
    if len(nav_series) < weeks + 1:
        return None
    start = nav_series.iloc[-(weeks + 1)]
    end = nav_series.iloc[-1]
    if start <= 0:
        return None
    years = weeks / WEEKS_PER_YEAR
    return (end / start) ** (1 / years) - 1


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard-deviation of weekly returns."""
    return returns.std() * np.sqrt(WEEKS_PER_YEAR)


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """Annualised downside deviation below *mar* (minimum acceptable return)."""
    weekly_mar = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    diff = returns - weekly_mar
    neg = diff[diff < 0]
    if len(neg) == 0:
        return 0.0
    return np.sqrt((neg ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR)


def sharpe_ratio(cagr: float, vol: float, rf: float = RISK_FREE_RATE) -> Optional[float]:
    if vol == 0 or vol is None or cagr is None:
        return None
    return (cagr - rf) / vol


def sortino_ratio(cagr: float, dd: float, rf: float = RISK_FREE_RATE) -> Optional[float]:
    if dd == 0 or dd is None or cagr is None:
        return None
    return (cagr - rf) / dd


def max_drawdown(nav_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (returned as a negative fraction)."""
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    return dd.min()


def calmar_ratio(cagr: float, mdd: float) -> Optional[float]:
    if mdd == 0 or mdd is None or cagr is None:
        return None
    return cagr / abs(mdd)


def compute_alpha_beta(fund_ret: pd.Series, bench_ret: pd.Series):
    """Jensen's alpha and beta via OLS regression of fund ~ benchmark."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None, None
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    weekly_rf = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    alpha_weekly = y_mean - weekly_rf - beta * (x_mean - weekly_rf)
    alpha_annual = alpha_weekly * WEEKS_PER_YEAR
    return alpha_annual, beta


def information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Annualised information ratio = annualised excess return / tracking error."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = excess.std() * np.sqrt(WEEKS_PER_YEAR)
    if te == 0:
        return None
    return (excess.mean() * WEEKS_PER_YEAR) / te


def capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series):
    """Up-capture and down-capture ratios."""
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return None, None
    f = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    up_mask = b > 0
    down_mask = b < 0

    up_capture = None
    if up_mask.sum() > 5:
        up_capture = f[up_mask].mean() / b[up_mask].mean()

    down_capture = None
    if down_mask.sum() > 5:
        down_capture = f[down_mask].mean() / b[down_mask].mean()

    return up_capture, down_capture


def rolling_benchmark_beat_pct(
    fund_nav: pd.Series, bench_nav: pd.Series, window: int = 52
) -> Optional[float]:
    """Fraction of rolling *window*-week periods where fund CAGR > benchmark."""
    if len(fund_nav) < window + 10 or len(bench_nav) < window + 10:
        return None

    aligned = pd.concat(
        [fund_nav.rename("fund"), bench_nav.rename("bench")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < window + 10:
        return None

    fund_roll = aligned["fund"].pct_change(window).dropna()
    bench_roll = aligned["bench"].pct_change(window).dropna()

    common = fund_roll.index.intersection(bench_roll.index)
    if len(common) < 10:
        return None

    wins = (fund_roll.loc[common] > bench_roll.loc[common]).sum()
    return wins / len(common)


# ===================================================================
# Main analysis pipeline
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    name: str,
    aum: float,
) -> dict:
    """Compute all metrics for a single fund and return a flat dict."""

    n = len(fund_nav)
    result = {"mfId": mf_id, "name": name, "aum": round(aum, 2)}

    # ---------- CAGR across horizons ----------
    result["cagr_1y"] = annualised_return(fund_nav, MIN_WEEKS_1Y)
    result["cagr_3y"] = annualised_return(fund_nav, MIN_WEEKS_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, MIN_WEEKS_5Y)

    # Use longest available CAGR as primary return metric
    primary_cagr = result["cagr_5y"] or result["cagr_3y"] or result["cagr_1y"]
    result["primary_cagr"] = primary_cagr

    # ---------- Volatility ----------
    rets = weekly_returns(fund_nav)
    if len(rets) < 20:
        logger.warning(f"{mf_id}: insufficient return data ({len(rets)} weeks)")
        return result

    vol = annualised_volatility(rets)
    result["volatility"] = vol

    # ---------- Downside deviation ----------
    dd = downside_deviation(rets, mar=RISK_FREE_RATE)
    result["downside_dev"] = dd

    # ---------- Risk-adjusted ratios ----------
    result["sharpe"] = sharpe_ratio(primary_cagr, vol)
    result["sortino"] = sortino_ratio(primary_cagr, dd)

    # ---------- Max drawdown ----------
    mdd = max_drawdown(fund_nav)
    result["max_drawdown"] = mdd
    result["calmar"] = calmar_ratio(primary_cagr, mdd)

    # ---------- Benchmark-relative metrics ----------
    bench_rets = weekly_returns(bench_nav)

    alpha, beta = compute_alpha_beta(rets, bench_rets)
    result["alpha"] = alpha
    result["beta"] = beta

    ir = information_ratio(rets, bench_rets)
    result["info_ratio"] = ir

    up_cap, down_cap = capture_ratios(rets, bench_rets)
    result["up_capture"] = up_cap
    result["down_capture"] = down_cap

    # ---------- Consistency ----------
    rolling_beat = rolling_benchmark_beat_pct(fund_nav, bench_nav, window=52)
    result["rolling_1y_beat_pct"] = rolling_beat

    # Win rate (weeks with positive return)
    result["win_rate"] = (rets > 0).mean()

    # ---------- Momentum (relative to benchmark) ----------
    # 3-month (~13 weeks) and 6-month (~26 weeks) returns
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

    # ---------- Data quality flags ----------
    result["data_weeks"] = n
    result["data_days"] = (fund_nav.index.max() - fund_nav.index.min()).days + 1
    result["has_5y"] = n >= MIN_WEEKS_5Y
    result["has_3y"] = n >= MIN_WEEKS_3Y

    return result


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank values to 0-100 percentile. NaN stays NaN."""
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked  # invert so lower raw value => higher percentile
    return ranked * 100


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite score from normalised metric percentiles.

    Weight allocation rationale for Small Cap prediction:
    - Alpha & risk-adjusted returns dominate because Small Cap returns
      are driven more by stock-picking skill than market beta.
    - Downside protection is critical: large drawdowns in small caps can
      take years to recover and destroy compounding.
    - Consistency signals a repeatable strategy rather than lucky bets.
    - Momentum captures regime persistence in small-cap rallies/corrections.
    - Raw CAGR is included but down-weighted to avoid chasing past returns.
    """

    score_components = {
        # (metric_column, higher_is_better, weight)
        "sharpe":               (True,  0.10),
        "sortino":              (True,  0.12),
        "alpha":                (True,  0.15),
        "info_ratio":           (True,  0.08),
        "up_capture":           (True,  0.05),
        "down_capture":         (False, 0.10),  # lower down-capture is better
        "max_drawdown":         (False, 0.07),  # less negative is better (inverted)
        "calmar":               (True,  0.05),
        "rolling_1y_beat_pct":  (True,  0.08),
        "win_rate":             (True,  0.02),
        "momentum_6m":          (True,  0.05),
        "momentum_3m":          (True,  0.05),
        "cagr_5y":              (True,  0.04),
        "cagr_3y":              (True,  0.04),
    }

    total_weight = sum(w for _, w in score_components.values())
    assert abs(total_weight - 1.0) < 1e-6, f"Weights sum to {total_weight}, expected 1.0"

    df = df.copy()
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)

    for col, (higher_better, weight) in score_components.items():
        if col not in df.columns:
            continue
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        contribution = pctl * weight
        # Only accumulate weight where metric is available
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += contribution[mask]
        applied_weight[mask] += weight

    # Normalise by actual weight applied (handles missing metrics gracefully)
    # raw_score is already on 0-100 scale (percentile * weight), so divide
    # by applied_weight to get a clean 0-100 composite score.
    df["score"] = np.where(
        applied_weight > 0,
        df["raw_score"] / applied_weight,
        0,
    )

    # ---------- Track-record penalty ----------
    # Funds with < 3Y data get a confidence penalty (max 15 % reduction)
    penalty = pd.Series(1.0, index=df.index)
    short_track = ~df["has_3y"]
    penalty[short_track] = 0.85
    df["score"] = df["score"] * penalty

    # Round
    df["score"] = df["score"].round(2)

    return df


# ===================================================================
# Entry point
# ===================================================================

def main():
    print("\n" + "=" * 70)
    print(f"  SMALL CAP MUTUAL FUND SCORING ALGORITHM")
    print(f"  Data date : {DATA_DATE}")
    print(f"  Benchmark : Nifty SmallCap 250 ({BENCHMARK_INDEX})")
    print("=" * 70)

    # --- Initialise data provider ---
    provider = MfDataProvider()

    # --- Load benchmark ---
    logger.info("Loading benchmark index data...")
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"], utc=True)
    bench_df = bench_df.sort_values("timestamp").reset_index(drop=True)
    bench_nav = bench_df.set_index("timestamp")["nav"]
    print(f"\n  Benchmark data : {len(bench_nav)} weeks  "
          f"({bench_nav.index.min().date()} → {bench_nav.index.max().date()})")

    # --- Load fund list and filter Small Cap ---
    df_all = provider.list_all_mf()
    small_cap_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Small Cap funds: {len(small_cap_df)}")

    # --- Analyse each fund ---
    logger.info("Analysing individual funds...")
    results = []
    for _, row in small_cap_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = row.get("aum", 0) or 0

        try:
            chart = provider.get_mf_chart(mf_id)
            if len(chart) < 20:
                logger.warning(f"Skipping {mf_id} ({name}): only {len(chart)} data points")
                continue

            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            chart = chart.sort_values("timestamp").reset_index(drop=True)
            fund_nav = chart.set_index("timestamp")["nav"]

            metrics = analyse_fund(mf_id, fund_nav, bench_nav, name, aum)
            results.append(metrics)

        except Exception as e:
            logger.error(f"Error analysing {mf_id} ({name}): {e}")
            continue

    if not results:
        logger.error("No funds analysed successfully. Exiting.")
        sys.exit(1)

    df_results = pd.DataFrame(results)
    print(f"  Funds analysed : {len(df_results)}")

    # --- Compute composite score ---
    logger.info("Computing composite scores...")
    df_scored = compute_composite_score(df_results)

    # --- Rank ---
    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min").astype(int)
    df_scored = df_scored.sort_values("rank")

    # --- Format output columns ---
    fmt = lambda v, mult=100: f"{v * mult:.2f}" if pd.notna(v) and v is not None else ""
    fmt_ratio = lambda v: f"{v:.3f}" if pd.notna(v) and v is not None else ""

    output = pd.DataFrame()
    output["mfId"] = df_scored["mfId"]
    output["name"] = df_scored["name"]
    output["rank"] = df_scored["rank"]
    output["score"] = df_scored["score"]
    output["data_days"] = df_scored["data_days"]
    output["cagr_1y"] = df_scored["cagr_1y"].apply(fmt)
    output["cagr_3y"] = df_scored["cagr_3y"].apply(fmt)
    output["cagr_5y"] = df_scored["cagr_5y"].apply(fmt)
    output["volatility"] = df_scored["volatility"].apply(fmt)
    output["sharpe"] = df_scored["sharpe"].apply(fmt_ratio)
    output["sortino"] = df_scored["sortino"].apply(fmt_ratio)
    output["alpha"] = df_scored["alpha"].apply(fmt)
    output["beta"] = df_scored["beta"].apply(fmt_ratio)
    output["info_ratio"] = df_scored["info_ratio"].apply(fmt_ratio)
    output["max_drawdown"] = df_scored["max_drawdown"].apply(fmt)
    output["calmar"] = df_scored["calmar"].apply(fmt_ratio)
    output["up_capture"] = df_scored["up_capture"].apply(fmt_ratio)
    output["down_capture"] = df_scored["down_capture"].apply(fmt_ratio)
    output["rolling_1y_beat_pct"] = df_scored["rolling_1y_beat_pct"].apply(fmt)
    output["momentum_6m"] = df_scored["momentum_6m"].apply(fmt)
    output["momentum_3m"] = df_scored["momentum_3m"].apply(fmt)
    output["aum"] = df_scored["aum"]
    output["data_weeks"] = df_scored["data_weeks"]

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, sep="\t", index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # --- Print top 15 ---
    print("\n" + "=" * 70)
    print("  TOP 15 SMALL CAP FUNDS BY COMPOSITE SCORE")
    print("=" * 70 + "\n")

    display_cols = ["rank", "name", "score", "cagr_5y", "sharpe", "alpha",
                    "max_drawdown", "rolling_1y_beat_pct", "aum"]
    top15 = output.head(15)[display_cols]
    print(top15.to_string(index=False))

    print(f"\n  Full results ({len(output)} funds) → {OUTPUT_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
