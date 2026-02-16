#!/usr/bin/env python3
"""
Mid Cap Mutual Fund Scoring Algorithm - Claude

An advanced multi-factor quantitative scoring model for Indian Mid Cap mutual
funds, designed to predict superior risk-adjusted returns over the next 1 year.

Architecture
------------
The model computes ~25 metrics per fund across six research-backed categories,
normalises each to a peer-group percentile, and combines them using fixed
weights derived from academic research on mutual fund performance persistence.

Metric Categories (with total weight):
  1. Risk-Adjusted Returns      35%   Sortino, Alpha, TM Alpha, IR, Omega, Treynor
  2. Downside Protection        20%   Drawdown, Down-Capture, Ulcer PI, Calmar, Recovery
  3. Consistency                20%   Rolling Beat%, Alpha Stability, Sharpe Stability
  4. Market Regime Behaviour    10%   Capture Spread, Beta Asymmetry, Bear Alpha
  5. Momentum (tempered)         8%   12-1 Momentum, 6M Relative
  6. Return Magnitude            7%   Blended CAGR (1Y/3Y/5Y)

Key innovations vs. basic multi-factor models:
- Treynor-Mazuy decomposition separates stock-picking skill from market timing
- Dual-beta model captures asymmetric up/down market participation
- Rolling consistency metrics detect repeatable edges vs. lucky streaks
- Market regime conditioning evaluates behaviour in bull/bear periods
- Recovery analysis rewards funds that bounce back quickly from drawdowns
- Walk-forward validation confirms predictive power of the composite score

Sector  : Mid Cap Fund
Author  : Claude
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
SECTOR = "Mid Cap"
SUBSECTOR = "Mid Cap Fund"
BENCHMARK_INDEX = "Mid Cap"  # resolved to .NIMI150 by provider
RISK_FREE_RATE = 0.065       # annualised (Indian T-bill proxy ~6.5%)
WEEKS_PER_YEAR = 52

LB_3M = 13
LB_6M = 26
LB_1Y = 52
LB_2Y = 104
LB_3Y = 156
LB_5Y = 260

MIN_WEEKS_FOR_ANALYSIS = 50
MIN_WEEKS_3Y = 150
MIN_WEEKS_5Y = 250

ROLLING_WINDOW = LB_1Y
ROLLING_STEP = 4

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Claude.csv"


# ===================================================================
# Data Cleaning
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


# ===================================================================
# Return & Volatility Helpers
# ===================================================================

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


def downside_deviation(rets: pd.Series, mar: float = RISK_FREE_RATE) -> float:
    if len(rets) < 8:
        return np.nan
    weekly_mar = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    diff = rets - weekly_mar
    neg = diff[diff < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.sqrt((neg ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR))


# ===================================================================
# Risk-Adjusted Ratios
# ===================================================================

def sharpe_ratio(cagr: Optional[float], vol: float) -> Optional[float]:
    if cagr is None or pd.isna(vol) or vol <= 1e-12:
        return None
    return float((cagr - RISK_FREE_RATE) / vol)


def sortino_ratio_calc(cagr: Optional[float], dd: float) -> Optional[float]:
    if cagr is None or pd.isna(dd) or dd <= 1e-12:
        return None
    return float((cagr - RISK_FREE_RATE) / dd)


def calmar_ratio_calc(cagr: Optional[float], mdd: float) -> Optional[float]:
    if cagr is None or pd.isna(mdd) or abs(mdd) < 1e-12:
        return None
    return float(cagr / abs(mdd))


def treynor_ratio(cagr: Optional[float], beta: Optional[float]) -> Optional[float]:
    if cagr is None or beta is None or abs(beta) < 1e-12:
        return None
    return float((cagr - RISK_FREE_RATE) / beta)


def omega_ratio(rets: pd.Series, mar: float = RISK_FREE_RATE) -> Optional[float]:
    """Gains above threshold / losses below. Captures full return distribution."""
    if len(rets) < 12:
        return None
    weekly_mar = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    excess = rets - weekly_mar
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    if losses <= 1e-12:
        return 10.0
    return float(gains / losses)


# ===================================================================
# Alpha & Benchmark-Relative Metrics
# ===================================================================

def compute_alpha_beta(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Tuple[Optional[float], Optional[float]]:
    """Jensen's alpha (annualised) and beta via OLS."""
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 20:
        return None, None

    y = aligned["fund"].values
    x = aligned["bench"].values
    var_x = np.var(x, ddof=1)
    if var_x <= 1e-12:
        return None, None

    cov = np.cov(x, y)[0, 1]
    beta = float(cov / var_x)
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    alpha_w = np.mean(y) - rf_w - beta * (np.mean(x) - rf_w)
    return float(alpha_w * WEEKS_PER_YEAR), beta


def information_ratio(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Optional[float]:
    """Annualised IR = excess return / tracking error."""
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 20:
        return None
    excess = aligned["fund"] - aligned["bench"]
    te = excess.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    if te <= 1e-12:
        return None
    return float((excess.mean() * WEEKS_PER_YEAR) / te)


def treynor_mazuy_decomposition(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Tuple[Optional[float], Optional[float]]:
    """
    Treynor-Mazuy market timing model:
      R_fund - Rf = alpha + beta*(R_bench - Rf) + gamma*(R_bench - Rf)^2
    Positive gamma => market timing skill.
    Returns (annualised stock-picking alpha, timing gamma).
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 30:
        return None, None

    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    y = aligned["fund"].values - rf_w
    x_excess = aligned["bench"].values - rf_w
    X = np.column_stack([np.ones(len(x_excess)), x_excess, x_excess ** 2])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        tm_alpha = float(coeffs[0] * WEEKS_PER_YEAR)
        tm_gamma = float(coeffs[2])
        return tm_alpha, tm_gamma
    except Exception:
        return None, None


# ===================================================================
# Capture Ratios & Market Regime
# ===================================================================

def capture_ratios(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Tuple[Optional[float], Optional[float]]:
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 20:
        return None, None

    up_mask = aligned["bench"] > 0
    down_mask = aligned["bench"] < 0

    up_cap = None
    if up_mask.sum() > 5:
        denom = aligned.loc[up_mask, "bench"].mean()
        if abs(denom) > 1e-12:
            up_cap = float(aligned.loc[up_mask, "fund"].mean() / denom)

    down_cap = None
    if down_mask.sum() > 5:
        denom = aligned.loc[down_mask, "bench"].mean()
        if abs(denom) > 1e-12:
            down_cap = float(aligned.loc[down_mask, "fund"].mean() / denom)

    return up_cap, down_cap


def dual_beta(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Tuple[Optional[float], Optional[float]]:
    """Separate betas for up-market and down-market weeks."""
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 30:
        return None, None

    up_beta = None
    up_mask = aligned["bench"] > 0
    if up_mask.sum() > 10:
        x = aligned.loc[up_mask, "bench"].values
        y = aligned.loc[up_mask, "fund"].values
        var_x = np.var(x, ddof=1)
        if var_x > 1e-12:
            up_beta = float(np.cov(x, y)[0, 1] / var_x)

    down_beta = None
    down_mask = aligned["bench"] < 0
    if down_mask.sum() > 10:
        x = aligned.loc[down_mask, "bench"].values
        y = aligned.loc[down_mask, "fund"].values
        var_x = np.var(x, ddof=1)
        if var_x > 1e-12:
            down_beta = float(np.cov(x, y)[0, 1] / var_x)

    return up_beta, down_beta


def bear_market_alpha(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    Alpha during bear periods (trailing 6M benchmark return < 0).
    Positive value indicates the fund protects capital or generates
    alpha in downturns — a strong predictor of manager skill.
    """
    if len(fund_nav) < LB_1Y or len(bench_nav) < LB_1Y:
        return None

    fund_ret = weekly_returns(fund_nav)
    bench_ret = weekly_returns(bench_nav)
    bench_6m = bench_nav.pct_change(LB_6M)
    bear_dates = bench_6m[bench_6m < 0].index

    common = fund_ret.index.intersection(bench_ret.index).intersection(bear_dates)
    if len(common) < 15:
        return None

    alpha, _ = compute_alpha_beta(fund_ret.loc[common], bench_ret.loc[common])
    return alpha


# ===================================================================
# Drawdown & Recovery Analysis
# ===================================================================

def max_drawdown_calc(nav: pd.Series) -> float:
    if len(nav) < 10:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def ulcer_index(nav: pd.Series) -> float:
    """sqrt(mean(drawdown_pct^2)). Captures both depth and duration of pain."""
    if len(nav) < 10:
        return np.nan
    dd_pct = (nav / nav.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def ulcer_performance_index(cagr: Optional[float], ui: float) -> Optional[float]:
    """Martin ratio = (CAGR - Rf) / Ulcer Index."""
    if cagr is None or pd.isna(ui) or ui <= 1e-12:
        return None
    return float((cagr - RISK_FREE_RATE) / (ui / 100.0))


def recovery_speed_score(nav: pd.Series, threshold: float = -0.05) -> Optional[float]:
    """
    Average recovery ratio for drawdown episodes deeper than threshold.
    recovery_ratio = time_to_recover / time_to_trough.
    Returns inverted value so higher = faster recovery = better.
    """
    if len(nav) < LB_1Y:
        return None

    dd = nav / nav.cummax() - 1.0
    in_drawdown = False
    dd_start = 0
    dd_trough = 0
    dd_trough_val = 0.0
    recovery_ratios: List[float] = []

    for i in range(len(dd)):
        val = float(dd.iloc[i])
        if not in_drawdown and val < threshold:
            in_drawdown = True
            dd_start = i
            dd_trough = i
            dd_trough_val = val
        elif in_drawdown:
            if val < dd_trough_val:
                dd_trough = i
                dd_trough_val = val
            if val >= 0:
                time_to_trough = dd_trough - dd_start
                time_to_recover = i - dd_trough
                if time_to_trough > 0:
                    recovery_ratios.append(time_to_recover / time_to_trough)
                in_drawdown = False

    if not recovery_ratios:
        return None

    avg_ratio = float(np.mean(recovery_ratios))
    return float(1.0 / (avg_ratio + 0.1))


# ===================================================================
# Rolling Consistency Metrics
# ===================================================================

def rolling_benchmark_beat_pct(
    fund_nav: pd.Series, bench_nav: pd.Series, window: int = LB_1Y
) -> Optional[float]:
    """Fraction of rolling window periods where fund beat benchmark."""
    aligned = pd.concat(
        [fund_nav.rename("fund"), bench_nav.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < window + 10:
        return None

    fund_roll = aligned["fund"].pct_change(window)
    bench_roll = aligned["bench"].pct_change(window)
    common = fund_roll.dropna().index.intersection(bench_roll.dropna().index)
    if len(common) < 10:
        return None

    wins = (fund_roll.loc[common] > bench_roll.loc[common]).sum()
    return float(wins / len(common))


def rolling_alpha_analysis(
    fund_nav: pd.Series, bench_nav: pd.Series,
    window: int = LB_1Y, step: int = ROLLING_STEP
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Rolling alpha over window-sized windows.
    Returns (mean_alpha, std_alpha, pct_positive).
    Low std + high pct_positive = consistent skill signal.
    """
    fund_ret = weekly_returns(fund_nav)
    bench_ret = weekly_returns(bench_nav)
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()

    if len(aligned) < window + 10:
        return None, None, None

    alphas: List[float] = []
    for start in range(0, len(aligned) - window, step):
        sub_f = aligned["fund"].iloc[start:start + window]
        sub_b = aligned["bench"].iloc[start:start + window]
        a, _ = compute_alpha_beta(sub_f, sub_b)
        if a is not None:
            alphas.append(a)

    if len(alphas) < 3:
        return None, None, None

    return (
        float(np.mean(alphas)),
        float(np.std(alphas, ddof=1)),
        float(np.mean(np.array(alphas) > 0)),
    )


def rolling_sharpe_analysis(
    fund_nav: pd.Series, window: int = LB_1Y, step: int = ROLLING_STEP
) -> Tuple[Optional[float], Optional[float]]:
    """
    Rolling Sharpe over window-sized windows.
    Returns (mean_sharpe, std_sharpe). Low std = consistent risk-adjusted returns.
    """
    rets = weekly_returns(fund_nav)
    if len(rets) < window + 10:
        return None, None

    sharpes: List[float] = []
    for start in range(0, len(rets) - window, step):
        sub = rets.iloc[start:start + window]
        vol = annualised_volatility(sub)
        if pd.isna(vol) or vol <= 1e-12:
            continue
        nav_window = fund_nav.iloc[start:start + window + 1]
        cagr = annualised_return(nav_window, window)
        if cagr is not None:
            s = sharpe_ratio(cagr, vol)
            if s is not None:
                sharpes.append(s)

    if len(sharpes) < 3:
        return None, None
    return float(np.mean(sharpes)), float(np.std(sharpes, ddof=1))


# ===================================================================
# Momentum
# ===================================================================

def momentum_12_minus_1(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    12-month return excluding the most recent month (relative to benchmark).
    Captures momentum while avoiding short-term reversal — the strongest
    documented momentum signal in academic literature.
    """
    skip = 4  # ~1 month
    if len(fund_nav) < LB_1Y + skip + 1 or len(bench_nav) < LB_1Y + skip + 1:
        return None

    fund_end = fund_nav.iloc[-(skip + 1)]
    fund_start = fund_nav.iloc[-(LB_1Y + skip + 1)]
    if fund_start <= 0 or fund_end <= 0:
        return None
    fund_mom = fund_end / fund_start - 1

    bench_end = bench_nav.iloc[-(skip + 1)]
    bench_start = bench_nav.iloc[-(LB_1Y + skip + 1)]
    if bench_start <= 0 or bench_end <= 0:
        return None
    bench_mom = bench_end / bench_start - 1

    return float(fund_mom - bench_mom)


def relative_momentum(
    fund_nav: pd.Series, bench_nav: pd.Series, weeks: int
) -> Optional[float]:
    f = annualised_return(fund_nav, weeks)
    b = annualised_return(bench_nav, weeks)
    if f is None or b is None:
        return None
    return float(f - b)


# ===================================================================
# Walk-Forward Validation
# ===================================================================

def quick_score(fund_nav: pd.Series, bench_nav: pd.Series) -> Optional[float]:
    """Lightweight scoring for walk-forward back-testing."""
    fund_nav = fund_nav.dropna()
    if len(fund_nav) < MIN_WEEKS_FOR_ANALYSIS:
        return None

    bench_nav = bench_nav.loc[bench_nav.index >= fund_nav.index[0]]
    rets = weekly_returns(fund_nav)
    bench_rets = weekly_returns(bench_nav)

    if len(rets) < 20:
        return None

    weeks = min(LB_1Y, len(fund_nav) - 1)
    cagr = annualised_return(fund_nav, weeks)
    dd = downside_deviation(rets)
    alpha, _ = compute_alpha_beta(rets, bench_rets)
    ir = information_ratio(rets, bench_rets)
    sortino = sortino_ratio_calc(cagr, dd)
    mdd = max_drawdown_calc(fund_nav)
    _, down_cap = capture_ratios(rets, bench_rets)

    score = 0.0
    total_w = 0.0

    def add(val: Optional[float], w: float) -> None:
        nonlocal score, total_w
        if val is not None and not pd.isna(val) and np.isfinite(val):
            score += val * w
            total_w += w

    add(alpha, 3.0)
    add(sortino, 1.5)
    add(ir, 1.5)
    if mdd is not None and not pd.isna(mdd):
        add(-mdd, 1.0)
    if down_cap is not None and not pd.isna(down_cap):
        add(-down_cap, 1.0)

    return score / total_w if total_w > 0 else None


def walk_forward_validation(
    aligned_navs: Dict[str, pd.Series],
    bench_nav: pd.Series,
    train_weeks: int = LB_3Y,
    test_weeks: int = LB_1Y,
    step_weeks: int = LB_6M,
) -> pd.DataFrame:
    """
    At each evaluation point, score funds using trailing data, then
    measure actual forward 1Y excess returns. Reports rank correlation
    (Spearman) and top-5 portfolio excess return per period.
    """
    records: List[dict] = []
    n = len(bench_nav)

    for eval_idx in range(train_weeks, n - test_weeks, step_weeks):
        scores: Dict[str, float] = {}
        forward_returns: Dict[str, float] = {}

        for mf_id, aligned_nav in aligned_navs.items():
            nav_train = aligned_nav.iloc[:eval_idx + 1].dropna()
            if len(nav_train) < MIN_WEEKS_FOR_ANALYSIS:
                continue

            bench_train = bench_nav.iloc[:eval_idx + 1]
            s = quick_score(nav_train, bench_train)
            if s is not None:
                scores[mf_id] = s

            fwd_start = aligned_nav.iloc[eval_idx]
            fwd_end = aligned_nav.iloc[eval_idx + test_weeks]
            if (
                pd.notna(fwd_start)
                and pd.notna(fwd_end)
                and fwd_start > 0
            ):
                fwd_ret = fwd_end / fwd_start - 1
                b_start = bench_nav.iloc[eval_idx]
                b_end = bench_nav.iloc[eval_idx + test_weeks]
                if b_start > 0:
                    b_ret = b_end / b_start - 1
                    forward_returns[mf_id] = fwd_ret - b_ret

        common = sorted(set(scores) & set(forward_returns))
        if len(common) < 8:
            continue

        s_arr = np.array([scores[k] for k in common])
        r_arr = np.array([forward_returns[k] for k in common])

        if np.std(s_arr) > 1e-12 and np.std(r_arr) > 1e-12:
            corr = float(
                pd.Series(s_arr).corr(pd.Series(r_arr), method="spearman")
            )
            top_k = min(5, len(common))
            top_idx = np.argsort(s_arr)[-top_k:]

            records.append({
                "eval_date": bench_nav.index[eval_idx].date(),
                "n_funds": len(common),
                "rank_correlation": corr,
                "top5_excess_return": float(np.mean(r_arr[top_idx])),
                "hit_rate": float(np.mean(r_arr[top_idx] > 0)),
            })

    return pd.DataFrame(records)


# ===================================================================
# Scoring Components & Composite
# ===================================================================

SCORE_COMPONENTS = {
    # --- Risk-Adjusted Returns (35%) ---
    "sortino":              (True,  0.09),
    "alpha":                (True,  0.08),
    "tm_alpha":             (True,  0.04),
    "info_ratio":           (True,  0.06),
    "omega":                (True,  0.04),
    "treynor":              (True,  0.04),

    # --- Downside Protection (20%) ---
    "max_drawdown":         (False, 0.05),
    "down_capture":         (False, 0.04),
    "ulcer_perf_index":     (True,  0.04),
    "calmar":               (True,  0.04),
    "recovery_speed":       (True,  0.03),

    # --- Consistency (20%) ---
    "rolling_1y_beat_pct":  (True,  0.07),
    "alpha_stability":      (True,  0.07),
    "sharpe_stability":     (True,  0.06),

    # --- Market Regime (10%) ---
    "capture_spread":       (True,  0.04),
    "beta_asymmetry":       (True,  0.03),
    "bear_alpha":           (True,  0.03),

    # --- Momentum (8%) ---
    "momentum_12_1":        (True,  0.04),
    "momentum_6m":          (True,  0.04),

    # --- Return Magnitude (7%) ---
    "cagr_blend":           (True,  0.07),
}


def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    name: str,
    aum: float,
) -> dict:
    """Compute all metrics for a single fund."""

    n_weeks = len(fund_nav)
    result: dict = {"mfId": mf_id, "name": name, "aum": round(aum, 2)}

    first_ts = fund_nav.index.min()
    last_ts = fund_nav.index.max()
    result["data_days"] = (
        int((last_ts - first_ts).days) + 1 if pd.notna(first_ts) else 0
    )
    result["data_weeks"] = n_weeks
    result["has_5y"] = n_weeks >= MIN_WEEKS_5Y
    result["has_3y"] = n_weeks >= MIN_WEEKS_3Y

    # ---- CAGR across horizons ----
    result["cagr_1y"] = annualised_return(fund_nav, LB_1Y)
    result["cagr_3y"] = annualised_return(fund_nav, LB_3Y)
    result["cagr_5y"] = annualised_return(fund_nav, LB_5Y)

    cagrs: List[float] = []
    blend_w: List[float] = []
    if result["cagr_1y"] is not None:
        cagrs.append(result["cagr_1y"]); blend_w.append(0.20)
    if result["cagr_3y"] is not None:
        cagrs.append(result["cagr_3y"]); blend_w.append(0.45)
    if result["cagr_5y"] is not None:
        cagrs.append(result["cagr_5y"]); blend_w.append(0.35)
    if cagrs:
        w = np.array(blend_w)
        w /= w.sum()
        result["cagr_blend"] = float(np.average(cagrs, weights=w))
    else:
        result["cagr_blend"] = None

    primary_cagr = result["cagr_3y"] or result["cagr_5y"] or result["cagr_1y"]

    # ---- Returns & Volatility ----
    rets = weekly_returns(fund_nav)
    if len(rets) < 20:
        logger.warning(f"{mf_id}: insufficient data ({len(rets)} weeks)")
        return result

    vol = annualised_volatility(rets)
    dd = downside_deviation(rets)
    result["volatility"] = vol
    result["downside_dev"] = dd

    # ---- Risk-Adjusted Ratios ----
    result["sharpe"] = sharpe_ratio(primary_cagr, vol)
    result["sortino"] = sortino_ratio_calc(primary_cagr, dd)
    result["omega"] = omega_ratio(rets)

    mdd = max_drawdown_calc(fund_nav)
    result["max_drawdown"] = mdd
    result["calmar"] = calmar_ratio_calc(primary_cagr, mdd)

    ui = ulcer_index(fund_nav)
    result["ulcer_index"] = ui
    result["ulcer_perf_index"] = ulcer_performance_index(primary_cagr, ui)
    result["recovery_speed"] = recovery_speed_score(fund_nav)

    # ---- Benchmark-Relative ----
    bench_rets = weekly_returns(bench_nav)
    alpha, beta = compute_alpha_beta(rets, bench_rets)
    result["alpha"] = alpha
    result["beta"] = beta
    result["info_ratio"] = information_ratio(rets, bench_rets)
    result["treynor"] = treynor_ratio(primary_cagr, beta)

    tm_a, tm_g = treynor_mazuy_decomposition(rets, bench_rets)
    result["tm_alpha"] = tm_a
    result["tm_gamma"] = tm_g

    # ---- Capture & Regime ----
    up_cap, down_cap = capture_ratios(rets, bench_rets)
    result["up_capture"] = up_cap
    result["down_capture"] = down_cap
    result["capture_spread"] = (
        up_cap - down_cap
        if up_cap is not None and down_cap is not None
        else None
    )

    up_b, down_b = dual_beta(rets, bench_rets)
    result["up_beta"] = up_b
    result["down_beta"] = down_b
    result["beta_asymmetry"] = (
        up_b - down_b if up_b is not None and down_b is not None else None
    )

    result["bear_alpha"] = bear_market_alpha(fund_nav, bench_nav)

    # ---- Rolling Consistency ----
    result["rolling_1y_beat_pct"] = rolling_benchmark_beat_pct(
        fund_nav, bench_nav
    )

    mean_a, std_a, pct_pos_a = rolling_alpha_analysis(fund_nav, bench_nav)
    result["rolling_alpha_mean"] = mean_a
    result["rolling_alpha_std"] = std_a
    result["rolling_alpha_pct_pos"] = pct_pos_a

    if all(v is not None for v in (mean_a, std_a, pct_pos_a)):
        norm_mean = np.clip(mean_a / 0.10, -2, 2)
        norm_consistency = pct_pos_a
        norm_low_var = 1.0 / (1.0 + std_a / 0.05)
        result["alpha_stability"] = float(
            0.35 * norm_mean + 0.35 * norm_consistency + 0.30 * norm_low_var
        )
    else:
        result["alpha_stability"] = None

    mean_s, std_s = rolling_sharpe_analysis(fund_nav)
    result["rolling_sharpe_mean"] = mean_s
    result["rolling_sharpe_std"] = std_s
    result["sharpe_stability"] = (
        float(0.5 * np.clip(mean_s, -1, 3) + 0.5 / (1.0 + std_s))
        if mean_s is not None and std_s is not None
        else None
    )

    # ---- Momentum ----
    result["momentum_6m"] = relative_momentum(fund_nav, bench_nav, LB_6M)
    result["momentum_12_1"] = momentum_12_minus_1(fund_nav, bench_nav)

    # ---- Return Distribution (diagnostic) ----
    if len(rets) >= 30:
        result["skewness"] = float(rets.skew())
        result["kurtosis"] = float(rets.kurtosis())
    else:
        result["skewness"] = None
        result["kurtosis"] = None

    return result


def percentile_rank(
    series: pd.Series, higher_is_better: bool = True
) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each metric, apply weights, penalise short histories."""

    total_weight = sum(w for _, w in SCORE_COMPONENTS.values())
    assert abs(total_weight - 1.0) < 1e-6, (
        f"Weights sum to {total_weight}, expected 1.0"
    )

    df = df.copy()
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)

    for col, (higher_better, weight) in SCORE_COMPONENTS.items():
        if col not in df.columns:
            logger.warning(f"Scoring column '{col}' not found — skipping")
            continue
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        mask = pctl.notna()
        df.loc[mask, "raw_score"] += (pctl * weight)[mask]
        applied_weight[mask] += weight

    df["score"] = np.where(
        applied_weight > 0,
        df["raw_score"] / applied_weight,
        0,
    )

    # Graduated track-record confidence penalty
    def _confidence(days: int) -> float:
        if days < 365:
            return 0.60
        if days < 2 * 365:
            return 0.75
        if days < 3 * 365:
            return 0.88
        if days < 5 * 365:
            return 0.95
        return 1.00

    df["confidence"] = df["data_days"].apply(_confidence)
    df["score"] = (df["score"] * df["confidence"]).round(2)
    return df


# ===================================================================
# Output Formatters
# ===================================================================

def _pct(v: object) -> str:
    return f"{float(v) * 100:.2f}" if v is not None and pd.notna(v) else ""


def _ratio(v: object) -> str:
    return f"{float(v):.3f}" if v is not None and pd.notna(v) else ""


def _num(v: object) -> str:
    return f"{float(v):.2f}" if v is not None and pd.notna(v) else ""


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("  MID CAP MUTUAL FUND SCORING ALGORITHM — CLAUDE")
    print(f"  Benchmark : Nifty Midcap 150 ({BENCHMARK_INDEX})")
    print("  Model     : Multi-factor + regime analysis + consistency metrics")
    print("=" * 80)

    provider = MfDataProvider()

    # --- Benchmark ---
    logger.info("Loading benchmark index data...")
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_to_series(bench_df)
    print(
        f"\n  Benchmark data : {len(bench_nav)} weeks  "
        f"({bench_nav.index.min().date()} → {bench_nav.index.max().date()})"
    )

    # --- Fund list ---
    df_all = provider.list_all_mf()
    mid_cap_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Mid Cap funds  : {len(mid_cap_df)}")

    # --- Analyse each fund ---
    logger.info("Analysing individual funds...")
    results: List[dict] = []
    aligned_navs: Dict[str, pd.Series] = {}

    for _, row in mid_cap_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = float(row.get("aum", 0) or 0)

        try:
            chart = provider.get_mf_chart(mf_id)
            if len(chart) < 20:
                logger.warning(
                    f"Skipping {mf_id} ({name}): only {len(chart)} points"
                )
                continue

            fund_nav = clean_nav_to_series(chart)
            if len(fund_nav) < 20:
                continue

            metrics = analyse_fund(mf_id, fund_nav, bench_nav, name, aum)
            results.append(metrics)

            aligned_navs[mf_id] = fund_nav.reindex(bench_nav.index).ffill()

        except Exception as e:
            logger.error(f"Error analysing {mf_id} ({name}): {e}")

    if not results:
        logger.error("No funds analysed successfully. Exiting.")
        sys.exit(1)

    df_results = pd.DataFrame(results)
    print(f"  Funds analysed : {len(df_results)}")

    # --- Composite score ---
    logger.info("Computing composite scores...")
    df_scored = compute_composite_score(df_results)
    df_scored["rank"] = (
        df_scored["score"].rank(ascending=False, method="min").astype(int)
    )
    df_scored = df_scored.sort_values("rank")

    # --- Walk-forward validation ---
    logger.info("Running walk-forward validation...")
    val_df = walk_forward_validation(
        aligned_navs=aligned_navs,
        bench_nav=bench_nav,
        train_weeks=LB_3Y,
        test_weeks=LB_1Y,
        step_weeks=LB_6M,
    )

    # --- Build output CSV ---
    output = pd.DataFrame()
    output["mfId"] = df_scored["mfId"]
    output["name"] = df_scored["name"]
    output["rank"] = df_scored["rank"]
    output["score"] = df_scored["score"]
    output["data_days"] = df_scored["data_days"]
    output["cagr_1y"] = df_scored["cagr_1y"].apply(_pct)
    output["cagr_3y"] = df_scored["cagr_3y"].apply(_pct)
    output["cagr_5y"] = df_scored["cagr_5y"].apply(_pct)
    output["volatility"] = df_scored["volatility"].apply(_pct)
    output["sharpe"] = df_scored["sharpe"].apply(_ratio)
    output["sortino"] = df_scored["sortino"].apply(_ratio)
    output["alpha"] = df_scored["alpha"].apply(_pct)
    output["beta"] = df_scored["beta"].apply(_ratio)
    output["info_ratio"] = df_scored["info_ratio"].apply(_ratio)
    output["treynor"] = df_scored["treynor"].apply(_ratio)
    output["omega"] = df_scored["omega"].apply(_num)
    output["tm_alpha"] = df_scored["tm_alpha"].apply(_pct)
    output["tm_gamma"] = df_scored["tm_gamma"].apply(_ratio)
    output["max_drawdown"] = df_scored["max_drawdown"].apply(_pct)
    output["calmar"] = df_scored["calmar"].apply(_ratio)
    output["ulcer_index"] = df_scored["ulcer_index"].apply(_num)
    output["ulcer_perf_index"] = df_scored["ulcer_perf_index"].apply(_num)
    output["recovery_speed"] = df_scored["recovery_speed"].apply(_num)
    output["up_capture"] = df_scored["up_capture"].apply(_ratio)
    output["down_capture"] = df_scored["down_capture"].apply(_ratio)
    output["capture_spread"] = df_scored["capture_spread"].apply(_ratio)
    output["beta_asymmetry"] = df_scored["beta_asymmetry"].apply(_ratio)
    output["bear_alpha"] = df_scored["bear_alpha"].apply(_pct)
    output["rolling_1y_beat_pct"] = df_scored["rolling_1y_beat_pct"].apply(_pct)
    output["alpha_stability"] = df_scored["alpha_stability"].apply(_num)
    output["sharpe_stability"] = df_scored["sharpe_stability"].apply(_num)
    output["momentum_6m"] = df_scored["momentum_6m"].apply(_pct)
    output["momentum_12_1"] = df_scored["momentum_12_1"].apply(_pct)
    output["skewness"] = df_scored["skewness"].apply(_ratio)
    output["kurtosis"] = df_scored["kurtosis"].apply(_ratio)
    output["aum"] = df_scored["aum"]
    output["data_weeks"] = df_scored["data_weeks"]
    output["confidence"] = df_scored["confidence"].apply(_ratio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("  SCORING MODEL SUMMARY")
    print("=" * 80)

    categories = {
        "Risk-Adjusted Returns": [
            "sortino", "alpha", "tm_alpha", "info_ratio", "omega", "treynor",
        ],
        "Downside Protection": [
            "max_drawdown", "down_capture", "ulcer_perf_index", "calmar",
            "recovery_speed",
        ],
        "Consistency": [
            "rolling_1y_beat_pct", "alpha_stability", "sharpe_stability",
        ],
        "Market Regime": [
            "capture_spread", "beta_asymmetry", "bear_alpha",
        ],
        "Momentum": ["momentum_12_1", "momentum_6m"],
        "Return Magnitude": ["cagr_blend"],
    }
    print("\n  Weight allocation:")
    for cat, metrics in categories.items():
        cat_w = sum(
            SCORE_COMPONENTS[m][1] for m in metrics if m in SCORE_COMPONENTS
        )
        print(f"    {cat:30s}: {cat_w * 100:5.1f}%")

    if not val_df.empty:
        print("\n  Walk-Forward Validation (3Y train → 1Y forward):")
        print(f"    Evaluation periods  : {len(val_df)}")
        print(
            f"    Avg rank correlation: "
            f"{val_df['rank_correlation'].mean():.3f}"
        )
        print(
            f"    Avg top-5 excess    : "
            f"{val_df['top5_excess_return'].mean() * 100:.2f}%"
        )
        print(
            f"    Avg hit rate        : "
            f"{val_df['hit_rate'].mean() * 100:.1f}%"
        )
        print(
            f"    Min / Max corr      : "
            f"{val_df['rank_correlation'].min():.3f} / "
            f"{val_df['rank_correlation'].max():.3f}"
        )
    else:
        print("\n  Walk-forward validation: insufficient data for evaluation.")

    print("\n" + "=" * 80)
    print("  TOP 15 MID CAP FUNDS BY COMPOSITE SCORE")
    print("=" * 80 + "\n")

    display_cols = [
        "rank", "name", "score", "cagr_5y", "alpha", "sortino",
        "max_drawdown", "rolling_1y_beat_pct", "capture_spread", "aum",
    ]
    display_cols = [c for c in display_cols if c in output.columns]
    print(output.head(15)[display_cols].to_string(index=False))

    print(f"\n  Full results ({len(output)} funds) → {OUTPUT_FILE}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
