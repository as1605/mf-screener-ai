#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Codex (Walk-Forward Tuned)

This model ranks Indian Total Market mutual funds (Contra, Flexi Cap, Focused,
Multi Cap, Value) for next-1Y return potential using:

1) Multi-factor cross-sectional scoring.
2) Walk-forward backtesting and weight tuning.
3) Subsector-aware rank blending to reduce one-style overconcentration.
4) Mild market-regime tilt on top of tuned base weights for live ranking.

Outputs:
- results/Total Market_Codex.csv
- data/tmp/Total Market_Codex_weights.csv
- data/tmp/Total Market_Codex_backtest.csv
- data/tmp/Total Market_Codex_tuning_trials.csv
- data/tmp/Total Market_Codex_regime.csv
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
SECTOR = "Total Market"
SUBSECTORS = [
    "Contra Fund",
    "Flexi Cap Fund",
    "Focused Fund",
    "Multi Cap Fund",
    "Value Fund",
]
BENCHMARK_INDEX = "Total Market"  # resolves to .NIFTY500

RISK_FREE_RATE = 0.065
WEEKS_PER_YEAR = 52

LB_3M = 13
LB_6M = 26
LB_1Y = 52
LB_2Y = 104
LB_3Y = 156
LB_5Y = 260

FORWARD_HORIZON_WEEKS = 52
EVAL_STEP_WEEKS = 8

MIN_WEEKS_TO_CONSIDER_FUND = 50
MIN_FUNDS_PER_DATE = 14
TOP_K = 7
MAX_STALE_WEEKS = 6

# Tuning controls
SEARCH_TRIALS = 220
REFINE_MUTATIONS_PER_BASE = 2
TOP_TRIALS_FOR_ENSEMBLE = 8
RANDOM_SEED = 77
MAX_FACTOR_WEIGHT = 0.22
MAX_CV_FOLDS = 3

# Blend a small amount of within-subsector ranking into the total rank.
SUBSECTOR_RANK_BLEND = 0.22

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Codex.csv"

TMP_OUTPUT_DIR = ROOT_DIR / "data" / "tmp"
WEIGHTS_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_weights.csv"
BACKTEST_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_backtest.csv"
TUNING_TRIALS_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_tuning_trials.csv"


# (factor_name, higher_is_better)
FACTOR_SPECS: List[Tuple[str, bool]] = [
    # --- Manager skill / active edge ---
    ("alpha_1y", True),
    ("info_1y", True),
    ("sortino_1y", True),
    ("rolling_alpha_stability_2y", True),
    ("rolling_active_ir_2y", True),

    # --- Market-cycle handling ---
    ("up_capture_1y", True),
    ("down_capture_1y", False),
    ("capture_spread_1y", True),
    ("bear_excess_1y", True),
    ("high_vol_excess_1y", True),

    # --- Drawdown resilience ---
    ("max_drawdown_2y", False),
    ("ulcer_2y", False),
    ("recovery_speed_2y", True),
    ("rolling_beat_rate_2y", True),

    # --- Multi-horizon structure ---
    ("cross_horizon_consistency", True),
    ("beta_distance_1y", False),

    # --- Momentum with overheat control ---
    ("momentum_6m_rel", True),
    ("momentum_12_1_rel", True),
    ("overheat_penalty", False),

    # --- Compounding anchors ---
    ("cagr_3y", True),
    ("cagr_5y", True),
]

FACTOR_NAMES = [name for name, _ in FACTOR_SPECS]
RANK_COLS = [f"rank_{name}" for name in FACTOR_NAMES]
FACTOR_INDEX = {name: i for i, name in enumerate(FACTOR_NAMES)}

SKILL_FACTORS = {
    "alpha_1y",
    "info_1y",
    "sortino_1y",
    "rolling_alpha_stability_2y",
    "rolling_active_ir_2y",
}
CYCLE_FACTORS = {
    "up_capture_1y",
    "down_capture_1y",
    "capture_spread_1y",
    "bear_excess_1y",
    "high_vol_excess_1y",
}
RESILIENCE_FACTORS = {
    "max_drawdown_2y",
    "ulcer_2y",
    "recovery_speed_2y",
    "rolling_beat_rate_2y",
}
STRUCTURE_FACTORS = {
    "cross_horizon_consistency",
    "beta_distance_1y",
}
MOMENTUM_FACTORS = {
    "momentum_6m_rel",
    "momentum_12_1_rel",
    "overheat_penalty",
}
ANCHOR_FACTORS = {
    "cagr_3y",
    "cagr_5y",
}

AGGRESSIVE_TILT_FACTORS = {
    "alpha_1y",
    "up_capture_1y",
    "capture_spread_1y",
    "momentum_6m_rel",
    "momentum_12_1_rel",
}
DEFENSIVE_TILT_FACTORS = {
    "down_capture_1y",
    "bear_excess_1y",
    "high_vol_excess_1y",
    "max_drawdown_2y",
    "ulcer_2y",
    "recovery_speed_2y",
    "cross_horizon_consistency",
    "beta_distance_1y",
}
ANCHOR_TILT_FACTORS = {
    "cagr_3y",
    "cagr_5y",
}

FACTOR_REQUIRED_WEEKS: Dict[str, int] = {
    "alpha_1y": LB_1Y,
    "info_1y": LB_1Y,
    "sortino_1y": LB_1Y,
    "rolling_alpha_stability_2y": LB_2Y,
    "rolling_active_ir_2y": LB_2Y,
    "up_capture_1y": LB_1Y,
    "down_capture_1y": LB_1Y,
    "capture_spread_1y": LB_1Y,
    "bear_excess_1y": LB_1Y,
    "high_vol_excess_1y": LB_1Y,
    "max_drawdown_2y": LB_2Y,
    "ulcer_2y": LB_2Y,
    "recovery_speed_2y": LB_2Y,
    "rolling_beat_rate_2y": LB_2Y,
    "cross_horizon_consistency": LB_5Y,
    "beta_distance_1y": LB_1Y,
    "momentum_6m_rel": LB_1Y,
    "momentum_12_1_rel": LB_1Y,
    "overheat_penalty": LB_1Y,
    "cagr_3y": LB_3Y,
    "cagr_5y": LB_5Y,
}

HISTORY_NEUTRAL_SCORE = 50.0
HISTORY_CONFIDENCE_CAP_WEEKS = LB_5Y
MIN_HISTORY_RELIABILITY = 0.55
MAX_HISTORY_RELIABILITY = 1.00
FACTOR_COVERAGE_WEIGHT = 0.45
HISTORY_LENGTH_WEIGHT = 0.55


# ===================================================================
# Data and metric helpers
# ===================================================================
def clean_nav_chart(df: pd.DataFrame) -> pd.Series:
    """Convert raw chart dataframe into sorted, clean NAV series."""
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


def get_nav_window(nav: pd.Series, end_idx: int, weeks: int) -> pd.Series:
    """Get [end_idx-weeks, end_idx] window; returns empty series when unavailable."""
    if end_idx < weeks:
        return pd.Series(dtype=float)
    window = nav.iloc[end_idx - weeks: end_idx + 1]
    if len(window) != weeks + 1:
        return pd.Series(dtype=float)
    if window.isna().any():
        return pd.Series(dtype=float)
    if (window <= 0).any():
        return pd.Series(dtype=float)
    return window


def cagr_from_window(nav_window: pd.Series, weeks: int) -> float:
    if nav_window.empty:
        return np.nan
    start = nav_window.iloc[0]
    end = nav_window.iloc[-1]
    if start <= 0 or end <= 0:
        return np.nan
    years = weeks / WEEKS_PER_YEAR
    return float((end / start) ** (1.0 / years) - 1.0)


def cagr_at(nav: pd.Series, end_idx: int, weeks: int) -> float:
    return cagr_from_window(get_nav_window(nav, end_idx, weeks), weeks)


def simple_return_from_window(nav_window: pd.Series) -> float:
    if nav_window.empty:
        return np.nan
    start = nav_window.iloc[0]
    end = nav_window.iloc[-1]
    if start <= 0 or end <= 0:
        return np.nan
    return float(end / start - 1.0)


def simple_return_at(nav: pd.Series, end_idx: int, weeks: int) -> float:
    return simple_return_from_window(get_nav_window(nav, end_idx, weeks))


def annualised_volatility(returns: pd.Series) -> float:
    if len(returns) < 8:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def downside_deviation(returns: pd.Series, mar: float = RISK_FREE_RATE) -> float:
    if len(returns) < 8:
        return np.nan
    mar_weekly = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    downside = returns - mar_weekly
    downside = downside[downside < 0]
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR))


def sortino_ratio(cagr_value: float, down_dev: float) -> float:
    if pd.isna(cagr_value) or pd.isna(down_dev) or down_dev <= 1e-12:
        return np.nan
    return float((cagr_value - RISK_FREE_RATE) / down_dev)


def alpha_beta_info(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float, float]:
    """
    Jensen alpha (annualized), beta, and information ratio from weekly returns.
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")],
        axis=1,
    ).dropna()
    if len(aligned) < 12:
        return np.nan, np.nan, np.nan

    f = aligned["fund"].to_numpy(dtype=float)
    b = aligned["bench"].to_numpy(dtype=float)
    var_b = np.var(b, ddof=1)

    if var_b <= 1e-12:
        alpha = np.nan
        beta = np.nan
    else:
        cov = np.cov(b, f)[0, 1]
        beta = float(cov / var_b)
        rf_weekly = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
        alpha_weekly = np.mean(f) - rf_weekly - beta * (np.mean(b) - rf_weekly)
        alpha = float(alpha_weekly * WEEKS_PER_YEAR)

    active = f - b
    tracking_err = np.std(active, ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    if tracking_err <= 1e-12:
        info = np.nan
    else:
        info = float((np.mean(active) * WEEKS_PER_YEAR) / tracking_err)

    return alpha, beta, info


def capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")],
        axis=1,
    ).dropna()
    if len(aligned) < 12:
        return np.nan, np.nan

    up_mask = aligned["bench"] > 0
    down_mask = aligned["bench"] < 0

    up_capture = np.nan
    if up_mask.sum() >= 6:
        denom = aligned.loc[up_mask, "bench"].mean()
        if abs(denom) > 1e-12:
            up_capture = float(aligned.loc[up_mask, "fund"].mean() / denom)

    down_capture = np.nan
    if down_mask.sum() >= 6:
        denom = aligned.loc[down_mask, "bench"].mean()
        if abs(denom) > 1e-12:
            down_capture = float(aligned.loc[down_mask, "fund"].mean() / denom)

    return up_capture, down_capture


def max_drawdown_depth(nav_window: pd.Series) -> float:
    if len(nav_window) < 10:
        return np.nan
    drawdown = nav_window / nav_window.cummax() - 1.0
    return float(abs(drawdown.min()))


def ulcer_index(nav_window: pd.Series) -> float:
    if len(nav_window) < 10:
        return np.nan
    drawdown_pct = (nav_window / nav_window.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(drawdown_pct ** 2)))


def recovery_speed_score(nav_window: pd.Series, threshold: float = -0.05) -> float:
    """
    Higher is better. Measures how quickly a fund tends to recover from
    meaningful drawdowns within the lookback window.
    """
    if len(nav_window) < 40:
        return np.nan

    dd = nav_window / nav_window.cummax() - 1.0
    if float(dd.min()) > abs(threshold):
        return 1.0

    in_episode = False
    start_i = 0
    trough_i = 0
    trough_val = 0.0
    ratios: List[float] = []

    for i, val in enumerate(dd.to_numpy(dtype=float)):
        if not in_episode and val <= threshold:
            in_episode = True
            start_i = i
            trough_i = i
            trough_val = val
            continue

        if in_episode:
            if val < trough_val:
                trough_i = i
                trough_val = val
            if val >= -1e-6:
                fall_len = max(trough_i - start_i, 1)
                rec_len = max(i - trough_i, 0)
                ratios.append(rec_len / float(fall_len))
                in_episode = False

    if not ratios:
        return 0.2

    med_ratio = float(np.median(ratios))
    return float(1.0 / (1.0 + med_ratio))


def rolling_beat_rate(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    end_idx: int,
    lookback: int = LB_2Y,
    roll: int = LB_6M,
) -> float:
    """Fraction of rolling windows in which fund beats benchmark."""
    if end_idx < lookback:
        return np.nan
    f = fund_nav.iloc[end_idx - lookback: end_idx + 1]
    b = bench_nav.iloc[end_idx - lookback: end_idx + 1]
    if len(f) < lookback + 1 or len(b) < lookback + 1:
        return np.nan

    active_roll = f.pct_change(roll) - b.pct_change(roll)
    active_roll = active_roll.dropna()
    if len(active_roll) < 10:
        return np.nan
    return float(np.mean(active_roll > 0))


def rolling_active_ir(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    end_idx: int,
    lookback: int = LB_2Y,
    roll: int = LB_6M,
) -> float:
    """
    Information-ratio style consistency based on rolling active returns.
    Higher means stronger and steadier relative performance.
    """
    if end_idx < lookback:
        return np.nan
    f = fund_nav.iloc[end_idx - lookback: end_idx + 1]
    b = bench_nav.iloc[end_idx - lookback: end_idx + 1]
    if len(f) < lookback + 1 or len(b) < lookback + 1:
        return np.nan

    active_roll = f.pct_change(roll) - b.pct_change(roll)
    active_roll = active_roll.dropna()
    if len(active_roll) < 10:
        return np.nan
    std = float(active_roll.std(ddof=1))
    if std <= 1e-12:
        return np.nan
    return float(active_roll.mean() / std)


def rolling_alpha_stability(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    end_idx: int,
    lookback: int = LB_2Y,
    window: int = LB_6M,
    step: int = 8,
) -> float:
    """
    Combines rolling alpha level and stability over trailing lookback.
    """
    if end_idx < lookback:
        return np.nan
    f = fund_nav.iloc[end_idx - lookback: end_idx + 1]
    b = bench_nav.iloc[end_idx - lookback: end_idx + 1]
    f_ret = f.pct_change().dropna()
    b_ret = b.pct_change().dropna()

    aligned = pd.concat([f_ret.rename("f"), b_ret.rename("b")], axis=1).dropna()
    if len(aligned) < window + 10:
        return np.nan

    alphas: List[float] = []
    for start in range(0, len(aligned) - window + 1, step):
        sub_f = aligned["f"].iloc[start: start + window]
        sub_b = aligned["b"].iloc[start: start + window]
        alpha, _, _ = alpha_beta_info(sub_f, sub_b)
        if pd.notna(alpha):
            alphas.append(float(alpha))

    if len(alphas) < 3:
        return np.nan

    arr = np.array(alphas, dtype=float)
    mean_alpha = float(np.mean(arr))
    std_alpha = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    pos_rate = float(np.mean(arr > 0))

    alpha_scaled = np.clip(mean_alpha / 0.08, -2.0, 2.0)
    stable_scaled = 1.0 / (1.0 + (std_alpha / 0.05))
    return float(0.40 * alpha_scaled + 0.35 * pos_rate + 0.25 * stable_scaled)


def regime_excess_metrics(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    """
    Returns:
    - bear_excess: average excess return in benchmark down weeks
    - high_vol_excess: average excess return in high-volatility weeks
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")],
        axis=1,
    ).dropna()
    if len(aligned) < 20:
        return np.nan, np.nan

    active = aligned["fund"] - aligned["bench"]

    bear_mask = aligned["bench"] < 0
    if bear_mask.sum() >= 6:
        bear_excess = float(active[bear_mask].mean())
    else:
        bear_excess = np.nan

    vol_threshold = float(np.abs(aligned["bench"]).quantile(0.75))
    high_vol_mask = np.abs(aligned["bench"]) >= vol_threshold
    if high_vol_mask.sum() >= 6:
        high_vol_excess = float(active[high_vol_mask].mean())
    else:
        high_vol_excess = np.nan

    return bear_excess, high_vol_excess


def momentum_12_minus_1(nav: pd.Series, end_idx: int, lookback: int = LB_1Y, skip: int = 4) -> float:
    """
    12-1 momentum style return: return from (t-lookback-skip) to (t-skip).
    """
    if end_idx < lookback + skip:
        return np.nan
    start = nav.iloc[end_idx - lookback - skip]
    end = nav.iloc[end_idx - skip]
    if pd.isna(start) or pd.isna(end) or start <= 0 or end <= 0:
        return np.nan
    return float(end / start - 1.0)


def cross_horizon_consistency(cagr_1y: float, cagr_3y: float, cagr_5y: float) -> float:
    """
    Higher means returns are coherent across 1Y/3Y/5Y horizons.
    """
    vals = [v for v in [cagr_1y, cagr_3y, cagr_5y] if pd.notna(v)]
    if len(vals) < 2:
        return np.nan
    arr = np.array(vals, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    scale = float(np.mean(np.abs(arr)) + 1e-9)
    dispersion = std / scale
    return float(1.0 / (1.0 + dispersion))


def build_feature_row(fund_nav: pd.Series, bench_nav: pd.Series, end_idx: int) -> Dict[str, float]:
    """
    Build factor values using only data available up to end_idx.
    """
    fund_1y = get_nav_window(fund_nav, end_idx, LB_1Y)
    bench_1y = get_nav_window(bench_nav, end_idx, LB_1Y)
    if fund_1y.empty or bench_1y.empty:
        return {}

    feat: Dict[str, float] = {name: np.nan for name in FACTOR_NAMES}

    cagr_1y = cagr_from_window(fund_1y, LB_1Y)
    cagr_3y = cagr_at(fund_nav, end_idx, LB_3Y)
    cagr_5y = cagr_at(fund_nav, end_idx, LB_5Y)

    fund_ret_1y = fund_1y.pct_change().dropna()
    bench_ret_1y = bench_1y.pct_change().dropna()
    if len(fund_ret_1y) < 12 or len(bench_ret_1y) < 12:
        return {}

    alpha, beta, info = alpha_beta_info(fund_ret_1y, bench_ret_1y)
    down_dev = downside_deviation(fund_ret_1y)
    sortino = sortino_ratio(cagr_1y, down_dev)
    up_cap, down_cap = capture_ratios(fund_ret_1y, bench_ret_1y)
    capture_spread = up_cap - down_cap if pd.notna(up_cap) and pd.notna(down_cap) else np.nan
    bear_excess, high_vol_excess = regime_excess_metrics(fund_ret_1y, bench_ret_1y)

    beta_distance = abs(beta - 1.0) if pd.notna(beta) else np.nan
    horizon_consistency = cross_horizon_consistency(cagr_1y, cagr_3y, cagr_5y)

    fund_2y = get_nav_window(fund_nav, end_idx, LB_2Y)
    if fund_2y.empty:
        max_dd_2y = np.nan
        ulcer_2y = np.nan
        rec_speed = np.nan
    else:
        max_dd_2y = max_drawdown_depth(fund_2y)
        ulcer_2y = ulcer_index(fund_2y)
        rec_speed = recovery_speed_score(fund_2y)

    roll_beat = rolling_beat_rate(fund_nav, bench_nav, end_idx=end_idx)
    roll_ir = rolling_active_ir(fund_nav, bench_nav, end_idx=end_idx)
    roll_alpha_stab = rolling_alpha_stability(fund_nav, bench_nav, end_idx=end_idx)

    fund_3m = simple_return_at(fund_nav, end_idx, LB_3M)
    bench_3m = simple_return_at(bench_nav, end_idx, LB_3M)
    fund_6m = simple_return_at(fund_nav, end_idx, LB_6M)
    bench_6m = simple_return_at(bench_nav, end_idx, LB_6M)
    mom_6m_rel = fund_6m - bench_6m if pd.notna(fund_6m) and pd.notna(bench_6m) else np.nan

    fund_12_1 = momentum_12_minus_1(fund_nav, end_idx=end_idx)
    bench_12_1 = momentum_12_minus_1(bench_nav, end_idx=end_idx)
    mom_12_1_rel = (
        fund_12_1 - bench_12_1 if pd.notna(fund_12_1) and pd.notna(bench_12_1) else np.nan
    )

    mom_3m_rel = fund_3m - bench_3m if pd.notna(fund_3m) and pd.notna(bench_3m) else np.nan
    if pd.notna(mom_3m_rel) and pd.notna(mom_12_1_rel):
        overheat_penalty = float(max(0.0, mom_3m_rel - mom_12_1_rel))
    else:
        overheat_penalty = np.nan

    feat["alpha_1y"] = alpha
    feat["info_1y"] = info
    feat["sortino_1y"] = sortino
    feat["rolling_alpha_stability_2y"] = roll_alpha_stab
    feat["rolling_active_ir_2y"] = roll_ir

    feat["up_capture_1y"] = up_cap
    feat["down_capture_1y"] = down_cap
    feat["capture_spread_1y"] = capture_spread
    feat["bear_excess_1y"] = bear_excess
    feat["high_vol_excess_1y"] = high_vol_excess

    feat["max_drawdown_2y"] = max_dd_2y
    feat["ulcer_2y"] = ulcer_2y
    feat["recovery_speed_2y"] = rec_speed
    feat["rolling_beat_rate_2y"] = roll_beat

    feat["cross_horizon_consistency"] = horizon_consistency
    feat["beta_distance_1y"] = beta_distance

    feat["momentum_6m_rel"] = mom_6m_rel
    feat["momentum_12_1_rel"] = mom_12_1_rel
    feat["overheat_penalty"] = overheat_penalty

    feat["cagr_3y"] = cagr_3y
    feat["cagr_5y"] = cagr_5y

    # diagnostics for final output
    feat["cagr_1y"] = cagr_1y
    feat["beta_1y"] = beta
    feat["volatility_1y"] = annualised_volatility(fund_ret_1y)
    feat["momentum_3m_rel"] = mom_3m_rel
    return feat


# ===================================================================
# Panel construction and scoring primitives
# ===================================================================
def add_cross_sectional_ranks(
    df: pd.DataFrame,
    group_col: str = "date",
    subsector_col: str = "subsector",
) -> pd.DataFrame:
    """
    Add percentile-rank columns for each factor.

    For Total Market, blend overall rank with within-subsector rank.
    """
    out = df.copy()
    for factor_name, higher_is_better in FACTOR_SPECS:
        rank_col = f"rank_{factor_name}"

        if group_col:
            overall = out.groupby(group_col)[factor_name].rank(pct=True, na_option="keep")
        else:
            overall = out[factor_name].rank(pct=True, na_option="keep")

        ranks = overall
        if group_col and subsector_col in out.columns:
            local = out.groupby([group_col, subsector_col])[factor_name].rank(
                pct=True, na_option="keep"
            )
            local_count = out.groupby([group_col, subsector_col])[factor_name].transform("count")
            # Tiny groups get near-zero local blend; larger groups get up to SUBSECTOR_RANK_BLEND.
            local_strength = np.clip((local_count.astype(float) - 4.0) / 8.0, 0.0, 1.0)
            local_strength = local_strength * SUBSECTOR_RANK_BLEND
            ranks = overall * (1.0 - local_strength) + local * local_strength

        if not higher_is_better:
            ranks = 1.0 - ranks
        out[rank_col] = ranks * 100.0
    return out


def score_from_rank_matrix(rank_matrix: np.ndarray, valid_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted score with missing-data-aware normalization."""
    matrix_filled = np.nan_to_num(rank_matrix, nan=0.0)
    weighted_sum = matrix_filled @ weights
    applied_weight = valid_matrix @ weights
    score = np.where(applied_weight > 0, weighted_sum / applied_weight, np.nan)
    return score


def weight_entropy(weights: np.ndarray) -> float:
    """Normalized entropy in [0, 1]."""
    eps = 1e-12
    entropy = -np.sum(weights * np.log(weights + eps))
    return float(entropy / np.log(len(weights)))


def group_weight_sum(weights: np.ndarray, factors: set) -> float:
    return float(sum(weights[FACTOR_INDEX[name]] for name in factors))


def rebalance_weights_with_caps(weights: np.ndarray, max_cap: float = MAX_FACTOR_WEIGHT) -> np.ndarray:
    """Re-normalize to sum 1 while enforcing per-factor cap."""
    w = np.clip(np.array(weights, dtype=float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    for _ in range(12):
        over = w > max_cap
        if not over.any():
            break
        excess = float((w[over] - max_cap).sum())
        w[over] = max_cap
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 1e-12:
            w = np.ones_like(w) / len(w)
            break
        w[under] += excess * (w[under] / under_sum)
        w = w / w.sum()
    return w


def is_weight_vector_valid(weights: np.ndarray) -> bool:
    """Constrain search to diversified and balanced allocations."""
    if len(weights) != len(FACTOR_NAMES):
        return False
    if np.any(weights < 0):
        return False
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        return False
    if float(weights.max()) > MAX_FACTOR_WEIGHT:
        return False

    skill = group_weight_sum(weights, SKILL_FACTORS)
    cycle = group_weight_sum(weights, CYCLE_FACTORS)
    resilience = group_weight_sum(weights, RESILIENCE_FACTORS)
    structure = group_weight_sum(weights, STRUCTURE_FACTORS)
    momentum = group_weight_sum(weights, MOMENTUM_FACTORS)
    anchors = group_weight_sum(weights, ANCHOR_FACTORS)

    if not (0.20 <= skill <= 0.45):
        return False
    if not (0.16 <= cycle <= 0.36):
        return False
    if not (0.15 <= resilience <= 0.34):
        return False
    if not (0.06 <= structure <= 0.20):
        return False
    if not (0.08 <= momentum <= 0.24):
        return False
    if not (0.08 <= anchors <= 0.22):
        return False

    if int((weights >= 0.02).sum()) < 8:
        return False
    if weight_entropy(weights) < 0.78:
        return False
    return True


def build_walk_forward_folds(
    dev_dates: List[pd.Timestamp],
    min_train_periods: int = 12,
    val_periods: int = 8,
    step: int = 8,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """Build expanding-window walk-forward folds."""
    folds: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []
    start = min_train_periods
    n = len(dev_dates)

    while start + val_periods <= n:
        tr = dev_dates[:start]
        va = dev_dates[start: start + val_periods]
        if len(tr) >= 8 and len(va) >= 3:
            folds.append((tr, va))
        start += step

    if len(folds) < 2 and n >= 14:
        split1 = max(8, int(n * 0.58))
        split2 = max(split1 + 3, int(n * 0.78))
        split2 = min(split2, n - 2)
        folds = [
            (dev_dates[:split1], dev_dates[split1:split2]),
            (dev_dates[:split2], dev_dates[split2:]),
        ]
    return folds


def build_historical_panel(funds: List[Dict[str, Any]], bench_nav: pd.Series) -> pd.DataFrame:
    """
    Build time-series cross-sectional panel:
    each row = (date, fund) factors + next-1Y realized return target.
    """
    rows: List[Dict[str, Any]] = []

    eval_start_idx = LB_1Y
    eval_end_idx = len(bench_nav) - FORWARD_HORIZON_WEEKS - 1

    for end_idx in range(eval_start_idx, eval_end_idx + 1, EVAL_STEP_WEEKS):
        date_rows: List[Dict[str, Any]] = []
        snap_date = bench_nav.index[end_idx]
        bench_now = bench_nav.iat[end_idx]
        bench_future = bench_nav.iat[end_idx + FORWARD_HORIZON_WEEKS]
        if pd.isna(bench_now) or pd.isna(bench_future) or bench_now <= 0 or bench_future <= 0:
            continue
        bench_forward = float(bench_future / bench_now - 1.0)

        for fund in funds:
            nav = fund["aligned_nav"]
            nav_now = nav.iat[end_idx]
            nav_future = nav.iat[end_idx + FORWARD_HORIZON_WEEKS]
            if pd.isna(nav_now) or pd.isna(nav_future) or nav_now <= 0 or nav_future <= 0:
                continue

            feat = build_feature_row(nav, bench_nav, end_idx)
            if not feat:
                continue

            fund_forward = float(nav_future / nav_now - 1.0)
            row = {
                "date": snap_date,
                "mfId": fund["mfId"],
                "name": fund["name"],
                "subsector": fund["subsector"],
                "aum": fund["aum"],
                "data_weeks": fund["data_weeks"],
                "data_days": fund["data_days"],
                "history_years": fund["history_years"],
                "forward_1y_return": fund_forward,
                "forward_1y_excess": fund_forward - bench_forward,
            }
            row.update(feat)
            date_rows.append(row)

        if len(date_rows) >= MIN_FUNDS_PER_DATE:
            rows.extend(date_rows)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["date", "mfId"]).reset_index(drop=True)
    return panel


def split_dates(dates: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]:
    """Chronological train/validation/test split."""
    n = len(dates)
    if n < 15:
        raise RuntimeError(f"Need at least 15 evaluation dates for robust tuning, found {n}.")

    train_end = max(8, int(n * 0.60))
    val_end = max(train_end + 3, int(n * 0.82))
    val_end = min(val_end, n - 2)
    if val_end <= train_end:
        val_end = train_end + 1

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    if len(val_dates) < 2 or len(test_dates) < 2:
        raise RuntimeError("Insufficient date windows to create non-trivial train/val/test splits.")

    return train_dates, val_dates, test_dates


def make_segment(panel_ranked: pd.DataFrame, dates: List[pd.Timestamp], label: str) -> Dict[str, Any]:
    seg_df = panel_ranked[panel_ranked["date"].isin(dates)].copy()
    seg_df = seg_df.sort_values(["date", "mfId"]).reset_index(drop=True)

    if seg_df.empty:
        return {
            "label": label,
            "df": seg_df,
            "rank_matrix": np.empty((0, len(FACTOR_NAMES))),
            "valid_matrix": np.empty((0, len(FACTOR_NAMES))),
            "y_return": np.empty((0,)),
            "y_excess": np.empty((0,)),
            "subsector": np.empty((0,), dtype=object),
            "groups": [],
        }

    rank_matrix = seg_df[RANK_COLS].to_numpy(dtype=float)
    valid_matrix = (~np.isnan(rank_matrix)).astype(float)
    y_return = seg_df["forward_1y_return"].to_numpy(dtype=float)
    y_excess = seg_df["forward_1y_excess"].to_numpy(dtype=float)
    subsector = seg_df["subsector"].to_numpy(dtype=object)

    groups: List[np.ndarray] = []
    grouped = seg_df.groupby("date", sort=True).indices
    for date_key in sorted(grouped.keys()):
        groups.append(np.array(grouped[date_key], dtype=int))

    return {
        "label": label,
        "df": seg_df,
        "rank_matrix": rank_matrix,
        "valid_matrix": valid_matrix,
        "y_return": y_return,
        "y_excess": y_excess,
        "subsector": subsector,
        "groups": groups,
    }


def evaluate_segment(
    segment: Dict[str, Any],
    weights: np.ndarray,
    top_k: int = TOP_K,
    collect_records: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate one segment using fixed factor weights.
    """
    if segment["rank_matrix"].size == 0:
        metrics = {
            "n_periods": 0.0,
            "mean_ic": 0.0,
            "ic_ir": 0.0,
            "mean_top_return": 0.0,
            "mean_top_excess": 0.0,
            "hit_rate": 0.0,
            "downside_excess": 0.0,
            "vol_excess": 0.0,
            "avg_turnover": 0.0,
            "mean_top_subsector_concentration": 0.0,
            "subsector_concentration_penalty": 0.0,
            "objective": -1e9,
        }
        return metrics, pd.DataFrame()

    scores = score_from_rank_matrix(segment["rank_matrix"], segment["valid_matrix"], weights)
    y_return = segment["y_return"]
    y_excess = segment["y_excess"]
    subsector = segment["subsector"]
    seg_df = segment["df"]

    ic_values: List[float] = []
    top_returns: List[float] = []
    top_excesses: List[float] = []
    turnover_values: List[float] = []
    top_subsector_concentration: List[float] = []
    records: List[Dict[str, Any]] = []
    prev_top_set = None

    for idx in segment["groups"]:
        s = scores[idx]
        r = y_return[idx]
        e = y_excess[idx]
        sub = subsector[idx]

        valid = (~np.isnan(s)) & (~np.isnan(r)) & (~np.isnan(e))
        if valid.sum() < MIN_FUNDS_PER_DATE:
            continue

        s_valid = s[valid]
        r_valid = r[valid]
        e_valid = e[valid]
        sub_valid = sub[valid]
        if len(s_valid) < 3:
            continue

        ic = np.nan
        if np.nanstd(s_valid) > 1e-12 and np.nanstd(e_valid) > 1e-12:
            ic = pd.Series(s_valid).corr(pd.Series(e_valid), method="spearman")
            if pd.notna(ic):
                ic_values.append(float(ic))

        k = min(top_k, len(s_valid))
        if k == len(s_valid):
            top_local_idx = np.arange(len(s_valid), dtype=int)
        else:
            top_local_idx = np.argpartition(s_valid, len(s_valid) - k)[-k:]

        top_return = float(np.nanmean(r_valid[top_local_idx]))
        top_excess = float(np.nanmean(e_valid[top_local_idx]))
        top_returns.append(top_return)
        top_excesses.append(top_excess)

        top_sub = pd.Series(sub_valid[top_local_idx])
        if not top_sub.empty:
            conc = float(top_sub.value_counts(normalize=True).iloc[0])
        else:
            conc = np.nan
        if pd.notna(conc):
            top_subsector_concentration.append(conc)

        valid_global = idx[valid]
        top_global = valid_global[top_local_idx]
        top_funds_list = seg_df.iloc[top_global]["mfId"].tolist()
        top_set = set(top_funds_list)
        if prev_top_set is not None and k > 0:
            overlap = len(top_set.intersection(prev_top_set))
            turnover = 1.0 - (overlap / float(k))
            turnover_values.append(turnover)
        prev_top_set = top_set

        if collect_records:
            top_subsector = ""
            if not top_sub.empty:
                top_subsector = str(top_sub.value_counts().index[0])
            records.append(
                {
                    "segment": segment["label"],
                    "date": seg_df.iloc[idx[0]]["date"],
                    "n_funds": int(valid.sum()),
                    "ic": ic if pd.notna(ic) else np.nan,
                    "top_k_return": top_return,
                    "top_k_excess": top_excess,
                    "top_subsector": top_subsector,
                    "top_subsector_concentration": conc,
                    "top_funds": ",".join(top_funds_list),
                }
            )

    if not top_excesses:
        metrics = {
            "n_periods": 0.0,
            "mean_ic": 0.0,
            "ic_ir": 0.0,
            "mean_top_return": 0.0,
            "mean_top_excess": 0.0,
            "hit_rate": 0.0,
            "downside_excess": 0.0,
            "vol_excess": 0.0,
            "avg_turnover": 0.0,
            "mean_top_subsector_concentration": 0.0,
            "subsector_concentration_penalty": 0.0,
            "objective": -1e9,
        }
        return metrics, pd.DataFrame(records)

    top_ex_arr = np.array(top_excesses, dtype=float)
    top_ret_arr = np.array(top_returns, dtype=float)

    mean_ic = float(np.nanmean(ic_values)) if ic_values else 0.0
    if len(ic_values) > 1:
        ic_ir = float(mean_ic / (np.nanstd(ic_values, ddof=1) + 1e-12))
    else:
        ic_ir = 0.0

    mean_top_return = float(np.nanmean(top_ret_arr))
    mean_top_excess = float(np.nanmean(top_ex_arr))
    hit_rate = float(np.mean(top_ex_arr > 0))
    downside_excess = float(-np.mean(np.minimum(top_ex_arr, 0.0)))
    vol_excess = float(np.std(top_ex_arr, ddof=1)) if len(top_ex_arr) > 1 else 0.0
    avg_turnover = float(np.mean(turnover_values)) if turnover_values else 0.0

    if top_subsector_concentration:
        mean_conc = float(np.mean(top_subsector_concentration))
    else:
        mean_conc = 0.0
    concentration_penalty = float(max(0.0, mean_conc - 0.60))

    objective = (
        0.50 * (mean_top_excess * 100.0)
        + 0.17 * (mean_ic * 100.0)
        + 0.12 * ((hit_rate - 0.5) * 100.0)
        + 0.08 * (mean_top_return * 100.0)
        - 0.10 * (downside_excess * 100.0)
        - 0.06 * (vol_excess * 100.0)
        - 0.07 * (avg_turnover * 100.0)
        - 0.08 * (concentration_penalty * 100.0)
    )

    metrics = {
        "n_periods": float(len(top_ex_arr)),
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "mean_top_return": mean_top_return,
        "mean_top_excess": mean_top_excess,
        "hit_rate": hit_rate,
        "downside_excess": downside_excess,
        "vol_excess": vol_excess,
        "avg_turnover": avg_turnover,
        "mean_top_subsector_concentration": mean_conc,
        "subsector_concentration_penalty": concentration_penalty,
        "objective": float(objective),
    }
    return metrics, pd.DataFrame(records)


def sample_weight_vector(rng: np.random.Generator, n_factors: int) -> np.ndarray:
    """Sample constrained, non-negative weights that sum to 1."""
    while True:
        w = rng.dirichlet(np.full(n_factors, 0.95))
        if is_weight_vector_valid(w):
            return w


def seed_candidates() -> List[np.ndarray]:
    """Deterministic priors before random search."""
    n = len(FACTOR_NAMES)
    idx = {name: i for i, name in enumerate(FACTOR_NAMES)}

    def vec(mapping: Dict[str, float]) -> np.ndarray:
        w = np.zeros(n, dtype=float)
        for name, val in mapping.items():
            w[idx[name]] = val
        if w.sum() <= 0:
            w[:] = 1.0 / n
        else:
            w /= w.sum()
        return w

    raw = [
        np.ones(n, dtype=float) / n,
        vec(
            {
                "alpha_1y": 0.08,
                "info_1y": 0.08,
                "sortino_1y": 0.07,
                "rolling_alpha_stability_2y": 0.07,
                "rolling_active_ir_2y": 0.06,
                "up_capture_1y": 0.06,
                "down_capture_1y": 0.07,
                "capture_spread_1y": 0.05,
                "bear_excess_1y": 0.05,
                "high_vol_excess_1y": 0.04,
                "max_drawdown_2y": 0.07,
                "ulcer_2y": 0.07,
                "recovery_speed_2y": 0.06,
                "rolling_beat_rate_2y": 0.05,
                "cross_horizon_consistency": 0.05,
                "beta_distance_1y": 0.04,
                "momentum_6m_rel": 0.04,
                "momentum_12_1_rel": 0.03,
                "overheat_penalty": 0.03,
                "cagr_3y": 0.02,
                "cagr_5y": 0.01,
            }
        ),
        vec(
            {
                "alpha_1y": 0.06,
                "info_1y": 0.06,
                "sortino_1y": 0.06,
                "rolling_alpha_stability_2y": 0.05,
                "rolling_active_ir_2y": 0.05,
                "up_capture_1y": 0.08,
                "down_capture_1y": 0.08,
                "capture_spread_1y": 0.08,
                "bear_excess_1y": 0.06,
                "high_vol_excess_1y": 0.05,
                "max_drawdown_2y": 0.05,
                "ulcer_2y": 0.05,
                "recovery_speed_2y": 0.04,
                "rolling_beat_rate_2y": 0.04,
                "cross_horizon_consistency": 0.03,
                "beta_distance_1y": 0.03,
                "momentum_6m_rel": 0.06,
                "momentum_12_1_rel": 0.05,
                "overheat_penalty": 0.04,
                "cagr_3y": 0.03,
                "cagr_5y": 0.02,
            }
        ),
        vec(
            {
                "alpha_1y": 0.07,
                "info_1y": 0.07,
                "sortino_1y": 0.06,
                "rolling_alpha_stability_2y": 0.06,
                "rolling_active_ir_2y": 0.05,
                "up_capture_1y": 0.05,
                "down_capture_1y": 0.08,
                "capture_spread_1y": 0.05,
                "bear_excess_1y": 0.07,
                "high_vol_excess_1y": 0.06,
                "max_drawdown_2y": 0.08,
                "ulcer_2y": 0.08,
                "recovery_speed_2y": 0.07,
                "rolling_beat_rate_2y": 0.06,
                "cross_horizon_consistency": 0.05,
                "beta_distance_1y": 0.05,
                "momentum_6m_rel": 0.03,
                "momentum_12_1_rel": 0.02,
                "overheat_penalty": 0.03,
                "cagr_3y": 0.03,
                "cagr_5y": 0.03,
            }
        ),
    ]

    candidates = [w for w in raw if is_weight_vector_valid(w)]
    return candidates


def tune_weights(cv_folds: List[Dict[str, Any]]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Tune weights with walk-forward CV and ensemble averaging.
    """
    if not cv_folds:
        raise RuntimeError("No CV folds supplied for tuning.")

    rng = np.random.default_rng(RANDOM_SEED)
    n_factors = len(FACTOR_NAMES)

    # Stage 1: broad random search.
    candidates = seed_candidates()
    while len(candidates) < SEARCH_TRIALS:
        candidates.append(sample_weight_vector(rng, n_factors))

    # Lightweight pre-screen on most recent fold.
    screening_fold = cv_folds[-1]
    screening_scores: List[float] = []
    for w in candidates:
        m_tr, _ = evaluate_segment(screening_fold["train"], w, top_k=TOP_K, collect_records=False)
        m_va, _ = evaluate_segment(screening_fold["val"], w, top_k=TOP_K, collect_records=False)
        score = (
            m_va["objective"]
            + 0.15 * m_tr["objective"]
            + 0.20 * weight_entropy(w)
            - 0.05 * (m_va["subsector_concentration_penalty"] * 100.0)
        )
        screening_scores.append(score)

    pre_top = min(len(candidates), max(24, SEARCH_TRIALS // 7))
    pre_idx = np.argsort(screening_scores)[-pre_top:]

    # Stage 2: local refinement.
    refined: List[np.ndarray] = []
    for i in pre_idx:
        base = candidates[i]
        refined.append(base)
        for _ in range(REFINE_MUTATIONS_PER_BASE):
            noise = sample_weight_vector(rng, n_factors)
            mixed = 0.84 * base + 0.16 * noise
            mixed = rebalance_weights_with_caps(mixed, max_cap=MAX_FACTOR_WEIGHT)
            mixed = mixed / mixed.sum()
            if is_weight_vector_valid(mixed):
                refined.append(mixed)

    # De-duplicate.
    unique: Dict[Tuple[float, ...], np.ndarray] = {}
    for w in refined:
        key = tuple(np.round(w, 4))
        if key not in unique and is_weight_vector_valid(w):
            unique[key] = w
    refined = list(unique.values())
    if not refined:
        raise RuntimeError("No valid refined candidates generated during tuning.")

    trial_rows: List[Dict[str, Any]] = []
    for trial_num, w in enumerate(refined, start=1):
        train_objs: List[float] = []
        val_objs: List[float] = []
        val_excesses: List[float] = []
        val_hits: List[float] = []
        val_ics: List[float] = []
        val_turnovers: List[float] = []
        val_conc_penalties: List[float] = []

        for fold in cv_folds:
            m_tr, _ = evaluate_segment(fold["train"], w, top_k=TOP_K, collect_records=False)
            m_va, _ = evaluate_segment(fold["val"], w, top_k=TOP_K, collect_records=False)
            train_objs.append(m_tr["objective"])
            val_objs.append(m_va["objective"])
            val_excesses.append(m_va["mean_top_excess"])
            val_hits.append(m_va["hit_rate"])
            val_ics.append(m_va["mean_ic"])
            val_turnovers.append(m_va["avg_turnover"])
            val_conc_penalties.append(m_va["subsector_concentration_penalty"])

        train_obj_mean = float(np.mean(train_objs))
        val_obj_mean = float(np.mean(val_objs))
        val_obj_std = float(np.std(val_objs, ddof=1)) if len(val_objs) > 1 else 0.0
        val_excess_mean = float(np.mean(val_excesses))
        val_excess_std = float(np.std(val_excesses, ddof=1)) if len(val_excesses) > 1 else 0.0
        val_excess_worst = float(np.min(val_excesses))
        val_hit_mean = float(np.mean(val_hits))
        val_ic_mean = float(np.mean(val_ics))
        val_turnover_mean = float(np.mean(val_turnovers))
        val_conc_penalty_mean = float(np.mean(val_conc_penalties))
        entropy = weight_entropy(w)

        selection_score = (
            val_obj_mean
            - 0.35 * val_obj_std
            + 0.20 * train_obj_mean
            + 0.20 * entropy
            + 0.08 * (val_hit_mean - 0.5) * 100.0
            + 0.08 * val_excess_worst * 100.0
            - 0.06 * val_excess_std * 100.0
            + 0.05 * val_ic_mean * 100.0
            - 0.08 * val_turnover_mean * 100.0
            - 0.10 * val_conc_penalty_mean * 100.0
        )

        row: Dict[str, Any] = {
            "trial": trial_num,
            "selection_score": selection_score,
            "cv_mean_train_obj": train_obj_mean,
            "cv_mean_val_obj": val_obj_mean,
            "cv_std_val_obj": val_obj_std,
            "cv_mean_val_top_excess_pct": val_excess_mean * 100.0,
            "cv_std_val_top_excess_pct": val_excess_std * 100.0,
            "cv_worst_val_top_excess_pct": val_excess_worst * 100.0,
            "cv_mean_val_hit_rate_pct": val_hit_mean * 100.0,
            "cv_mean_val_ic": val_ic_mean,
            "cv_mean_val_turnover_pct": val_turnover_mean * 100.0,
            "cv_mean_val_subsector_conc_penalty_pct": val_conc_penalty_mean * 100.0,
            "entropy": entropy,
        }
        for i, factor_name in enumerate(FACTOR_NAMES):
            row[f"w_{factor_name}"] = w[i]
        trial_rows.append(row)

    trials_df = pd.DataFrame(trial_rows).sort_values("selection_score", ascending=False)
    if trials_df.empty:
        raise RuntimeError("Weight tuning produced no trial rows.")

    eligible = trials_df[
        (trials_df["cv_worst_val_top_excess_pct"] > -6.0)
        & (trials_df["cv_mean_val_subsector_conc_penalty_pct"] < 14.0)
    ]
    if eligible.empty:
        eligible = trials_df
    top_trials = eligible.head(TOP_TRIALS_FOR_ENSEMBLE).copy()
    if top_trials.empty:
        raise RuntimeError("No eligible top trials to build ensemble weights.")

    weight_matrix = top_trials[[f"w_{name}" for name in FACTOR_NAMES]].to_numpy(dtype=float)
    scores = top_trials["selection_score"].to_numpy(dtype=float)
    scaled = scores - scores.max()
    blend = np.exp(scaled / 2.8)
    blend = blend / blend.sum()
    ensemble_weights = np.average(weight_matrix, axis=0, weights=blend)
    ensemble_weights = rebalance_weights_with_caps(ensemble_weights, max_cap=MAX_FACTOR_WEIGHT)
    ensemble_weights = ensemble_weights / ensemble_weights.sum()

    if not is_weight_vector_valid(ensemble_weights):
        best_single = top_trials.iloc[0][[f"w_{name}" for name in FACTOR_NAMES]].to_numpy(dtype=float)
        best_single = rebalance_weights_with_caps(best_single, max_cap=MAX_FACTOR_WEIGHT)
        best_single = best_single / best_single.sum()
        ensemble_weights = best_single

    return ensemble_weights, trials_df


def detect_market_regime(bench_nav: pd.Series) -> Dict[str, Any]:
    """Classify current benchmark regime for mild live-score weight tilting."""
    end_idx = len(bench_nav) - 1
    if end_idx < LB_1Y:
        return {
            "regime": "mixed",
            "ret_3m": np.nan,
            "ret_6m": np.nan,
            "vol_ratio": np.nan,
            "dd_6m": np.nan,
        }

    ret_3m = simple_return_at(bench_nav, end_idx, LB_3M)
    ret_6m = simple_return_at(bench_nav, end_idx, LB_6M)

    ret = bench_nav.pct_change().dropna()
    vol_recent = annualised_volatility(ret.tail(LB_3M))
    vol_trailing = annualised_volatility(ret.tail(LB_1Y))
    if pd.notna(vol_recent) and pd.notna(vol_trailing) and vol_trailing > 1e-12:
        vol_ratio = float(vol_recent / vol_trailing)
    else:
        vol_ratio = np.nan

    bench_6m = get_nav_window(bench_nav, end_idx, LB_6M)
    dd_6m = max_drawdown_depth(bench_6m) if not bench_6m.empty else np.nan

    if pd.notna(ret_6m) and pd.notna(ret_3m):
        if ret_6m > 0.10 and ret_3m > 0 and (pd.isna(vol_ratio) or vol_ratio < 1.05) and (
            pd.isna(dd_6m) or dd_6m < 0.08
        ):
            regime = "bull_calm"
        elif ret_6m < -0.03 or ret_3m < -0.04 or (pd.notna(dd_6m) and dd_6m > 0.12):
            regime = "correction"
        elif pd.notna(vol_ratio) and vol_ratio > 1.22:
            regime = "high_vol"
        else:
            regime = "mixed"
    else:
        regime = "mixed"

    return {
        "regime": regime,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "vol_ratio": vol_ratio,
        "dd_6m": dd_6m,
    }


def apply_regime_tilt(weights: np.ndarray, regime: str) -> np.ndarray:
    """
    Apply a conservative market-regime tilt to tuned weights for live ranking.
    """
    w = np.array(weights, dtype=float).copy()
    multipliers = np.ones(len(FACTOR_NAMES), dtype=float)

    if regime == "bull_calm":
        for f in AGGRESSIVE_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 1.10
        for f in DEFENSIVE_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 0.93
        for f in ANCHOR_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 0.95
    elif regime in {"correction", "high_vol"}:
        for f in AGGRESSIVE_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 0.88
        for f in DEFENSIVE_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 1.12
        for f in ANCHOR_TILT_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 1.05
    else:
        for f in SKILL_FACTORS:
            multipliers[FACTOR_INDEX[f]] *= 1.03

    w = w * multipliers
    w = rebalance_weights_with_caps(w, max_cap=MAX_FACTOR_WEIGHT)
    w = w / w.sum()
    return w


def score_current_snapshot(
    funds: List[Dict[str, Any]],
    bench_nav: pd.Series,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Compute latest-date ranking with history-aware reliability adjustment.
    """
    latest_idx = len(bench_nav) - 1
    rows: List[Dict[str, Any]] = []

    for fund in funds:
        nav = fund["aligned_nav"]
        latest_nav = nav.iat[latest_idx]
        if pd.isna(latest_nav):
            continue

        feat = build_feature_row(nav, bench_nav, latest_idx)
        if not feat:
            continue

        row = {
            "mfId": fund["mfId"],
            "name": fund["name"],
            "subsector": fund["subsector"],
            "aum": fund["aum"],
            "data_weeks": fund["data_weeks"],
            "data_days": fund["data_days"],
            "history_years": fund["history_years"],
            "date": bench_nav.index[latest_idx],
        }
        row.update(feat)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    snapshot = pd.DataFrame(rows)
    snapshot = add_cross_sectional_ranks(snapshot, group_col="date", subsector_col="subsector")

    rank_matrix = snapshot[RANK_COLS].to_numpy(dtype=float)
    valid_matrix = (~np.isnan(rank_matrix)).astype(float)
    score_raw = score_from_rank_matrix(rank_matrix, valid_matrix, weights)
    snapshot["raw_score"] = score_raw

    snapshot["factor_coverage"] = valid_matrix.mean(axis=1)
    history_confidence = np.clip(snapshot["data_weeks"] / HISTORY_CONFIDENCE_CAP_WEEKS, 0.0, 1.0)
    combined_confidence = (
        HISTORY_LENGTH_WEIGHT * history_confidence
        + FACTOR_COVERAGE_WEIGHT * snapshot["factor_coverage"]
    )
    snapshot["history_reliability"] = (
        MIN_HISTORY_RELIABILITY
        + (MAX_HISTORY_RELIABILITY - MIN_HISTORY_RELIABILITY) * combined_confidence
    )

    snapshot["score"] = (
        snapshot["raw_score"] * snapshot["history_reliability"]
        + HISTORY_NEUTRAL_SCORE * (1.0 - snapshot["history_reliability"])
    )

    snapshot = snapshot.sort_values("score", ascending=False).reset_index(drop=True)
    snapshot["rank"] = np.arange(1, len(snapshot) + 1, dtype=int)
    return snapshot.drop(columns=["date"])


def pct_fmt(x: float) -> str:
    return f"{x * 100:.2f}" if pd.notna(x) else ""


def ratio_fmt(x: float) -> str:
    return f"{x:.3f}" if pd.notna(x) else ""


def num_fmt(x: float) -> str:
    return f"{x:.2f}" if pd.notna(x) else ""


def factor_group_name(factor: str) -> str:
    if factor in SKILL_FACTORS:
        return "skill"
    if factor in CYCLE_FACTORS:
        return "cycle"
    if factor in RESILIENCE_FACTORS:
        return "resilience"
    if factor in STRUCTURE_FACTORS:
        return "structure"
    if factor in MOMENTUM_FACTORS:
        return "momentum"
    if factor in ANCHOR_FACTORS:
        return "anchor"
    return "other"


def main() -> None:
    print("\n" + "=" * 92)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM - CODEX")
    print("  Method: Walk-forward tuned multi-factor model with regime tilt")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print("  Subsectors: " + ", ".join(SUBSECTORS))
    print("=" * 92)

    provider = MfDataProvider()

    # --- Benchmark ---
    logger.info("Loading benchmark index data...")
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_chart(bench_df)
    if len(bench_nav) < LB_5Y + FORWARD_HORIZON_WEEKS + 10:
        raise RuntimeError(
            f"Benchmark history too short ({len(bench_nav)} weeks) for robust 5Y+forward analysis."
        )
    print(
        f"\n  Benchmark history: {len(bench_nav)} weeks "
        f"({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})"
    )

    # --- Funds ---
    df_all = provider.list_all_mf()
    df_tm = df_all[df_all["subsector"].isin(SUBSECTORS)].copy()
    print(f"  Candidate funds in Total Market universe: {len(df_tm)}")
    for sub in SUBSECTORS:
        print(f"    - {sub:20s}: {(df_tm['subsector'] == sub).sum()}")

    funds: List[Dict[str, Any]] = []
    skipped = 0
    for _, row in df_tm.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        subsector = row["subsector"]
        aum = float(row.get("aum", 0) or 0)

        try:
            chart = provider.get_mf_chart(mf_id)
            nav_raw = clean_nav_chart(chart)
            if len(nav_raw) < MIN_WEEKS_TO_CONSIDER_FUND:
                skipped += 1
                continue

            aligned_nav = nav_raw.reindex(bench_nav.index).ffill()
            first_ts = nav_raw.index.min()
            last_ts = nav_raw.index.max()
            data_days = int((last_ts - first_ts).days) if pd.notna(first_ts) and pd.notna(last_ts) else 0
            history_years = float(data_days / 365.25) if data_days > 0 else 0.0
            stale_days = int((bench_nav.index.max() - last_ts).days) if pd.notna(last_ts) else 10**9
            stale_weeks = stale_days / 7.0
            if stale_weeks > MAX_STALE_WEEKS:
                skipped += 1
                continue
            funds.append(
                {
                    "mfId": mf_id,
                    "name": name,
                    "subsector": subsector,
                    "aum": aum,
                    "data_weeks": int(len(nav_raw)),
                    "data_days": data_days,
                    "history_years": history_years,
                    "aligned_nav": aligned_nav,
                }
            )
        except Exception as exc:
            logger.error("Failed to load %s (%s): %s", mf_id, name, exc)
            skipped += 1

    if not funds:
        raise RuntimeError("No eligible Total Market funds available for analysis.")

    print(f"  Eligible funds: {len(funds)} (skipped: {skipped})")

    # --- Build panel for tuning/backtesting ---
    panel = build_historical_panel(funds, bench_nav)
    if panel.empty:
        raise RuntimeError("Historical panel is empty; cannot tune model.")

    unique_dates = sorted(panel["date"].unique().tolist())
    print(f"  Backtest snapshots: {len(unique_dates)}")
    print(f"  Panel rows: {len(panel)}")

    train_dates, val_dates, test_dates = split_dates(unique_dates)
    print(f"  Split dates -> train:{len(train_dates)}  val:{len(val_dates)}  test:{len(test_dates)}")

    panel_ranked = add_cross_sectional_ranks(panel, group_col="date", subsector_col="subsector")

    dev_dates = train_dates + val_dates
    cv_pairs = build_walk_forward_folds(
        dev_dates,
        min_train_periods=max(12, len(dev_dates) // 2),
        val_periods=max(6, len(dev_dates) // 5),
        step=max(6, len(dev_dates) // 8),
    )
    if len(cv_pairs) < 2:
        raise RuntimeError("Insufficient walk-forward folds for robust tuning.")
    if len(cv_pairs) > MAX_CV_FOLDS:
        cv_pairs = cv_pairs[-MAX_CV_FOLDS:]
    print(f"  Walk-forward CV folds: {len(cv_pairs)}")

    train_segment = make_segment(panel_ranked, train_dates, label="train")
    val_segment = make_segment(panel_ranked, val_dates, label="validation")
    test_segment = make_segment(panel_ranked, test_dates, label="test")
    full_segment = make_segment(panel_ranked, unique_dates, label="full")

    cv_folds: List[Dict[str, Any]] = []
    for fold_idx, (cv_train_dates, cv_val_dates) in enumerate(cv_pairs, start=1):
        cv_folds.append(
            {
                "name": f"cv_{fold_idx}",
                "train": make_segment(panel_ranked, cv_train_dates, label=f"cv_{fold_idx}_train"),
                "val": make_segment(panel_ranked, cv_val_dates, label=f"cv_{fold_idx}_val"),
            }
        )

    # --- Tune static base weights ---
    print("\n  Tuning factor weights with walk-forward cross-validation...")
    base_weights, trials_df = tune_weights(cv_folds)

    # --- Detect current regime and apply live tilt ---
    regime_info = detect_market_regime(bench_nav)
    live_weights = apply_regime_tilt(base_weights, regime=regime_info["regime"])

    # --- Backtest diagnostics use static base (pure tuned model) ---
    train_metrics, train_records = evaluate_segment(train_segment, base_weights, top_k=TOP_K, collect_records=True)
    val_metrics, val_records = evaluate_segment(val_segment, base_weights, top_k=TOP_K, collect_records=True)
    test_metrics, test_records = evaluate_segment(test_segment, base_weights, top_k=TOP_K, collect_records=True)
    full_metrics, full_records = evaluate_segment(full_segment, base_weights, top_k=TOP_K, collect_records=True)

    # --- Live ranking uses regime-tilted weights ---
    current_ranked = score_current_snapshot(funds, bench_nav, live_weights)
    if current_ranked.empty:
        raise RuntimeError("Failed to build current ranking from tuned model.")

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame()
    output["mfId"] = current_ranked["mfId"]
    output["name"] = current_ranked["name"]
    output["rank"] = current_ranked["rank"]
    output["score"] = current_ranked["score"].round(2)
    output["data_days"] = current_ranked["data_days"]
    output["subsector"] = current_ranked["subsector"]

    output["cagr_1y"] = current_ranked["cagr_1y"].apply(pct_fmt)
    output["cagr_3y"] = current_ranked["cagr_3y"].apply(pct_fmt)
    output["cagr_5y"] = current_ranked["cagr_5y"].apply(pct_fmt)
    output["alpha_1y"] = current_ranked["alpha_1y"].apply(pct_fmt)
    output["beta_1y"] = current_ranked["beta_1y"].apply(ratio_fmt)
    output["beta_distance_1y"] = current_ranked["beta_distance_1y"].apply(num_fmt)
    output["info_1y"] = current_ranked["info_1y"].apply(ratio_fmt)
    output["sortino_1y"] = current_ranked["sortino_1y"].apply(ratio_fmt)

    output["up_capture_1y"] = current_ranked["up_capture_1y"].apply(ratio_fmt)
    output["down_capture_1y"] = current_ranked["down_capture_1y"].apply(ratio_fmt)
    output["capture_spread_1y"] = current_ranked["capture_spread_1y"].apply(ratio_fmt)
    output["bear_excess_1y"] = current_ranked["bear_excess_1y"].apply(pct_fmt)
    output["high_vol_excess_1y"] = current_ranked["high_vol_excess_1y"].apply(pct_fmt)

    output["max_drawdown_2y"] = current_ranked["max_drawdown_2y"].apply(pct_fmt)
    output["ulcer_2y"] = current_ranked["ulcer_2y"].apply(num_fmt)
    output["recovery_speed_2y"] = current_ranked["recovery_speed_2y"].apply(ratio_fmt)
    output["rolling_beat_rate_2y"] = current_ranked["rolling_beat_rate_2y"].apply(pct_fmt)
    output["rolling_alpha_stability_2y"] = current_ranked["rolling_alpha_stability_2y"].apply(num_fmt)
    output["rolling_active_ir_2y"] = current_ranked["rolling_active_ir_2y"].apply(ratio_fmt)
    output["cross_horizon_consistency"] = current_ranked["cross_horizon_consistency"].apply(ratio_fmt)

    output["momentum_3m_rel"] = current_ranked["momentum_3m_rel"].apply(pct_fmt)
    output["momentum_6m_rel"] = current_ranked["momentum_6m_rel"].apply(pct_fmt)
    output["momentum_12_1_rel"] = current_ranked["momentum_12_1_rel"].apply(pct_fmt)
    output["overheat_penalty"] = current_ranked["overheat_penalty"].apply(pct_fmt)

    output["volatility_1y"] = current_ranked["volatility_1y"].apply(pct_fmt)
    output["aum"] = current_ranked["aum"].round(2)
    output["data_weeks"] = current_ranked["data_weeks"]
    output["history_years"] = current_ranked["history_years"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )
    output["factor_coverage"] = current_ranked["factor_coverage"].apply(ratio_fmt)
    output["history_reliability"] = current_ranked["history_reliability"].apply(ratio_fmt)
    output["market_regime"] = regime_info["regime"]

    output.to_csv(OUTPUT_FILE, index=False)

    weights_df = pd.DataFrame(
        {
            "factor": [name for name, _ in FACTOR_SPECS],
            "group": [factor_group_name(name) for name, _ in FACTOR_SPECS],
            "higher_is_better": [higher for _, higher in FACTOR_SPECS],
            "required_weeks": [FACTOR_REQUIRED_WEEKS[name] for name, _ in FACTOR_SPECS],
            "base_weight": base_weights,
            "base_weight_pct": base_weights * 100.0,
            "live_weight": live_weights,
            "live_weight_pct": live_weights * 100.0,
        }
    ).sort_values("live_weight", ascending=False)
    weights_df.to_csv(WEIGHTS_FILE, index=False)

    regime_df = pd.DataFrame([regime_info])
    regime_df.to_csv(TMP_OUTPUT_DIR / f"{SECTOR}_Codex_regime.csv", index=False)

    backtest_df = pd.concat(
        [train_records, val_records, test_records, full_records],
        ignore_index=True,
    )
    if not backtest_df.empty:
        backtest_df = backtest_df.sort_values(["segment", "date"]).reset_index(drop=True)
    backtest_df.to_csv(BACKTEST_FILE, index=False)

    trials_df.head(300).to_csv(TUNING_TRIALS_FILE, index=False)

    # --- Console summary ---
    def metric_line(label: str, m: Dict[str, float]) -> str:
        return (
            f"{label:10s} | periods={int(m['n_periods']):2d} | "
            f"obj={m['objective']:6.2f} | "
            f"top_excess={m['mean_top_excess']*100:6.2f}% | "
            f"hit={m['hit_rate']*100:5.1f}% | "
            f"IC={m['mean_ic']:.3f} | "
            f"turn={m['avg_turnover']*100:4.1f}% | "
            f"subConc={m['mean_top_subsector_concentration']*100:4.1f}%"
        )

    print("\n" + "-" * 92)
    print("  BACKTEST SUMMARY (Top-7 portfolio each snapshot, static tuned weights)")
    print("-" * 92)
    print(metric_line("Train", train_metrics))
    print(metric_line("Validation", val_metrics))
    print(metric_line("Test", test_metrics))
    print(metric_line("Full", full_metrics))

    print("\n  Current market regime:")
    print(f"    - regime: {regime_info['regime']}")
    if pd.notna(regime_info["ret_3m"]):
        print(f"    - benchmark 3M return: {regime_info['ret_3m']*100:.2f}%")
    if pd.notna(regime_info["ret_6m"]):
        print(f"    - benchmark 6M return: {regime_info['ret_6m']*100:.2f}%")
    if pd.notna(regime_info["vol_ratio"]):
        print(f"    - recent/trailing vol ratio: {regime_info['vol_ratio']:.3f}")
    if pd.notna(regime_info["dd_6m"]):
        print(f"    - max drawdown last 6M: {regime_info['dd_6m']*100:.2f}%")

    print("\n  Top live-weight factors:")
    for _, r in weights_df.head(8).iterrows():
        direction = "higher" if bool(r["higher_is_better"]) else "lower"
        print(
            f"    - {r['factor']}: {r['live_weight_pct']:.2f}% "
            f"(group={r['group']}, {direction} is better)"
        )

    print("\n" + "-" * 92)
    print("  TOP 20 TOTAL MARKET FUNDS (CODEX)")
    print("-" * 92)
    print(
        output.head(20)[
            [
                "rank",
                "name",
                "subsector",
                "score",
                "cagr_5y",
                "alpha_1y",
                "info_1y",
                "momentum_6m_rel",
                "max_drawdown_2y",
                "rolling_beat_rate_2y",
            ]
        ].to_string(index=False)
    )

    print("\n  Files generated:")
    print(f"    - {OUTPUT_FILE}")
    print(f"    - {WEIGHTS_FILE}")
    print(f"    - {BACKTEST_FILE}")
    print(f"    - {TUNING_TRIALS_FILE}")
    print(f"    - {TMP_OUTPUT_DIR / f'{SECTOR}_Codex_regime.csv'}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
