#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm - Codex (Validation-Driven)

This model is explicitly optimized for forward 1-year outcomes, not just
current rankings. It works in four stages:

1) Build a historical panel of cross-sectional fund features at many past dates.
2) Backtest each date against the next 1-year forward return.
3) Tune factor weights using train/validation splits (time-ordered).
4) Use the tuned weights to rank funds on the latest date.

Outputs:
- results/Small Cap_Codex.tsv
- data/tmp/Small Cap_Codex_weights.tsv
- data/tmp/Small Cap_Codex_backtest.tsv
- data/tmp/Small Cap_Codex_tuning_trials.tsv
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
SECTOR = "Small Cap"
SUBSECTOR = "Small Cap Fund"
BENCHMARK_INDEX = "Small Cap"  # .NISM250

RISK_FREE_RATE = 0.065
WEEKS_PER_YEAR = 52

LOOKBACK_3M = 13
LOOKBACK_6M = 26
LOOKBACK_1Y = 52
LOOKBACK_2Y = 104
LOOKBACK_3Y = 156
LOOKBACK_5Y = 260

FORWARD_HORIZON_WEEKS = 52
EVAL_STEP_WEEKS = 5  # slightly coarser snapshots for faster tuning

MIN_WEEKS_TO_CONSIDER_FUND = 50
MIN_FUNDS_PER_DATE = 8
TOP_K = 5

# Tuning controls
SEARCH_TRIALS = 700
REFINE_MUTATIONS_PER_BASE = 4
TOP_TRIALS_FOR_ENSEMBLE = 10
RANDOM_SEED = 42
MAX_FACTOR_WEIGHT = 0.28
MAX_CV_FOLDS = 4

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Codex.tsv"

# Keep intermediate diagnostics out of results/.
TMP_OUTPUT_DIR = ROOT_DIR / "data" / "tmp"
WEIGHTS_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_weights.tsv"
BACKTEST_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_backtest.tsv"
TUNING_TRIALS_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_Codex_tuning_trials.tsv"


# (factor_name, higher_is_better)
FACTOR_SPECS: List[Tuple[str, bool]] = [
    ("alpha_1y", True),
    ("info_1y", True),
    ("sortino_1y", True),
    ("omega_1y", True),
    ("momentum_3m_rel", True),
    ("momentum_6m_rel", True),
    ("momentum_12m_rel", True),
    ("overheat_gap", False),          # avoid chasing short-term spikes
    ("rolling_active_2y", True),
    ("down_capture_1y", False),       # lower down-capture is better
    ("swing_down_rel_1y", True),
    ("max_drawdown_2y", False),
    ("ulcer_2y", False),
    ("vol_regime_2y", False),
    ("cagr_3y", True),
    ("cagr_5y", True),
]

FACTOR_NAMES = [name for name, _ in FACTOR_SPECS]
RANK_COLS = [f"rank_{name}" for name in FACTOR_NAMES]
FACTOR_INDEX = {name: i for i, name in enumerate(FACTOR_NAMES)}

MOMENTUM_FACTORS = {"momentum_3m_rel", "momentum_6m_rel", "momentum_12m_rel"}
DEFENSIVE_FACTORS = {
    "down_capture_1y",
    "swing_down_rel_1y",
    "max_drawdown_2y",
    "ulcer_2y",
    "vol_regime_2y",
    "overheat_gap",
}
QUALITY_FACTORS = {
    "alpha_1y",
    "info_1y",
    "sortino_1y",
    "omega_1y",
    "rolling_active_2y",
    "cagr_3y",
    "cagr_5y",
}
ANCHOR_FACTORS = {"cagr_3y", "cagr_5y"}

FACTOR_HORIZON: Dict[str, str] = {
    "alpha_1y": "short",
    "info_1y": "short",
    "sortino_1y": "short",
    "omega_1y": "short",
    "momentum_3m_rel": "short",
    "momentum_6m_rel": "short",
    "momentum_12m_rel": "medium",
    "overheat_gap": "short",
    "rolling_active_2y": "medium",
    "down_capture_1y": "short",
    "swing_down_rel_1y": "short",
    "max_drawdown_2y": "medium",
    "ulcer_2y": "medium",
    "vol_regime_2y": "medium",
    "cagr_3y": "medium",
    "cagr_5y": "long",
}

FACTOR_REQUIRED_WEEKS: Dict[str, int] = {
    "alpha_1y": LOOKBACK_1Y,
    "info_1y": LOOKBACK_1Y,
    "sortino_1y": LOOKBACK_1Y,
    "omega_1y": LOOKBACK_1Y,
    "momentum_3m_rel": LOOKBACK_1Y,
    "momentum_6m_rel": LOOKBACK_1Y,
    "momentum_12m_rel": LOOKBACK_1Y,
    "overheat_gap": LOOKBACK_1Y,
    "rolling_active_2y": LOOKBACK_2Y,
    "down_capture_1y": LOOKBACK_1Y,
    "swing_down_rel_1y": LOOKBACK_1Y,
    "max_drawdown_2y": LOOKBACK_2Y,
    "ulcer_2y": LOOKBACK_2Y,
    "vol_regime_2y": LOOKBACK_2Y,
    "cagr_3y": LOOKBACK_3Y,
    "cagr_5y": LOOKBACK_5Y,
}

DEFAULT_TIMEFRAME_IMPORTANCE: Dict[str, float] = {
    "short": 0.45,
    "medium": 0.40,
    "long": 0.15,
}

HISTORY_NEUTRAL_SCORE = 50.0
HISTORY_CONFIDENCE_CAP_WEEKS = LOOKBACK_5Y
MIN_HISTORY_RELIABILITY = 0.78
MAX_HISTORY_RELIABILITY = 1.00


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


def annualised_volatility(returns: pd.Series) -> float:
    if len(returns) < 8:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def downside_deviation(returns: pd.Series, mar: float = RISK_FREE_RATE) -> float:
    if len(returns) < 8:
        return np.nan
    mar_weekly = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    downside = (returns - mar_weekly)
    downside = downside[downside < 0]
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR))


def sortino_ratio(cagr_value: float, down_dev: float) -> float:
    if pd.isna(cagr_value) or pd.isna(down_dev) or down_dev <= 0:
        return np.nan
    return float((cagr_value - RISK_FREE_RATE) / down_dev)


def omega_ratio(returns: pd.Series, mar: float = RISK_FREE_RATE) -> float:
    if len(returns) < 12:
        return np.nan
    mar_weekly = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    excess = returns - mar_weekly
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    if losses <= 1e-12:
        return 10.0
    return float(gains / losses)


def alpha_beta_info(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float, float]:
    """
    Jensen alpha (annualized), beta, and information ratio from weekly returns.
    """
    aligned = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1)
    aligned = aligned.dropna()
    if len(aligned) < 12:
        return np.nan, np.nan, np.nan

    f = aligned["fund"].to_numpy()
    b = aligned["bench"].to_numpy()
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

    excess = f - b
    tracking_error = np.std(excess, ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    if tracking_error <= 1e-12:
        info = np.nan
    else:
        info = float((np.mean(excess) * WEEKS_PER_YEAR) / tracking_error)

    return alpha, beta, info


def capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    aligned = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1)
    aligned = aligned.dropna()
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


def swing_down_relative(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    """
    Relative performance in benchmark down/up swing buckets.
    """
    aligned = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1)
    aligned = aligned.dropna()
    if len(aligned) < 16:
        return np.nan, np.nan

    low_q = aligned["bench"].quantile(0.25)
    high_q = aligned["bench"].quantile(0.75)

    down_mask = aligned["bench"] <= low_q
    up_mask = aligned["bench"] >= high_q

    down_rel = np.nan
    if down_mask.sum() >= 8:
        down_rel = float((aligned.loc[down_mask, "fund"] - aligned.loc[down_mask, "bench"]).mean())

    up_rel = np.nan
    if up_mask.sum() >= 8:
        up_rel = float((aligned.loc[up_mask, "fund"] - aligned.loc[up_mask, "bench"]).mean())

    return down_rel, up_rel


def max_drawdown(nav_window: pd.Series) -> float:
    if len(nav_window) < 10:
        return np.nan
    drawdown = nav_window / nav_window.cummax() - 1.0
    return float(drawdown.min())


def ulcer_index(nav_window: pd.Series) -> float:
    if len(nav_window) < 10:
        return np.nan
    drawdown_pct = (nav_window / nav_window.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(drawdown_pct ** 2)))


def rolling_active_median(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    end_idx: int,
    roll_weeks: int = LOOKBACK_6M,
) -> float:
    """
    Median rolling active return over trailing 2Y horizon.
    """
    if end_idx < roll_weeks + 20:
        return np.nan

    f_roll = fund_nav.iloc[: end_idx + 1].pct_change(roll_weeks)
    b_roll = bench_nav.iloc[: end_idx + 1].pct_change(roll_weeks)
    active = (f_roll - b_roll).dropna().tail(LOOKBACK_2Y)
    if len(active) < 12:
        return np.nan
    return float(active.median())


def vol_regime_ratio(nav_window: pd.Series) -> float:
    """
    Recent 13W annualized vol / trailing annualized vol.
    Lower indicates cooling volatility.
    """
    if len(nav_window) < LOOKBACK_1Y:
        return np.nan
    ret = nav_window.pct_change().dropna()
    if len(ret) < 26:
        return np.nan
    recent = ret.tail(13).std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    trailing = ret.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    if trailing <= 1e-12:
        return np.nan
    return float(recent / trailing)


def build_feature_row(fund_nav: pd.Series, bench_nav: pd.Series, end_idx: int) -> Dict[str, float]:
    """
    Build features using only data available up to `end_idx`.
    Returns empty dict if base requirements are not met.
    """
    fund_1y = get_nav_window(fund_nav, end_idx, LOOKBACK_1Y)
    bench_1y = get_nav_window(bench_nav, end_idx, LOOKBACK_1Y)
    if fund_1y.empty or bench_1y.empty:
        return {}

    feat: Dict[str, float] = {name: np.nan for name in FACTOR_NAMES}

    cagr_1y = cagr_from_window(fund_1y, LOOKBACK_1Y)
    cagr_3y = cagr_at(fund_nav, end_idx, LOOKBACK_3Y)
    cagr_5y = cagr_at(fund_nav, end_idx, LOOKBACK_5Y)

    bench_3m = cagr_at(bench_nav, end_idx, LOOKBACK_3M)
    bench_6m = cagr_at(bench_nav, end_idx, LOOKBACK_6M)
    bench_12m = cagr_at(bench_nav, end_idx, LOOKBACK_1Y)
    fund_3m = cagr_at(fund_nav, end_idx, LOOKBACK_3M)
    fund_6m = cagr_at(fund_nav, end_idx, LOOKBACK_6M)
    fund_12m = cagr_at(fund_nav, end_idx, LOOKBACK_1Y)

    mom_3m = fund_3m - bench_3m if not pd.isna(fund_3m) and not pd.isna(bench_3m) else np.nan
    mom_6m = fund_6m - bench_6m if not pd.isna(fund_6m) and not pd.isna(bench_6m) else np.nan
    mom_12m = fund_12m - bench_12m if not pd.isna(fund_12m) and not pd.isna(bench_12m) else np.nan

    fund_ret_1y = fund_1y.pct_change().dropna()
    bench_ret_1y = bench_1y.pct_change().dropna()

    alpha, beta, info = alpha_beta_info(fund_ret_1y, bench_ret_1y)
    down_dev = downside_deviation(fund_ret_1y)
    sortino = sortino_ratio(cagr_1y, down_dev)
    omega = omega_ratio(fund_ret_1y)
    up_capture, down_capture = capture_ratios(fund_ret_1y, bench_ret_1y)
    swing_down_rel, swing_up_rel = swing_down_relative(fund_ret_1y, bench_ret_1y)

    fund_2y = get_nav_window(fund_nav, end_idx, LOOKBACK_2Y)
    if fund_2y.empty:
        max_dd_2y = np.nan
        ulcer_2y = np.nan
        vol_reg_2y = np.nan
    else:
        max_dd_2y = max_drawdown(fund_2y)
        ulcer_2y = ulcer_index(fund_2y)
        vol_reg_2y = vol_regime_ratio(fund_2y)

    roll_active = rolling_active_median(fund_nav, bench_nav, end_idx)
    overheat_gap = mom_3m - mom_12m if not pd.isna(mom_3m) and not pd.isna(mom_12m) else np.nan

    feat["alpha_1y"] = alpha
    feat["info_1y"] = info
    feat["sortino_1y"] = sortino
    feat["omega_1y"] = omega
    feat["momentum_3m_rel"] = mom_3m
    feat["momentum_6m_rel"] = mom_6m
    feat["momentum_12m_rel"] = mom_12m
    feat["overheat_gap"] = overheat_gap
    feat["rolling_active_2y"] = roll_active
    feat["down_capture_1y"] = down_capture
    feat["swing_down_rel_1y"] = swing_down_rel
    feat["max_drawdown_2y"] = max_dd_2y
    feat["ulcer_2y"] = ulcer_2y
    feat["vol_regime_2y"] = vol_reg_2y
    feat["cagr_3y"] = cagr_3y
    feat["cagr_5y"] = cagr_5y

    # Additional diagnostics used in output
    feat["cagr_1y"] = cagr_1y
    feat["beta_1y"] = beta
    feat["up_capture_1y"] = up_capture
    feat["swing_up_rel_1y"] = swing_up_rel
    feat["volatility_1y"] = annualised_volatility(fund_ret_1y)

    return feat


# ===================================================================
# Panel construction and scoring
# ===================================================================
def add_cross_sectional_ranks(df: pd.DataFrame, group_col: str = "date") -> pd.DataFrame:
    """Add percentile-rank columns for each factor."""
    out = df.copy()
    for factor_name, higher_is_better in FACTOR_SPECS:
        rank_col = f"rank_{factor_name}"
        if group_col:
            ranks = out.groupby(group_col)[factor_name].rank(pct=True, na_option="keep")
        else:
            ranks = out[factor_name].rank(pct=True, na_option="keep")
        if not higher_is_better:
            ranks = 1.0 - ranks
        out[rank_col] = ranks * 100.0
    return out


def score_from_rank_matrix(rank_matrix: np.ndarray, valid_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted score with missing-data-aware normalization.
    rank_matrix: N x F matrix with NaN values.
    valid_matrix: N x F matrix with 1.0 where value exists else 0.0.
    """
    matrix_filled = np.nan_to_num(rank_matrix, nan=0.0)
    weighted_sum = matrix_filled @ weights
    applied_weight = valid_matrix @ weights
    score = np.where(applied_weight > 0, weighted_sum / applied_weight, np.nan)
    return score


def weight_entropy(weights: np.ndarray) -> float:
    """Normalized entropy in [0,1], higher means less concentrated weights."""
    eps = 1e-12
    entropy = -np.sum(weights * np.log(weights + eps))
    return float(entropy / np.log(len(weights)))


def group_weight_sum(weights: np.ndarray, factors: set) -> float:
    """Sum model weight over a named factor group."""
    return float(sum(weights[FACTOR_INDEX[name]] for name in factors))


def is_weight_vector_valid(weights: np.ndarray) -> bool:
    """
    Constrain search to robust, diversified allocations.
    These convex constraints intentionally avoid over-concentrated solutions.
    """
    if len(weights) != len(FACTOR_NAMES):
        return False
    if np.any(weights < 0):
        return False
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        return False
    if float(weights.max()) > MAX_FACTOR_WEIGHT:
        return False

    momentum = group_weight_sum(weights, MOMENTUM_FACTORS)
    defensive = group_weight_sum(weights, DEFENSIVE_FACTORS)
    quality = group_weight_sum(weights, QUALITY_FACTORS)
    anchors = group_weight_sum(weights, ANCHOR_FACTORS)

    if not (0.15 <= momentum <= 0.45):
        return False
    if not (0.20 <= defensive <= 0.55):
        return False
    if not (0.20 <= quality <= 0.60):
        return False
    if not (0.04 <= anchors <= 0.25):
        return False

    # Ensure minimum diversification.
    if int((weights >= 0.03).sum()) < 5:
        return False
    return True


def build_walk_forward_folds(
    dev_dates: List[pd.Timestamp],
    min_train_periods: int = 16,
    val_periods: int = 8,
    step: int = 6,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """
    Build expanding-window walk-forward folds on development dates.
    """
    folds: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []
    start = min_train_periods
    n = len(dev_dates)
    while start + val_periods <= n:
        tr = dev_dates[:start]
        va = dev_dates[start: start + val_periods]
        if len(tr) >= 8 and len(va) >= 3:
            folds.append((tr, va))
        start += step

    # Fallback for short histories.
    if len(folds) < 3 and n >= 14:
        split1 = max(8, int(n * 0.55))
        split2 = max(split1 + 3, int(n * 0.75))
        split2 = min(split2, n - 2)
        folds = [
            (dev_dates[:split1], dev_dates[split1:split2]),
            (dev_dates[:split2], dev_dates[split2:]),
        ]
    return folds


def project_three_bucket_importance(
    raw: Dict[str, float],
    lower: Dict[str, float],
    upper: Dict[str, float],
) -> Dict[str, float]:
    """
    Clamp and normalize 3-bucket importance while respecting bounds.
    """
    out = {k: float(np.clip(raw[k], lower[k], upper[k])) for k in ["short", "medium", "long"]}
    diff = 1.0 - sum(out.values())
    for _ in range(30):
        if abs(diff) <= 1e-8:
            break
        if diff > 0:
            keys = [k for k in out if out[k] < upper[k] - 1e-12]
            if not keys:
                break
            slack = sum(upper[k] - out[k] for k in keys)
            for k in keys:
                out[k] += diff * ((upper[k] - out[k]) / slack)
        else:
            keys = [k for k in out if out[k] > lower[k] + 1e-12]
            if not keys:
                break
            room = sum(out[k] - lower[k] for k in keys)
            for k in keys:
                out[k] -= (-diff) * ((out[k] - lower[k]) / room)
        diff = 1.0 - sum(out.values())

    total = sum(out.values())
    if total > 0:
        out = {k: out[k] / total for k in out}
    return out


def rebalance_weights_with_caps(weights: np.ndarray, max_cap: float = MAX_FACTOR_WEIGHT) -> np.ndarray:
    """
    Re-normalize weights to sum 1 while enforcing per-factor max cap.
    """
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


def estimate_timeframe_importance(
    panel_ranked: pd.DataFrame,
    dev_dates: List[pd.Timestamp],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Estimate relative importance of short/medium/long history buckets from
    out-of-sample target correlations on development data.
    """
    default = DEFAULT_TIMEFRAME_IMPORTANCE.copy()
    subset = panel_ranked[panel_ranked["date"].isin(dev_dates)].copy()
    if subset.empty:
        return default, pd.DataFrame()

    rows: List[Dict[str, float]] = []
    for factor in FACTOR_NAMES:
        rank_col = f"rank_{factor}"
        ic_values: List[float] = []
        for _, g in subset.groupby("date", sort=True):
            x = g[rank_col]
            y = g["forward_1y_excess"]
            mask = x.notna() & y.notna()
            if mask.sum() < MIN_FUNDS_PER_DATE:
                continue
            xv = x[mask]
            yv = y[mask]
            if xv.nunique() < 3 or yv.nunique() < 3:
                continue
            ic = xv.corr(yv, method="spearman")
            if pd.notna(ic):
                ic_values.append(float(ic))

        ic_mean = float(np.mean(ic_values)) if ic_values else 0.0
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        stability = max(ic_mean - 0.35 * ic_std, 0.0)
        rows.append(
            {
                "factor": factor,
                "horizon": FACTOR_HORIZON[factor],
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "stability_score": stability,
            }
        )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return default, stats_df

    horizon_score = {
        hz: float(stats_df.loc[stats_df["horizon"] == hz, "stability_score"].sum())
        for hz in ["short", "medium", "long"]
    }
    blended = {
        hz: 0.58 * default[hz] + 0.42 * horizon_score.get(hz, 0.0)
        for hz in ["short", "medium", "long"]
    }

    total = sum(blended.values())
    if total <= 1e-12:
        return default, stats_df
    blended = {k: v / total for k, v in blended.items()}

    lower = {"short": 0.30, "medium": 0.25, "long": 0.10}
    upper = {"short": 0.60, "medium": 0.55, "long": 0.30}
    importance = project_three_bucket_importance(blended, lower, upper)
    return importance, stats_df


def apply_timeframe_importance_to_weights(
    base_weights: np.ndarray,
    timeframe_importance: Dict[str, float],
) -> np.ndarray:
    """
    Re-scale factor weights so bucket shares match learned timeframe importance.
    """
    adjusted = np.array(base_weights, dtype=float).copy()
    current_bucket = {
        hz: float(sum(adjusted[FACTOR_INDEX[f]] for f in FACTOR_NAMES if FACTOR_HORIZON[f] == hz))
        for hz in ["short", "medium", "long"]
    }

    for hz in ["short", "medium", "long"]:
        cur = current_bucket[hz]
        target = timeframe_importance[hz]
        if cur <= 1e-12:
            continue
        multiplier = target / cur
        for factor_name in FACTOR_NAMES:
            if FACTOR_HORIZON[factor_name] == hz:
                adjusted[FACTOR_INDEX[factor_name]] *= multiplier

    adjusted = rebalance_weights_with_caps(adjusted, max_cap=MAX_FACTOR_WEIGHT)
    adjusted = adjusted / adjusted.sum()
    return adjusted


def build_historical_panel(funds: List[Dict[str, Any]], bench_nav: pd.Series) -> pd.DataFrame:
    """
    Build time-series cross-sectional panel:
    each row = (date, fund) features + next-1Y realized return target.
    """
    bench_forward = bench_nav.shift(-FORWARD_HORIZON_WEEKS) / bench_nav - 1.0
    rows: List[Dict[str, Any]] = []

    eval_start_idx = LOOKBACK_1Y
    eval_end_idx = len(bench_nav) - FORWARD_HORIZON_WEEKS

    for idx in range(eval_start_idx, eval_end_idx, EVAL_STEP_WEEKS):
        asof_date = bench_nav.index[idx]
        bench_fwd = bench_forward.iat[idx]
        if pd.isna(bench_fwd):
            continue

        date_rows: List[Dict[str, Any]] = []
        for fund in funds:
            nav = fund["aligned_nav"]
            nav_now = nav.iat[idx]
            nav_fwd = nav.iat[idx + FORWARD_HORIZON_WEEKS]

            if pd.isna(nav_now) or pd.isna(nav_fwd) or nav_now <= 0:
                continue

            feature_row = build_feature_row(nav, bench_nav, idx)
            if not feature_row:
                continue

            forward_return = float(nav_fwd / nav_now - 1.0)
            row = {
                "date": asof_date,
                "mfId": fund["mfId"],
                "name": fund["name"],
                "aum": fund["aum"],
                "data_weeks": fund["data_weeks"],
                "forward_1y_return": forward_return,
                "bench_forward_1y_return": float(bench_fwd),
                "forward_1y_excess": float(forward_return - bench_fwd),
            }
            row.update(feature_row)
            date_rows.append(row)

        # Keep only dates with enough cross-sectional breadth.
        if len(date_rows) >= MIN_FUNDS_PER_DATE:
            rows.extend(date_rows)

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel

    panel = panel.sort_values(["date", "mfId"]).reset_index(drop=True)
    return panel


def split_dates(dates: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]:
    """Chronological train/validation/test split."""
    n = len(dates)
    if n < 15:
        raise RuntimeError(
            f"Need at least 15 evaluation dates for robust tuning, found {n}."
        )

    train_end = max(8, int(n * 0.60))
    val_end = max(train_end + 3, int(n * 0.82))
    val_end = min(val_end, n - 2)

    if val_end <= train_end:
        val_end = train_end + 1

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    if len(val_dates) < 2 or len(test_dates) < 2:
        raise RuntimeError(
            "Could not create non-trivial train/validation/test split from available dates."
        )

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
            "groups": [],
        }

    rank_matrix = seg_df[RANK_COLS].to_numpy(dtype=float)
    valid_matrix = (~np.isnan(rank_matrix)).astype(float)
    y_return = seg_df["forward_1y_return"].to_numpy(dtype=float)
    y_excess = seg_df["forward_1y_excess"].to_numpy(dtype=float)

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
        "groups": groups,
    }


def evaluate_segment(
    segment: Dict[str, Any],
    weights: np.ndarray,
    top_k: int = TOP_K,
    collect_records: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate one segment using fixed weights.
    Returns summary metrics and per-date backtest records (optional).
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
            "objective": -1e9,
        }
        return metrics, pd.DataFrame()

    scores = score_from_rank_matrix(segment["rank_matrix"], segment["valid_matrix"], weights)
    y_return = segment["y_return"]
    y_excess = segment["y_excess"]
    seg_df = segment["df"]

    ic_values: List[float] = []
    top_returns: List[float] = []
    top_excesses: List[float] = []
    turnover_values: List[float] = []
    records: List[Dict[str, Any]] = []
    prev_top_set = None

    for idx in segment["groups"]:
        s = scores[idx]
        r = y_return[idx]
        e = y_excess[idx]

        valid = (~np.isnan(s)) & (~np.isnan(r)) & (~np.isnan(e))
        if valid.sum() < MIN_FUNDS_PER_DATE:
            continue

        s_valid = s[valid]
        r_valid = r[valid]
        e_valid = e[valid]
        if len(s_valid) < 3:
            continue

        ic = np.nan
        if np.nanstd(s_valid) > 1e-12 and np.nanstd(e_valid) > 1e-12:
            ic = pd.Series(s_valid).corr(pd.Series(e_valid), method="spearman")
            if pd.notna(ic):
                ic_values.append(float(ic))

        k = min(top_k, len(s_valid))
        top_local_idx = np.argpartition(s_valid, -k)[-k:]
        top_return = float(np.nanmean(r_valid[top_local_idx]))
        top_excess = float(np.nanmean(e_valid[top_local_idx]))
        top_returns.append(top_return)
        top_excesses.append(top_excess)

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
            top_funds = ",".join(top_funds_list)
            records.append(
                {
                    "segment": segment["label"],
                    "date": seg_df.iloc[idx[0]]["date"],
                    "n_funds": int(valid.sum()),
                    "ic": ic if pd.notna(ic) else np.nan,
                    "top_k_return": top_return,
                    "top_k_excess": top_excess,
                    "top_funds": top_funds,
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

    # Objective tuned for forward 1Y alpha with risk control.
    objective = (
        0.50 * (mean_top_excess * 100.0)
        + 0.20 * (mean_ic * 100.0)
        + 0.15 * ((hit_rate - 0.5) * 100.0)
        + 0.10 * (mean_top_return * 100.0)
        - 0.10 * (downside_excess * 100.0)
        - 0.05 * (vol_excess * 100.0)
        - 0.07 * (avg_turnover * 100.0)
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
        "objective": float(objective),
    }
    return metrics, pd.DataFrame(records)


def sample_weight_vector(rng: np.random.Generator, n_factors: int) -> np.ndarray:
    """Sample constrained non-negative weights that sum to 1."""
    while True:
        w = rng.dirichlet(np.full(n_factors, 0.90))
        if is_weight_vector_valid(w):
            return w


def seed_candidates() -> List[np.ndarray]:
    """A few deterministic priors before random search."""
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

    candidates_raw = [
        np.ones(n, dtype=float) / n,
        vec(
            {
                "alpha_1y": 0.16,
                "info_1y": 0.14,
                "rolling_active_2y": 0.12,
                "sortino_1y": 0.10,
                "down_capture_1y": 0.09,
                "max_drawdown_2y": 0.09,
                "momentum_6m_rel": 0.08,
                "momentum_12m_rel": 0.08,
                "vol_regime_2y": 0.06,
                "cagr_3y": 0.05,
                "cagr_5y": 0.03,
            }
        ),
        vec(
            {
                "momentum_3m_rel": 0.14,
                "momentum_6m_rel": 0.16,
                "momentum_12m_rel": 0.14,
                "overheat_gap": 0.08,
                "alpha_1y": 0.10,
                "info_1y": 0.08,
                "sortino_1y": 0.07,
                "max_drawdown_2y": 0.08,
                "down_capture_1y": 0.07,
                "vol_regime_2y": 0.08,
            }
        ),
        vec(
            {
                "sortino_1y": 0.13,
                "omega_1y": 0.09,
                "down_capture_1y": 0.13,
                "swing_down_rel_1y": 0.09,
                "max_drawdown_2y": 0.12,
                "ulcer_2y": 0.11,
                "alpha_1y": 0.09,
                "info_1y": 0.08,
                "rolling_active_2y": 0.08,
                "vol_regime_2y": 0.08,
            }
        ),
    ]
    candidates = [w for w in candidates_raw if is_weight_vector_valid(w)]
    return candidates


def tune_weights(
    cv_folds: List[Dict[str, Any]],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Tune weights with walk-forward cross-validation and ensemble averaging.
    """
    if not cv_folds:
        raise RuntimeError("No CV folds supplied for tuning.")

    rng = np.random.default_rng(RANDOM_SEED)
    n_factors = len(FACTOR_NAMES)

    # Stage 1: broad random search.
    candidates = seed_candidates()
    while len(candidates) < SEARCH_TRIALS:
        candidates.append(sample_weight_vector(rng, n_factors))

    # Lightweight pre-screen on the most recent fold to focus compute.
    screening_fold = cv_folds[-1]
    screening_scores: List[float] = []
    for w in candidates:
        m_tr, _ = evaluate_segment(
            screening_fold["train"], w, top_k=TOP_K, collect_records=False
        )
        m_va, _ = evaluate_segment(
            screening_fold["val"], w, top_k=TOP_K, collect_records=False
        )
        score = (
            m_va["objective"]
            + 0.15 * m_tr["objective"]
            + 0.20 * weight_entropy(w)
        )
        screening_scores.append(score)

    pre_top = max(90, SEARCH_TRIALS // 4)
    pre_idx = np.argsort(screening_scores)[-pre_top:]

    # Stage 2: local refinement around promising candidates.
    refined: List[np.ndarray] = []
    for i in pre_idx:
        base = candidates[i]
        refined.append(base)
        for _ in range(REFINE_MUTATIONS_PER_BASE):
            noise = sample_weight_vector(rng, n_factors)
            mixed = 0.82 * base + 0.18 * noise
            mixed = mixed / mixed.sum()
            if is_weight_vector_valid(mixed):
                refined.append(mixed)

    # De-duplicate by rounded key.
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

        for fold in cv_folds:
            m_tr, _ = evaluate_segment(fold["train"], w, top_k=TOP_K, collect_records=False)
            m_va, _ = evaluate_segment(fold["val"], w, top_k=TOP_K, collect_records=False)
            train_objs.append(m_tr["objective"])
            val_objs.append(m_va["objective"])
            val_excesses.append(m_va["mean_top_excess"])
            val_hits.append(m_va["hit_rate"])
            val_ics.append(m_va["mean_ic"])
            val_turnovers.append(m_va["avg_turnover"])

        train_obj_mean = float(np.mean(train_objs))
        val_obj_mean = float(np.mean(val_objs))
        val_obj_std = float(np.std(val_objs, ddof=1)) if len(val_objs) > 1 else 0.0
        val_excess_mean = float(np.mean(val_excesses))
        val_excess_std = float(np.std(val_excesses, ddof=1)) if len(val_excesses) > 1 else 0.0
        val_excess_worst = float(np.min(val_excesses))
        val_hit_mean = float(np.mean(val_hits))
        val_ic_mean = float(np.mean(val_ics))
        val_turnover_mean = float(np.mean(val_turnovers))
        entropy = weight_entropy(w)

        selection_score = (
            val_obj_mean
            - 0.40 * val_obj_std
            + 0.18 * train_obj_mean
            + 0.20 * entropy
            + 0.10 * (val_hit_mean - 0.5) * 100.0
            + 0.08 * val_excess_worst * 100.0
            - 0.10 * val_excess_std * 100.0
            + 0.05 * val_ic_mean * 100.0
            - 0.10 * val_turnover_mean * 100.0
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
            "entropy": entropy,
        }
        for i, factor_name in enumerate(FACTOR_NAMES):
            row[f"w_{factor_name}"] = w[i]
        trial_rows.append(row)

    trials_df = pd.DataFrame(trial_rows).sort_values("selection_score", ascending=False)
    if trials_df.empty:
        raise RuntimeError("Weight tuning produced no trial rows.")

    # Ensemble top robust candidates instead of picking a single winner.
    eligible = trials_df[trials_df["cv_worst_val_top_excess_pct"] > -4.0]
    if eligible.empty:
        eligible = trials_df
    top_trials = eligible.head(TOP_TRIALS_FOR_ENSEMBLE).copy()

    if top_trials.empty:
        raise RuntimeError("No eligible top trials to build ensemble weights.")

    weight_matrix = top_trials[[f"w_{name}" for name in FACTOR_NAMES]].to_numpy(dtype=float)
    scores = top_trials["selection_score"].to_numpy(dtype=float)
    scaled = scores - scores.max()
    blend = np.exp(scaled / 2.5)
    blend = blend / blend.sum()
    ensemble_weights = np.average(weight_matrix, axis=0, weights=blend)
    ensemble_weights = ensemble_weights / ensemble_weights.sum()

    if not is_weight_vector_valid(ensemble_weights):
        # Fallback to top single candidate if convex blend violates constraints.
        best_single = top_trials.iloc[0][[f"w_{name}" for name in FACTOR_NAMES]].to_numpy(dtype=float)
        best_single = best_single / best_single.sum()
        ensemble_weights = best_single

    return ensemble_weights, trials_df


def score_current_snapshot(
    funds: List[Dict[str, Any]],
    bench_nav: pd.Series,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Compute latest-date ranking using tuned weights.
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
            "aum": fund["aum"],
            "data_weeks": fund["data_weeks"],
            "data_days": fund["data_days"],
            "history_years": fund["history_years"],
        }
        row.update(feat)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    current = pd.DataFrame(rows).copy()
    current_ranked = add_cross_sectional_ranks(current, group_col="")

    rank_matrix = current_ranked[RANK_COLS].to_numpy(dtype=float)
    data_weeks = current_ranked["data_weeks"].to_numpy(dtype=float)

    # History-aware normalization:
    # 1) Missing/insufficient horizon metrics revert toward neutral score (50),
    #    preventing short-history funds from getting an artificial advantage.
    # 2) Confidence saturates at 5Y, so 8-10Y history is not further rewarded.
    effective_rank = np.where(np.isnan(rank_matrix), HISTORY_NEUTRAL_SCORE, rank_matrix.copy())
    for j, factor_name in enumerate(FACTOR_NAMES):
        required = float(FACTOR_REQUIRED_WEEKS[factor_name])
        coverage = np.clip(data_weeks / required, 0.0, 1.0)
        effective_rank[:, j] = HISTORY_NEUTRAL_SCORE + (
            (effective_rank[:, j] - HISTORY_NEUTRAL_SCORE) * coverage
        )

    base_score = effective_rank @ weights
    history_ratio = np.clip(data_weeks / float(HISTORY_CONFIDENCE_CAP_WEEKS), 0.0, 1.0)
    reliability = MIN_HISTORY_RELIABILITY + (
        (MAX_HISTORY_RELIABILITY - MIN_HISTORY_RELIABILITY) * history_ratio
    )
    final_score = HISTORY_NEUTRAL_SCORE + (
        (base_score - HISTORY_NEUTRAL_SCORE) * reliability
    )

    current_ranked["history_reliability"] = reliability
    current_ranked["score"] = np.round(final_score, 2)
    current_ranked["rank"] = (
        current_ranked["score"].rank(ascending=False, method="min").astype(int)
    )
    current_ranked = current_ranked.sort_values(["rank", "name"]).reset_index(drop=True)
    return current_ranked


def pct_fmt(x: float) -> str:
    return f"{x * 100:.2f}" if pd.notna(x) else ""


def ratio_fmt(x: float) -> str:
    return f"{x:.3f}" if pd.notna(x) else ""


def num_fmt(x: float) -> str:
    return f"{x:.2f}" if pd.notna(x) else ""


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("\n" + "=" * 80)
    print("  SMALL CAP MUTUAL FUND SCORING - CODEX (TUNED + BACKTESTED)")
    print(f"  Benchmark : Nifty SmallCap 250 ({BENCHMARK_INDEX})")
    print("=" * 80)

    provider = MfDataProvider()

    # --- Benchmark ---
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_nav = clean_nav_chart(bench_df)
    if len(bench_nav) < LOOKBACK_1Y + FORWARD_HORIZON_WEEKS:
        raise RuntimeError("Benchmark data insufficient for forward 1Y backtests.")

    print(
        f"  Benchmark points: {len(bench_nav)}"
        f" ({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})"
    )

    # --- Funds ---
    df_all = provider.list_all_mf()
    df_small = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Funds in subsector: {len(df_small)}")

    funds: List[Dict[str, Any]] = []
    skipped = 0
    for _, row in df_small.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
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
            funds.append(
                {
                    "mfId": mf_id,
                    "name": name,
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
        raise RuntimeError("No eligible funds available for analysis.")

    print(f"  Eligible funds: {len(funds)} (skipped: {skipped})")

    # --- Build panel for tuning/backtesting ---
    panel = build_historical_panel(funds, bench_nav)
    if panel.empty:
        raise RuntimeError("Historical panel is empty; cannot tune weights.")

    unique_dates = sorted(panel["date"].unique().tolist())
    print(f"  Backtest snapshots: {len(unique_dates)}")
    print(f"  Panel rows: {len(panel)}")

    train_dates, val_dates, test_dates = split_dates(unique_dates)
    print(
        f"  Split dates -> train:{len(train_dates)}  val:{len(val_dates)}  test:{len(test_dates)}"
    )

    panel_ranked = add_cross_sectional_ranks(panel, group_col="date")

    # Development period uses train+validation for walk-forward CV tuning.
    dev_dates = train_dates + val_dates
    cv_pairs = build_walk_forward_folds(
        dev_dates,
        min_train_periods=max(14, len(dev_dates) // 3),
        val_periods=max(5, len(dev_dates) // 6),
        step=max(3, len(dev_dates) // 12),
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
                "train": make_segment(
                    panel_ranked, cv_train_dates, label=f"cv_{fold_idx}_train"
                ),
                "val": make_segment(
                    panel_ranked, cv_val_dates, label=f"cv_{fold_idx}_val"
                ),
            }
        )

    # --- Tune weights ---
    print("\n  Tuning factor weights with walk-forward cross-validation...")
    base_weights, trials_df = tune_weights(cv_folds)

    # Learn which history timeframes are actually predictive on development data
    # and align final weights to that regime importance.
    timeframe_importance, timeframe_stats = estimate_timeframe_importance(
        panel_ranked=panel_ranked,
        dev_dates=dev_dates,
    )
    final_weights = apply_timeframe_importance_to_weights(
        base_weights=base_weights,
        timeframe_importance=timeframe_importance,
    )

    # --- Final backtest diagnostics ---
    train_metrics, train_records = evaluate_segment(
        train_segment, final_weights, top_k=TOP_K, collect_records=True
    )
    val_metrics, val_records = evaluate_segment(
        val_segment, final_weights, top_k=TOP_K, collect_records=True
    )
    test_metrics, test_records = evaluate_segment(
        test_segment, final_weights, top_k=TOP_K, collect_records=True
    )
    full_metrics, full_records = evaluate_segment(
        full_segment, final_weights, top_k=TOP_K, collect_records=True
    )

    # --- Rank current funds using tuned weights ---
    current_ranked = score_current_snapshot(funds, bench_nav, final_weights)
    if current_ranked.empty:
        raise RuntimeError("Failed to build current ranking from tuned model.")

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame()
    output["mfId"] = current_ranked["mfId"]
    output["name"] = current_ranked["name"]
    output["rank"] = current_ranked["rank"]
    output["score"] = current_ranked["score"]
    output["data_days"] = current_ranked["data_days"]

    output["cagr_1y"] = current_ranked["cagr_1y"].apply(pct_fmt)
    output["cagr_3y"] = current_ranked["cagr_3y"].apply(pct_fmt)
    output["cagr_5y"] = current_ranked["cagr_5y"].apply(pct_fmt)
    output["alpha_1y"] = current_ranked["alpha_1y"].apply(pct_fmt)
    output["beta_1y"] = current_ranked["beta_1y"].apply(ratio_fmt)
    output["info_1y"] = current_ranked["info_1y"].apply(ratio_fmt)
    output["sortino_1y"] = current_ranked["sortino_1y"].apply(ratio_fmt)
    output["omega_1y"] = current_ranked["omega_1y"].apply(num_fmt)
    output["max_drawdown_2y"] = current_ranked["max_drawdown_2y"].apply(pct_fmt)
    output["ulcer_2y"] = current_ranked["ulcer_2y"].apply(num_fmt)
    output["up_capture_1y"] = current_ranked["up_capture_1y"].apply(ratio_fmt)
    output["down_capture_1y"] = current_ranked["down_capture_1y"].apply(ratio_fmt)
    output["rolling_active_2y"] = current_ranked["rolling_active_2y"].apply(pct_fmt)
    output["swing_down_rel_1y"] = current_ranked["swing_down_rel_1y"].apply(pct_fmt)
    output["swing_up_rel_1y"] = current_ranked["swing_up_rel_1y"].apply(pct_fmt)
    output["momentum_3m_rel"] = current_ranked["momentum_3m_rel"].apply(pct_fmt)
    output["momentum_6m_rel"] = current_ranked["momentum_6m_rel"].apply(pct_fmt)
    output["momentum_12m_rel"] = current_ranked["momentum_12m_rel"].apply(pct_fmt)
    output["overheat_gap"] = current_ranked["overheat_gap"].apply(pct_fmt)
    output["vol_regime_2y"] = current_ranked["vol_regime_2y"].apply(ratio_fmt)
    output["aum"] = current_ranked["aum"].round(2)
    output["data_weeks"] = current_ranked["data_weeks"]
    output["history_years"] = current_ranked["history_years"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    output["history_reliability"] = current_ranked["history_reliability"].apply(ratio_fmt)

    output.to_csv(OUTPUT_FILE, sep="\t", index=False)

    weights_df = pd.DataFrame(
        {
            "factor": [name for name, _ in FACTOR_SPECS],
            "higher_is_better": [higher for _, higher in FACTOR_SPECS],
            "horizon": [FACTOR_HORIZON[name] for name, _ in FACTOR_SPECS],
            "required_weeks": [FACTOR_REQUIRED_WEEKS[name] for name, _ in FACTOR_SPECS],
            "base_weight": base_weights,
            "base_weight_pct": base_weights * 100.0,
            "final_weight": final_weights,
            "final_weight_pct": final_weights * 100.0,
        }
    ).sort_values("final_weight", ascending=False)
    weights_df.to_csv(WEIGHTS_FILE, sep="\t", index=False)

    timeframe_df = pd.DataFrame(
        [
            {
                "horizon": hz,
                "importance": timeframe_importance[hz],
                "importance_pct": timeframe_importance[hz] * 100.0,
            }
            for hz in ["short", "medium", "long"]
        ]
    )
    if not timeframe_stats.empty:
        timeframe_stats = timeframe_stats.sort_values(
            ["horizon", "stability_score"], ascending=[True, False]
        ).reset_index(drop=True)
        timeframe_stats.to_csv(
            TMP_OUTPUT_DIR / f"{SECTOR}_Codex_timeframe_factor_stats.tsv",
            sep="\t",
            index=False,
        )
    timeframe_df.to_csv(
        TMP_OUTPUT_DIR / f"{SECTOR}_Codex_timeframe_importance.tsv",
        sep="\t",
        index=False,
    )

    backtest_df = pd.concat(
        [train_records, val_records, test_records, full_records],
        ignore_index=True,
    )
    if not backtest_df.empty:
        backtest_df = backtest_df.sort_values(["segment", "date"]).reset_index(drop=True)
    backtest_df.to_csv(BACKTEST_FILE, sep="\t", index=False)

    trials_df.head(250).to_csv(TUNING_TRIALS_FILE, sep="\t", index=False)

    # --- Console summary ---
    def metric_line(label: str, m: Dict[str, float]) -> str:
        return (
            f"{label:10s} | periods={int(m['n_periods']):2d} | "
            f"obj={m['objective']:6.2f} | "
            f"top_excess={m['mean_top_excess']*100:5.2f}% | "
            f"hit={m['hit_rate']*100:5.1f}% | "
            f"IC={m['mean_ic']:.3f} | "
            f"turn={m['avg_turnover']*100:4.1f}%"
        )

    print("\n" + "-" * 80)
    print("  BACKTEST SUMMARY (Top-5 portfolio each snapshot)")
    print("-" * 80)
    print(metric_line("Train", train_metrics))
    print(metric_line("Validation", val_metrics))
    print(metric_line("Test", test_metrics))
    print(metric_line("Full", full_metrics))

    print("\n  Learned timeframe importance:")
    print(
        "    - Short (3-12M/1Y): "
        f"{timeframe_importance['short']*100:.1f}%"
    )
    print(
        "    - Medium (2-3Y): "
        f"{timeframe_importance['medium']*100:.1f}%"
    )
    print(
        "    - Long (5Y): "
        f"{timeframe_importance['long']*100:.1f}%"
    )

    print("\n  Top tuned factors:")
    top_factors = weights_df.head(8)
    for _, r in top_factors.iterrows():
        direction = "higher" if bool(r["higher_is_better"]) else "lower"
        print(f"    - {r['factor']}: {r['final_weight_pct']:.2f}% ({direction} is better)")

    print("\n" + "-" * 80)
    print("  TOP 15 FUNDS (TUNED CODEX MODEL)")
    print("-" * 80)
    print(
        output.head(15)[
            [
                "rank",
                "name",
                "score",
                "cagr_5y",
                "alpha_1y",
                "info_1y",
                "momentum_6m_rel",
                "max_drawdown_2y",
            ]
        ].to_string(index=False)
    )

    print("\n  Files generated:")
    print(f"    - {OUTPUT_FILE}")
    print(f"    - {WEIGHTS_FILE}")
    print(f"    - {BACKTEST_FILE}")
    print(f"    - {TUNING_TRIALS_FILE}")
    print(f"    - {TMP_OUTPUT_DIR / f'{SECTOR}_Codex_timeframe_importance.tsv'}")
    print(f"    - {TMP_OUTPUT_DIR / f'{SECTOR}_Codex_timeframe_factor_stats.tsv'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
