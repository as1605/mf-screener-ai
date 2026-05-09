#!/usr/bin/env python3
"""
Multi Asset Allocation Fund Scoring Algorithm - Claude
======================================================

Objective
---------
Predict the next 1-year monthly-SIP outcome for every Indian Multi Asset
Allocation fund and rank accordingly.

Why this algorithm is intrinsically different from the other Claude algos
------------------------------------------------------------------------
The Claude algos for Small Cap / Mid Cap / Total Market are built around a
single equity benchmark - the alpha they measure is stock-selection skill
within one asset class. For Multi Asset funds the alpha IS the asset mix:
each fund freely allocates across equity, debt, gold and silver, and we have
no holdings data, only weekly NAVs. So this model is built on a different
backbone:

1.  Returns-Based Style Analysis (Sharpe, 1988) - constrained quadratic
    programming to *infer* each fund's effective sleeve weights from NAVs
    alone. Pre-2022 windows automatically drop the silver column.

2.  Brinson-style strategic vs tactical decomposition - separates value-add
    from a smart persistent mix (strategic) from value-add from changing
    weights over time (tactical). Morningstar's evidence shows tactical
    timing rarely pays off, so strategic carries 70 % of this pillar.

3.  Downside-aware risk metrics (Sortino, CDaR-5 %, Calmar, Pain Index,
    current drawdown) - more meaningful than symmetric vol for funds whose
    pitch is downside protection.

4.  Forward-looking outlook fit - asymmetric squared distance from a 2026
    target mix that reflects: moderate equity returns ahead, RBI rate cuts
    helping debt, and mean-reversion risk in gold/silver after the 2025
    super-rally. Over-weighting metals is penalised harder than other misses.

5.  SIP XIRR sanity - rolling realized 1Y SIP XIRR median + consistency, kept
    at a small 10 % weight because past SIP returns are a noisy predictor.

Composite (0-100) is the weight-aggregated cross-sectional percentile of the
five pillars. Funds with < 2 yr of NAV get a 0.85 haircut (one Indian rate
cycle hasn't been observed). Funds with < 6 mo are emitted with score = 0.

Sector : Multi Asset Allocation Fund
Author : Claude
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Constants
# ===================================================================
SECTOR = "Multi Asset"
SUBSECTOR = "Multi Asset Allocation Fund"

# Asset-class proxies (per task spec)
EQUITY_PROXY = "_NIFTY500"     # via get_index_chart
GOLD_PROXY = "M_SBIGL"          # via get_mf_chart (SBI Gold Fund, ~13 yr history)
SILVER_PROXY = "M_ICPVF"        # via get_mf_chart (ICICI Pru Silver, since 2022)

RISK_FREE_RATE = 0.065           # ~6.5 % annualised (Indian T-bill proxy)
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 365.25
WEEKLY_RF = (1.0 + RISK_FREE_RATE) ** (1.0 / WEEKS_PER_YEAR) - 1.0

SIP_MONTHS = 12
SIP_AMOUNT = 1.0
SIP_HURDLE = 0.08                # 8 % SIP-XIRR threshold for "consistency" metric

# Track-record thresholds
MIN_WEEKS_FOR_RANK = 26          # < this  -> score 0, but row still emitted
MIN_WEEKS_FULL_TRACK = 104       # < this  -> 0.85 haircut on composite
TRACK_RECORD_HAIRCUT = 0.85

# RBSA windows
ROLLING_RBSA_WINDOW = 52         # 1-year window for time-varying weights
ROLLING_RBSA_STEP = 4            # advance by 4 weeks per snapshot
CURRENT_RBSA_WINDOW = 52         # 1-year window for the "today" mix
MIN_BRINSON_SNAPSHOTS = 6        # need 6+ rolling weight snapshots for skill pillar

CAGR_LB_3Y = 156                 # weeks
CAGR_LB_5Y = 260
DOWNSIDE_LB_3Y = 156

# Forward-looking 2026 target mix.  Tunable here.
# Rationale (synthesised from market-outlook research, May 2026):
#   - Equity ~60 %: moderate earnings-driven returns, large-cap preferred.
#   - Debt   ~15 %: RBI 50-75 bps cuts ahead = bond tailwind.
#   - Gold   ~15 %: keep diversifier but trim after +72 % in 2025.
#   - Silver ~10 %: structural deficit but profit-taking risk after +122 %.
TARGET_MIX_2026 = {
    "eq":     0.60,
    "debt":   0.15,
    "gold":   0.15,
    "silver": 0.10,
}
METAL_OVERWEIGHT_PENALTY = 1.5   # over-weighting gold/silver vs target hits 1.5x harder

# Active-sleeve threshold (used in diversification metrics)
ACTIVE_SLEEVE_THRESHOLD = 0.05

# Composite pillar weights (sum to 1.0)
PILLAR_WEIGHTS = {
    "p1_alloc_profile": 0.25,
    "p2_outlook_fit":   0.15,
    "p3_alloc_skill":   0.25,
    "p4_downside":      0.25,
    "p5_sip_history":   0.10,
}

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Claude.csv"


# ===================================================================
# Generic NAV utilities
# ===================================================================

def to_clean_weekly_nav(chart: pd.DataFrame) -> Optional[pd.Series]:
    """Convert a raw chart DataFrame to a clean, weekly-indexed NAV Series.

    Tickertape's series is roughly weekly already, but funds and indices use
    slightly different snapshot weekdays. Resampling to a canonical W-MON
    cadence makes inner-joins across funds/indices well-defined.
    """
    if chart is None or chart.empty:
        return None
    df = chart.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["timestamp", "nav"])
    df = df[df["nav"] > 0]
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if df.empty:
        return None
    nav = df.set_index("timestamp")["nav"].astype(float)
    weekly = nav.resample("W-MON").last().dropna()
    return weekly if len(weekly) > 0 else None


# ===================================================================
# Asset-class proxy loader
# ===================================================================

def load_asset_proxies(provider: MfDataProvider) -> Dict[str, pd.Series]:
    """Return weekly RETURN series for the four asset-class sleeves.

    Keys: 'eq', 'gold', 'silver', 'cash'.

    'cash' is a synthetic constant series at the weekly risk-free rate -
    it represents the "debt-like" residual sleeve in the RBSA basis. The
    standard RBSA convention is: anything in NAV variation not explained by
    the other proxies gets attributed to the cash bucket.
    """
    out: Dict[str, pd.Series] = {}

    eq_chart = provider.get_index_chart(EQUITY_PROXY)
    eq_nav = to_clean_weekly_nav(eq_chart)
    if eq_nav is None or len(eq_nav) < 30:
        raise RuntimeError(f"Insufficient {EQUITY_PROXY} index data")
    out["eq"] = eq_nav.pct_change().dropna()
    out["eq"].name = "eq"

    gold_chart = provider.get_mf_chart(GOLD_PROXY)
    gold_nav = to_clean_weekly_nav(gold_chart)
    if gold_nav is None or len(gold_nav) < 30:
        raise RuntimeError(f"Insufficient {GOLD_PROXY} fund data")
    out["gold"] = gold_nav.pct_change().dropna()
    out["gold"].name = "gold"

    silver_chart = provider.get_mf_chart(SILVER_PROXY)
    silver_nav = to_clean_weekly_nav(silver_chart)
    if silver_nav is None or len(silver_nav) < 30:
        logger.warning(f"Silver proxy {SILVER_PROXY} unavailable - silver sleeve disabled")
        out["silver"] = pd.Series(dtype=float, name="silver")
    else:
        out["silver"] = silver_nav.pct_change().dropna()
        out["silver"].name = "silver"

    union_idx = out["eq"].index
    if not out["gold"].empty:
        union_idx = union_idx.union(out["gold"].index)
    if not out["silver"].empty:
        union_idx = union_idx.union(out["silver"].index)
    out["cash"] = pd.Series(WEEKLY_RF, index=union_idx, name="cash")

    return out


# ===================================================================
# Returns-Based Style Analysis (Sharpe 1988)
# ===================================================================

def _rbsa_solve(fund_ret: pd.Series, asset_rets: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Constrained QP: min ||fund_ret - asset_rets @ w||^2  s.t. sum(w)=1, w_i>=0.

    Returns {asset_name: weight} or None if the solve is infeasible / no data.
    """
    aligned = pd.concat([fund_ret.rename("fund"), asset_rets], axis=1, join="inner").dropna()
    if len(aligned) < 12:
        return None
    y = aligned["fund"].values.astype(float)
    X = aligned.drop(columns=["fund"]).values.astype(float)
    cols = list(aligned.drop(columns=["fund"]).columns)
    n_assets = X.shape[1]
    if n_assets == 0:
        return None

    def loss(w):
        resid = y - X @ w
        return float(np.mean(resid ** 2))

    def loss_grad(w):
        resid = y - X @ w
        return -2.0 / len(y) * X.T @ resid

    w0 = np.full(n_assets, 1.0 / n_assets)
    cons = ({"type": "eq",
             "fun": lambda w: np.sum(w) - 1.0,
             "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * n_assets

    try:
        res = minimize(
            loss, w0, jac=loss_grad, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not res.success:
            res = minimize(
                loss, w0, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 500, "ftol": 1e-10},
            )
        if not res.success:
            return None
        w = np.clip(res.x, 0.0, 1.0)
        s = w.sum()
        if s <= 0:
            return None
        w = w / s
        return dict(zip(cols, w.tolist()))
    except Exception as e:
        logger.debug(f"RBSA solve failed: {e}")
        return None


def _columns_for_window(
    asset_rets_dict: Dict[str, pd.Series],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    min_silver_overlap: int = 13,
) -> List[str]:
    """Decide which asset columns to include in an RBSA window.

    Silver is included only if the window has at least `min_silver_overlap`
    weeks of silver returns - otherwise we fall back to a 3-asset solve.
    """
    cols = ["eq", "gold", "cash"]
    silver = asset_rets_dict.get("silver", pd.Series(dtype=float))
    if not silver.empty:
        in_window = silver[(silver.index >= window_start) & (silver.index <= window_end)]
        if len(in_window) >= min_silver_overlap:
            cols.append("silver")
    return cols


def rbsa_current_mix(
    fund_ret: pd.Series,
    asset_rets_dict: Dict[str, pd.Series],
    lookback_weeks: int = CURRENT_RBSA_WINDOW,
) -> Optional[Dict[str, float]]:
    """RBSA over the last `lookback_weeks` of aligned data. Returns 4-key dict."""
    if fund_ret.empty:
        return None
    end_date = fund_ret.index[-1]
    start_date = end_date - pd.Timedelta(weeks=lookback_weeks)
    fund_window = fund_ret[fund_ret.index >= start_date]
    if len(fund_window) < 12:
        return None

    cols = _columns_for_window(asset_rets_dict, start_date, end_date)
    asset_df = pd.concat(
        [asset_rets_dict[c].rename(c) for c in cols],
        axis=1, join="inner",
    ).dropna()
    asset_window = asset_df[(asset_df.index >= start_date) & (asset_df.index <= end_date)]

    weights = _rbsa_solve(fund_window, asset_window)
    if weights is None:
        return None
    full = {"eq": 0.0, "gold": 0.0, "silver": 0.0, "cash": 0.0}
    for k, v in weights.items():
        full[k] = v
    return full


def rbsa_rolling_weights(
    fund_ret: pd.Series,
    asset_rets_dict: Dict[str, pd.Series],
    window_weeks: int = ROLLING_RBSA_WINDOW,
    step_weeks: int = ROLLING_RBSA_STEP,
) -> Optional[pd.DataFrame]:
    """Rolling RBSA snapshots. Returns DataFrame [eq, gold, silver, cash] indexed
    by window-end-date, or None if there isn't enough history for one snapshot."""
    if len(fund_ret) < window_weeks:
        return None

    rows: List[Tuple[pd.Timestamp, Dict[str, float]]] = []
    indices = list(fund_ret.index)

    for end_idx in range(window_weeks - 1, len(indices), step_weeks):
        end_date = indices[end_idx]
        start_date = end_date - pd.Timedelta(weeks=window_weeks)
        fund_win = fund_ret[(fund_ret.index >= start_date) & (fund_ret.index <= end_date)]
        if len(fund_win) < 12:
            continue

        cols = _columns_for_window(asset_rets_dict, start_date, end_date)
        asset_win = pd.concat(
            [asset_rets_dict[c].rename(c) for c in cols],
            axis=1, join="inner",
        ).dropna()
        asset_win = asset_win[(asset_win.index >= start_date) & (asset_win.index <= end_date)]

        weights = _rbsa_solve(fund_win, asset_win)
        if weights is None:
            continue
        full = {"eq": 0.0, "gold": 0.0, "silver": 0.0, "cash": 0.0}
        for k, v in weights.items():
            full[k] = v
        rows.append((end_date, full))

    if not rows:
        return None
    idx = [r[0] for r in rows]
    data = [r[1] for r in rows]
    df = pd.DataFrame(data, index=idx)
    for c in ["eq", "gold", "silver", "cash"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[["eq", "gold", "silver", "cash"]]


# ===================================================================
# Brinson-style strategic vs tactical decomposition
# ===================================================================

def brinson_decomposition(
    fund_ret: pd.Series,
    asset_rets_dict: Dict[str, pd.Series],
    rolling_weights: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    """Decompose annualised fund return into:

        strategic_alpha = E[ sum_i w_avg_i * R_t,i ] - E[ sum_i eq_w_i * R_t,i ]
            i.e. value-add from the persistent average mix vs an equal-weight
            mix of the sleeves the fund actually used.

        tactical_alpha  = E[ sum_i (w_t,i - w_avg_i) * R_t,i ]
            i.e. value-add from time-varying weight changes.

        selection_alpha = E[ R_fund - sum_i w_t,i * R_t,i ]
            i.e. the residual not explained by the inferred mix (security
            selection within the sleeves, factor tilts, fees, tracking error).

    All annualised by * 52.
    """
    out = {"strategic_alpha": None, "tactical_alpha": None, "selection_alpha": None}
    if rolling_weights is None or rolling_weights.empty:
        return out

    full_weekly_idx = fund_ret.index
    weekly_w = rolling_weights.reindex(full_weekly_idx).ffill().dropna(how="all")
    if weekly_w.empty:
        return out
    common_idx = weekly_w.index.intersection(fund_ret.index)
    if len(common_idx) < 26:
        return out

    weekly_w = weekly_w.loc[common_idx]
    fund_w_ret = fund_ret.loc[common_idx]

    # Asset return matrix on the common index. Pre-2022 silver returns are
    # filled with 0 so that w_silver=0 yields a 0 contribution (consistent
    # with the pre-2022 3-asset RBSA solve).
    asset_cols = ["eq", "gold", "silver", "cash"]
    asset_mat = pd.DataFrame(index=common_idx, columns=asset_cols, dtype=float)
    for c in asset_cols:
        asset_mat[c] = asset_rets_dict[c].reindex(common_idx)
    asset_mat = asset_mat.fillna(0.0)

    w_avg = weekly_w.mean(axis=0)

    strat_ret = (asset_mat * w_avg).sum(axis=1)
    actual_alloc_ret = (asset_mat * weekly_w).sum(axis=1)
    tactical_ret = actual_alloc_ret - strat_ret
    selection_ret = fund_w_ret - actual_alloc_ret

    active_sleeves = [c for c in asset_cols if weekly_w[c].mean() > 0.01]
    if not active_sleeves:
        return out
    eq_w = pd.Series({c: 1.0 / len(active_sleeves) for c in active_sleeves})
    eq_w = eq_w.reindex(asset_cols).fillna(0.0)
    eq_bench_ret = (asset_mat * eq_w).sum(axis=1)

    out["strategic_alpha"] = float((strat_ret - eq_bench_ret).mean() * WEEKS_PER_YEAR)
    out["tactical_alpha"] = float(tactical_ret.mean() * WEEKS_PER_YEAR)
    out["selection_alpha"] = float(selection_ret.mean() * WEEKS_PER_YEAR)
    return out


# ===================================================================
# Downside-aware risk metrics
# ===================================================================

def annualised_cagr(nav: pd.Series, weeks: int) -> Optional[float]:
    """CAGR over the trailing `weeks` weeks. Requires the full window."""
    if len(nav) < weeks + 1:
        return None
    end = float(nav.iloc[-1])
    start = float(nav.iloc[-(weeks + 1)])
    if start <= 0:
        return None
    years = weeks / WEEKS_PER_YEAR
    return float((end / start) ** (1.0 / years) - 1.0)


def sortino_ratio(nav: pd.Series, lookback_weeks: int = DOWNSIDE_LB_3Y) -> Optional[float]:
    """Annualised Sortino over the trailing window vs the weekly risk-free rate."""
    if len(nav) < 30:
        return None
    window = nav.tail(min(len(nav), lookback_weeks + 1))
    if len(window) < 30:
        return None
    rets = window.pct_change().dropna()
    if len(rets) < 20:
        return None
    excess = rets - WEEKLY_RF
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    dd = float(np.sqrt((downside ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR))
    if dd <= 1e-12:
        return None
    return float((excess.mean() * WEEKS_PER_YEAR) / dd)


def cdar_5pct(nav: pd.Series) -> Optional[float]:
    """Conditional Drawdown-at-Risk at 5 %.

    Mean of the worst 5 % drawdown observations. Returned as a non-positive
    number (-0.20 means the worst 5 % of weeks averaged a 20 % drawdown).
    Higher (less negative) is better.
    """
    if len(nav) < 30:
        return None
    dd = (nav / nav.cummax() - 1.0).dropna()
    if dd.empty:
        return None
    threshold = float(np.quantile(dd.values, 0.05))
    tail = dd[dd <= threshold]
    if tail.empty:
        return None
    return float(tail.mean())


def calmar_ratio(nav: pd.Series, lookback_weeks: int = DOWNSIDE_LB_3Y) -> Optional[float]:
    """CAGR / |Max Drawdown| over the trailing window."""
    if len(nav) < 30:
        return None
    window = nav.tail(min(len(nav), lookback_weeks + 1))
    if len(window) < 30:
        return None
    cagr = annualised_cagr(window, weeks=min(lookback_weeks, len(window) - 1))
    if cagr is None:
        return None
    mdd = float((window / window.cummax() - 1.0).min())
    if abs(mdd) < 1e-6:
        return None
    return float(cagr / abs(mdd))


def pain_index(nav: pd.Series) -> Optional[float]:
    """Average drawdown depth over the entire history (non-positive)."""
    if len(nav) < 30:
        return None
    dd = (nav / nav.cummax() - 1.0)
    return float(dd.mean())


def current_drawdown(nav: pd.Series) -> Optional[float]:
    """Current NAV vs trailing all-time peak (non-positive)."""
    if len(nav) < 5:
        return None
    return float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1.0)


# ===================================================================
# SIP XIRR (NPV bisection)
# ===================================================================

def to_month_start_nav(nav: pd.Series) -> pd.Series:
    """Daily-ffill the weekly NAV, then take the first observation per month.

    Models a SIP investor who buys at the most-recent NAV available on the
    1st of each month.
    """
    if nav.empty:
        return nav
    daily = nav.resample("D").ffill()
    return daily.resample("MS").first().dropna()


def _xirr_bisect(
    cashflows: List[float],
    times_years: List[float],
    lo: float = -0.99, hi: float = 5.0,
    tol: float = 1e-6, max_iter: int = 100,
) -> Optional[float]:
    """Solve NPV(r) = 0 by bisection. Returns None if no sign change."""
    def npv(r: float) -> float:
        try:
            return float(sum(cf / ((1.0 + r) ** t) for cf, t in zip(cashflows, times_years)))
        except (OverflowError, ZeroDivisionError):
            return float("nan")

    f_lo, f_hi = npv(lo), npv(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if not np.isfinite(f_mid):
            return None
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def sip_xirr_at(monthly: pd.Series, start_idx: int, months: int = SIP_MONTHS) -> Optional[float]:
    """Annualised XIRR of a monthly SIP starting at monthly[start_idx]."""
    end_idx = start_idx + months
    if start_idx < 0 or end_idx >= len(monthly):
        return None
    invest_navs = monthly.iloc[start_idx:end_idx]
    redeem_nav = monthly.iloc[end_idx]
    if (invest_navs <= 0).any() or redeem_nav <= 0:
        return None
    invest_dates = monthly.index[start_idx:end_idx]
    redeem_date = monthly.index[end_idx]

    units = SIP_AMOUNT / invest_navs.values
    redeem_value = float(units.sum() * redeem_nav)
    cashflows = [-float(SIP_AMOUNT)] * months + [redeem_value]
    times = [(d - invest_dates[0]).days / DAYS_PER_YEAR for d in invest_dates]
    times.append((redeem_date - invest_dates[0]).days / DAYS_PER_YEAR)
    return _xirr_bisect(cashflows, times)


def rolling_sip_xirr_series(monthly: pd.Series, months: int = SIP_MONTHS) -> pd.Series:
    """Series of realised 1Y SIP XIRRs indexed by SIP-start month."""
    if len(monthly) < months + 1:
        return pd.Series(dtype=float)
    out = {}
    for i in range(len(monthly) - months):
        r = sip_xirr_at(monthly, i, months=months)
        if r is not None and np.isfinite(r):
            out[monthly.index[i]] = r
    return pd.Series(out, dtype=float).sort_index()


def sip_history_stats(sip_series: pd.Series) -> Dict[str, Optional[float]]:
    """Summary stats on the rolling SIP-XIRR series."""
    out = {"sip_1y_p50": None, "sip_1y_p20": None, "sip_consistency": None}
    if sip_series.empty:
        return out
    out["sip_1y_p50"] = float(sip_series.median())
    out["sip_1y_p20"] = float(sip_series.quantile(0.20))
    out["sip_consistency"] = float((sip_series > SIP_HURDLE).mean())
    return out


# ===================================================================
# Forward outlook fit and diversification metrics
# ===================================================================

def outlook_fit_distance(weights: Dict[str, float]) -> float:
    """Asymmetric squared distance from TARGET_MIX_2026.

    Returns a positive number; smaller = closer to target = better.
    The pillar uses sign=-1 so larger distance lowers the score.
    """
    dist = 0.0
    for asset, target in TARGET_MIX_2026.items():
        w = weights.get(asset, 0.0)
        diff = w - target
        if asset in ("gold", "silver") and diff > 0:
            dist += METAL_OVERWEIGHT_PENALTY * (diff ** 2)
        else:
            dist += diff ** 2
    return float(dist)


def diversification_index(weights: Dict[str, float]) -> float:
    """1 - Herfindahl. Range [0, 1 - 1/N]. Higher = more spread across sleeves."""
    w = np.array(list(weights.values()))
    return float(1.0 - (w ** 2).sum())


def n_active_sleeves(weights: Dict[str, float], threshold: float = ACTIVE_SLEEVE_THRESHOLD) -> int:
    return int(sum(1 for v in weights.values() if v >= threshold))


# ===================================================================
# Composite score
# ===================================================================

def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; preserves NaN."""
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd is None or sd == 0 or not np.isfinite(sd):
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return (s - mu) / sd


def _pillar_score(
    df: pd.DataFrame,
    sub_weights: Dict[str, float],
    sub_signs: Dict[str, int],
) -> pd.Series:
    """Weighted-z pillar score. Funds missing a sub-feature have its weight
    redistributed; funds missing all sub-features return NaN (handled later)."""
    contrib = pd.DataFrame(index=df.index, dtype=float)
    weight_panel = pd.DataFrame(index=df.index, dtype=float)
    for col, w in sub_weights.items():
        if col not in df.columns:
            continue
        z = _zscore(df[col].astype(float))
        sign = float(sub_signs.get(col, 1))
        contrib[col] = w * sign * z
        weight_panel[col] = np.where(z.notna(), w, np.nan)
    if contrib.empty:
        return pd.Series(np.nan, index=df.index)
    raw_sum = contrib.sum(axis=1, skipna=True)
    used_w = weight_panel.sum(axis=1, skipna=True)
    return pd.Series(np.where(used_w > 0, raw_sum / used_w, np.nan), index=df.index)


def compute_composite(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build the five pillar scores (0-100 percentile) and the composite."""
    out = features_df.copy()

    p1 = _pillar_score(
        out,
        sub_weights={"diversification": 0.6, "n_active_sleeves": 0.4},
        sub_signs={"diversification": +1, "n_active_sleeves": +1},
    )
    p2 = _pillar_score(
        out,
        sub_weights={"outlook_fit": 1.0},
        sub_signs={"outlook_fit": -1},  # smaller distance is better
    )
    p3 = _pillar_score(
        out,
        sub_weights={"strategic_alpha": 0.7, "tactical_alpha": 0.3},
        sub_signs={"strategic_alpha": +1, "tactical_alpha": +1},
    )
    p4 = _pillar_score(
        out,
        sub_weights={
            "sortino_3y": 0.30,
            "cdar_5pct":  0.25,
            "calmar_3y":  0.25,
            "pain_index": 0.10,
            "dd_now":     0.10,
        },
        sub_signs={
            "sortino_3y": +1,
            "cdar_5pct":  +1,  # less-negative is better
            "calmar_3y":  +1,
            "pain_index": +1,  # less-negative is better
            "dd_now":     +1,  # less-negative is better
        },
    )
    p5 = _pillar_score(
        out,
        sub_weights={"sip_1y_p50": 0.5, "sip_consistency": 0.3, "sip_1y_p20": 0.2},
        sub_signs={"sip_1y_p50": +1, "sip_consistency": +1, "sip_1y_p20": +1},
    )

    # Convert each pillar to 0-100 percentile rank.  Funds missing the entire
    # pillar (NaN) get the neutral median 50 - they aren't penalised for not
    # having the data, they're penalised via the track-record haircut.
    out["p1_alloc_profile"] = (p1.rank(pct=True, na_option="keep") * 100.0).fillna(50.0)
    out["p2_outlook_fit"]   = (p2.rank(pct=True, na_option="keep") * 100.0).fillna(50.0)
    out["p3_alloc_skill"]   = (p3.rank(pct=True, na_option="keep") * 100.0).fillna(50.0)
    out["p4_downside"]      = (p4.rank(pct=True, na_option="keep") * 100.0).fillna(50.0)
    out["p5_sip_history"]   = (p5.rank(pct=True, na_option="keep") * 100.0).fillna(50.0)

    out["composite_raw"] = (
        PILLAR_WEIGHTS["p1_alloc_profile"] * out["p1_alloc_profile"] +
        PILLAR_WEIGHTS["p2_outlook_fit"]   * out["p2_outlook_fit"] +
        PILLAR_WEIGHTS["p3_alloc_skill"]   * out["p3_alloc_skill"] +
        PILLAR_WEIGHTS["p4_downside"]      * out["p4_downside"] +
        PILLAR_WEIGHTS["p5_sip_history"]   * out["p5_sip_history"]
    )
    return out


# ===================================================================
# Per-fund feature build
# ===================================================================

def _empty_feature_row(mf_id: str, name: str, aum: float, data_days: int) -> Dict:
    """Row for funds with too little NAV history to compute any feature."""
    return {
        "mfId": mf_id, "name": name, "aum": aum, "data_days": data_days,
        "cagr_3y": None, "cagr_5y": None,
        "eq_weight": None, "debt_weight": None,
        "gold_weight": None, "silver_weight": None,
        "diversification": None, "n_active_sleeves": None,
        "outlook_fit": None,
        "strategic_alpha": None, "tactical_alpha": None, "selection_alpha": None,
        "sortino_3y": None, "cdar_5pct": None, "calmar_3y": None,
        "pain_index": None, "dd_now": None,
        "sip_1y_p50": None, "sip_1y_p20": None, "sip_consistency": None,
    }


def build_fund_features(
    mf_id: str,
    name: str,
    aum: float,
    nav: pd.Series,
    asset_rets_dict: Dict[str, pd.Series],
) -> Optional[Dict]:
    """Compute every per-fund feature. Returns a row dict or None on hard failure."""
    n_weeks = len(nav)
    data_days = int((nav.index.max() - nav.index.min()).days + 1) if n_weeks >= 2 else n_weeks

    if n_weeks < MIN_WEEKS_FOR_RANK:
        return _empty_feature_row(mf_id, name, aum, data_days)

    fund_ret = nav.pct_change().dropna()
    fund_ret.name = "fund"

    current_mix = rbsa_current_mix(fund_ret, asset_rets_dict, lookback_weeks=CURRENT_RBSA_WINDOW)
    if current_mix is None:
        # Fall back to a lifetime solve before giving up
        current_mix = rbsa_current_mix(fund_ret, asset_rets_dict, lookback_weeks=len(fund_ret))
    if current_mix is None:
        return None

    rolling_w = rbsa_rolling_weights(
        fund_ret, asset_rets_dict,
        window_weeks=ROLLING_RBSA_WINDOW, step_weeks=ROLLING_RBSA_STEP,
    )
    if rolling_w is not None and len(rolling_w) >= MIN_BRINSON_SNAPSHOTS:
        skill = brinson_decomposition(fund_ret, asset_rets_dict, rolling_w)
    else:
        skill = {"strategic_alpha": None, "tactical_alpha": None, "selection_alpha": None}

    monthly = to_month_start_nav(nav)
    sip_series = rolling_sip_xirr_series(monthly)
    sip_stats = sip_history_stats(sip_series)

    return {
        "mfId": mf_id, "name": name, "aum": aum, "data_days": data_days,
        "cagr_3y": annualised_cagr(nav, CAGR_LB_3Y),
        "cagr_5y": annualised_cagr(nav, CAGR_LB_5Y),
        "eq_weight": current_mix["eq"],
        "debt_weight": current_mix["cash"],
        "gold_weight": current_mix["gold"],
        "silver_weight": current_mix["silver"],
        "diversification": diversification_index(current_mix),
        "n_active_sleeves": n_active_sleeves(current_mix),
        "outlook_fit": outlook_fit_distance(current_mix),
        "strategic_alpha": skill["strategic_alpha"],
        "tactical_alpha": skill["tactical_alpha"],
        "selection_alpha": skill["selection_alpha"],
        "sortino_3y": sortino_ratio(nav),
        "cdar_5pct": cdar_5pct(nav),
        "calmar_3y": calmar_ratio(nav),
        "pain_index": pain_index(nav),
        "dd_now": current_drawdown(nav),
        "sip_1y_p50": sip_stats["sip_1y_p50"],
        "sip_1y_p20": sip_stats["sip_1y_p20"],
        "sip_consistency": sip_stats["sip_consistency"],
    }


# ===================================================================
# Main pipeline
# ===================================================================

def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 78)
    print("  MULTI ASSET ALLOCATION FUND SCORING - CLAUDE MODEL")
    print("  Objective: predict next-1Y monthly-SIP outcomes")
    print("  Method   : RBSA + Brinson + downside-aware risk + 2026 macro fit")
    print("=" * 78)

    provider = MfDataProvider(date=date)

    print("\n  Loading asset-class proxies...")
    asset_rets = load_asset_proxies(provider)
    eq_n = len(asset_rets["eq"])
    gold_n = len(asset_rets["gold"])
    silver_n = len(asset_rets["silver"])
    eq_min = asset_rets["eq"].index.min().date()
    gold_min = asset_rets["gold"].index.min().date()
    silver_min = asset_rets["silver"].index.min().date() if silver_n > 0 else "n/a"
    print(f"    Equity ({EQUITY_PROXY}):  {eq_n:>4} weekly returns from {eq_min}")
    print(f"    Gold   ({GOLD_PROXY}):   {gold_n:>4} weekly returns from {gold_min}")
    print(f"    Silver ({SILVER_PROXY}):   {silver_n:>4} weekly returns from {silver_min}")
    print(f"    Cash:    constant {WEEKLY_RF*100:.4f}%/week ({RISK_FREE_RATE*100:.1f}% p.a.)")

    df_all = provider.list_all_mf()
    funds_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"\n  {SUBSECTOR}s in universe: {len(funds_df)}")

    rows: List[Dict] = []
    skipped: List[Tuple[str, str, str]] = []

    for _, row in funds_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = float(row.get("aum", 0) or 0)
        try:
            chart = provider.get_mf_chart(mf_id)
            nav = to_clean_weekly_nav(chart)
            if nav is None or len(nav) < 2:
                skipped.append((mf_id, name, "no NAV data"))
                continue

            feat = build_fund_features(mf_id, name, aum, nav, asset_rets)
            if feat is None:
                skipped.append((mf_id, name, "RBSA failed"))
                continue
            rows.append(feat)
            if feat["eq_weight"] is None:
                skipped.append(
                    (mf_id, name, f"only {len(nav)} weeks (<{MIN_WEEKS_FOR_RANK})")
                )
        except Exception as e:
            logger.error(f"Failed {mf_id} ({name}): {e}", exc_info=False)
            skipped.append((mf_id, name, f"error: {e}"))

    if not rows:
        logger.error("No funds processed. Exiting.")
        sys.exit(1)

    full_count = sum(1 for r in rows if r["eq_weight"] is not None)
    print(f"  Funds processed: {len(rows)}  ({full_count} with full feature set)")
    if skipped:
        print(f"  Notes ({len(skipped)}):")
        for mf_id, name, why in skipped[:12]:
            print(f"    - {mf_id:<8}  {name[:45]:<45}  ({why})")
        if len(skipped) > 12:
            print(f"    ... and {len(skipped) - 12} more")

    df = pd.DataFrame(rows)
    df = compute_composite(df)

    # Track-record haircut: <2y -> 0.85
    haircut = pd.Series(1.0, index=df.index)
    haircut[df["data_days"] < (MIN_WEEKS_FULL_TRACK * 7)] = TRACK_RECORD_HAIRCUT
    df["score"] = (df["composite_raw"] * haircut).round(2)

    # Funds with < 26w of data: explicitly score 0 and zero-out pillars in CSV
    no_data_mask = df["data_days"] < (MIN_WEEKS_FOR_RANK * 7)
    df.loc[no_data_mask, "score"] = 0.0
    for col in ("p1_alloc_profile", "p2_outlook_fit", "p3_alloc_skill",
                "p4_downside", "p5_sip_history"):
        df.loc[no_data_mask, col] = 0.0

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)

    # ---- Print self-backtest diagnostic ----
    print("\n  Pillar weights (cross-sectional percentile, then weighted)")
    print("  " + "-" * 72)
    for k, v in PILLAR_WEIGHTS.items():
        print(f"  {k:<24} {v*100:>6.1f}%")
    print(f"  Track-record haircut for funds < {MIN_WEEKS_FULL_TRACK} weeks: x{TRACK_RECORD_HAIRCUT}")
    print(f"  Forward target mix (eq/debt/gold/silver): "
          f"{TARGET_MIX_2026['eq']:.2f} / {TARGET_MIX_2026['debt']:.2f} / "
          f"{TARGET_MIX_2026['gold']:.2f} / {TARGET_MIX_2026['silver']:.2f} "
          f"(metal over-weight penalty x{METAL_OVERWEIGHT_PENALTY})")

    # ---- Format output ----
    pct_fmt = lambda v: f"{v*100:.2f}" if pd.notna(v) else ""
    w_fmt = lambda v: f"{v:.3f}" if pd.notna(v) else ""
    r_fmt = lambda v: f"{v:.3f}" if pd.notna(v) else ""

    out = pd.DataFrame()
    out["mfId"] = df["mfId"]
    out["name"] = df["name"]
    out["rank"] = df["rank"]
    out["score"] = df["score"]
    out["data_days"] = df["data_days"].astype(int)
    out["cagr_3y"] = df["cagr_3y"].apply(pct_fmt)
    out["cagr_5y"] = df["cagr_5y"].apply(pct_fmt)
    out["eq_weight"] = df["eq_weight"].apply(w_fmt)
    out["debt_weight"] = df["debt_weight"].apply(w_fmt)
    out["gold_weight"] = df["gold_weight"].apply(w_fmt)
    out["silver_weight"] = df["silver_weight"].apply(w_fmt)
    out["diversification"] = df["diversification"].apply(r_fmt)
    out["n_active_sleeves"] = df["n_active_sleeves"].apply(
        lambda v: int(v) if pd.notna(v) else "")
    out["outlook_fit"] = df["outlook_fit"].apply(r_fmt)
    out["strategic_alpha"] = df["strategic_alpha"].apply(pct_fmt)
    out["tactical_alpha"] = df["tactical_alpha"].apply(pct_fmt)
    out["sortino_3y"] = df["sortino_3y"].apply(r_fmt)
    out["cdar_5pct"] = df["cdar_5pct"].apply(pct_fmt)
    out["calmar_3y"] = df["calmar_3y"].apply(r_fmt)
    out["pain_index"] = df["pain_index"].apply(pct_fmt)
    out["dd_now"] = df["dd_now"].apply(pct_fmt)
    out["sip_1y_p50"] = df["sip_1y_p50"].apply(pct_fmt)
    out["sip_consistency"] = df["sip_consistency"].apply(r_fmt)
    out["p1_alloc_profile"] = df["p1_alloc_profile"].round(2)
    out["p2_outlook_fit"] = df["p2_outlook_fit"].round(2)
    out["p3_alloc_skill"] = df["p3_alloc_skill"].round(2)
    out["p4_downside"] = df["p4_downside"].round(2)
    out["p5_sip_history"] = df["p5_sip_history"].round(2)
    out["aum"] = df["aum"].round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # ---- Top 15 console table ----
    print("\n" + "=" * 78)
    print("  TOP 15 MULTI ASSET FUNDS - CLAUDE MODEL")
    print("=" * 78)
    cols_to_show = ["rank", "name", "score",
                    "eq_weight", "debt_weight", "gold_weight", "silver_weight",
                    "sortino_3y", "cagr_3y"]
    pretty = out[cols_to_show].head(15).copy()
    pretty["name"] = pretty["name"].apply(lambda s: s[:36] + ".." if len(s) > 36 else s)
    print()
    print(pretty.to_string(index=False))
    print(f"\n  Full results ({len(out)} funds) -> {OUTPUT_FILE}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multi Asset Allocation MF screener (Claude)"
    )
    p.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
