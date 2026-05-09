#!/usr/bin/env python3
"""
Mid Cap Mutual Fund Scoring Algorithm - Claude (Regime-Conditional Predictive Score)
====================================================================================

A fresh take on Mid Cap fund scoring built around the premise that
*next-1Y SIP outcomes are driven by regime-conditional skill, not snapshot
risk-adjusted ratios*. Five pillars combine into a composite 0-100 score:

  P1 - Regime-Conditional Forward Alpha (35%)
       Classify Mid Cap benchmark history into 4 regimes (StrongBull /
       MildBull / Correction / Bear). Estimate per-regime alpha for each
       fund. Project current regime probabilities forward through an
       empirical 52-week transition matrix and integrate fund alpha over
       the resulting forward regime mix.

  P2 - Bayesian Skill Persistence (20%)
       Compute non-overlapping 6-month block alphas. Apply James-Stein
       shrinkage of each fund's mean block-alpha toward the peer mean
       (heavier shrinkage when within-fund variance is high). Add a
       per-fund rank-stability term measuring how steadily a fund
       maintains its peer rank over time.

  P3 - SIP-Specific Forward Outcome (20%)
       Simulate rolling 1-year monthly SIP XIRRs (1st-of-month buys at
       next available NAV). Score blends median XIRR, 25th-percentile
       XIRR (downside) and hit-rate vs benchmark SIP XIRR. Directly
       addresses the task's monthly-SIP horizon.

  P4 - Tail Risk & Recovery Asymmetry (15%)
       CDaR-5%, recovery half-life from worst trough, downside-skew
       penalty and the downside / upside vol ratio - rolled into a
       single tail score (less tail pain -> higher score).

  P5 - Active Share & Liquidity (10%, holdings-aware, drops out if
       holdings missing)
       Top-10 concentration, equity-holding count, mean abs change3m
       (turnover proxy) and a piecewise mid-cap AUM penalty (>25k Cr).

Composite is the cross-sectional percentile rank of each pillar,
weight-averaged. A graduated confidence haircut by data history
(<1Y -> 0.55, ..., 5Y+ -> 1.00) tames lucky short-history funds.

Why this differs from the prior Claude / Codex / Gemini Mid Cap algos
--------------------------------------------------------------------
- Prior Claude: 25-metric weekly multi-factor with fixed weights, a single
  static alpha and a bear-period alpha. We replace static alpha with a
  forward-projected expectation under regime probabilities, and we add
  Bayesian shrinkage and explicit SIP XIRR.
- Codex: walk-forward weight TUNING to maximize forward IC. We instead
  use theory-driven weights and report per-pillar IC as a diagnostic
  (different research stance: explainability over fit).
- Gemini: daily-data 5Y standardised window with simple Sharpe / win-rate.
  We use weekly returns (matching the actual NAV cadence) and condition on
  market regime instead of treating all weeks the same.

Output
------
results/Mid Cap_Claude.csv with required columns mfId, name, rank, score,
data_days, cagr_3y, cagr_5y plus diagnostic pillar columns.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
# Configuration
# ===================================================================
SECTOR = "Mid Cap"
SUBSECTOR = "Mid Cap Fund"
BENCHMARK = "Mid Cap"  # MfDataProvider resolves to .NIMI150
RISK_FREE_RATE = 0.065  # India ~6.5% T-bill proxy
WEEKS_PER_YEAR = 52

# Lookback constants (in weeks)
LB_3M = 13
LB_6M = 26
LB_1Y = 52
LB_2Y = 104
LB_3Y = 156
LB_5Y = 260

MIN_WEEKS_FOR_ANALYSIS = 30   # below this fund gets score=0
SIP_MONTHLY_AMOUNT = 10000.0  # rupees per buy (only ratio matters for XIRR)

# Pillar weights (must sum to 1.0)
PILLAR_WEIGHTS = {
    "p1_regime_alpha":     0.35,
    "p2_skill_posterior":  0.20,
    "p3_sip_outcome":      0.20,
    "p4_tail_recovery":    0.15,
    "p5_active_liquidity": 0.10,
}
assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9

REGIMES = ("StrongBull", "MildBull", "Correction", "Bear")
N_REGIMES = len(REGIMES)

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Claude.csv"


# ===================================================================
# Data Cleaning Helpers
# ===================================================================

def clean_nav_to_series(df: pd.DataFrame) -> pd.Series:
    """Convert raw chart DataFrame to a sorted, dedup'd, tz-naive NAV Series."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["timestamp", "nav"])
    out = out[out["nav"] > 0]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    s = out.set_index("timestamp")["nav"]
    if s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    return s


def weekly_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def annualised_return(nav: pd.Series, weeks: int) -> Optional[float]:
    """CAGR over the trailing `weeks` periods. None if not enough data."""
    if nav is None or len(nav) < weeks + 1:
        return None
    start = nav.iloc[-(weeks + 1)]
    end = nav.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    years = weeks / WEEKS_PER_YEAR
    return float((end / start) ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series) -> Optional[float]:
    if returns is None or len(returns) < 12:
        return None
    return float(returns.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def downside_deviation(returns: pd.Series, mar: float = RISK_FREE_RATE) -> Optional[float]:
    if returns is None or len(returns) < 12:
        return None
    weekly_mar = (1 + mar) ** (1 / WEEKS_PER_YEAR) - 1
    diff = returns - weekly_mar
    neg = diff[diff < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(WEEKS_PER_YEAR))


def sortino_calc(cagr: Optional[float], dd: Optional[float]) -> Optional[float]:
    if cagr is None or dd is None or dd <= 1e-12:
        return None
    return float((cagr - RISK_FREE_RATE) / dd)


def alpha_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Annualised Jensen alpha and beta via OLS."""
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < 12:
        return None, None
    x = aligned["b"].values
    y = aligned["f"].values
    var_x = np.var(x, ddof=1)
    if var_x < 1e-12:
        return None, None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    alpha_w = float(np.mean(y) - rf_w - beta * (np.mean(x) - rf_w))
    return float(alpha_w * WEEKS_PER_YEAR), beta


# ===================================================================
# Regime Classifier (Pillar 1 backbone)
# ===================================================================

# Threshold definitions (Indian mid-cap historical context):
# - StrongBull : 6M return > +15% AND not in drawdown > -5%
# - MildBull   : 6M return >= 0 AND drawdown > -10%
# - Correction : drawdown between -10% and -20%, OR 6M return between -10% and 0
# - Bear       : drawdown <= -20% OR 6M return <= -10%
def classify_week(ret_6m: float, dd_1y: float) -> int:
    """Hard-classify a single week into 0/1/2/3 regime index. NaN -> -1."""
    if pd.isna(ret_6m) or pd.isna(dd_1y):
        return -1
    if dd_1y <= -0.20 or ret_6m <= -0.10:
        return 3   # Bear
    if dd_1y <= -0.10 or ret_6m < 0.0:
        return 2   # Correction
    if ret_6m >= 0.15 and dd_1y > -0.05:
        return 0   # StrongBull
    return 1       # MildBull


def regime_series(bench_nav: pd.Series) -> pd.Series:
    """Hard regime label (0..3) per week of benchmark, NaN where lookback short."""
    if len(bench_nav) < LB_6M + 4:
        return pd.Series(dtype="float64", index=bench_nav.index)
    ret_6m = bench_nav.pct_change(LB_6M)
    peak_1y = bench_nav.rolling(LB_1Y, min_periods=10).max()
    dd_1y = bench_nav / peak_1y - 1.0
    out = pd.Series(np.nan, index=bench_nav.index, dtype="float64")
    for ts in bench_nav.index:
        r = ret_6m.get(ts, np.nan)
        d = dd_1y.get(ts, np.nan)
        cls = classify_week(r, d)
        out[ts] = float(cls) if cls >= 0 else np.nan
    return out


def empirical_forward_mix(reg: pd.Series, horizon_weeks: int = LB_1Y) -> np.ndarray:
    """
    For each starting regime r, compute the mean fraction of the next
    `horizon_weeks` weeks spent in each regime r'. Returns a row-stochastic
    4x4 matrix M[r, r'].
    """
    valid = reg.dropna().astype(int)
    if len(valid) < horizon_weeks + 4:
        # Fallback: identity-ish (assume regime persists)
        M = np.eye(N_REGIMES) * 0.7 + np.ones((N_REGIMES, N_REGIMES)) * (0.3 / N_REGIMES)
        return M

    arr = valid.values
    M = np.zeros((N_REGIMES, N_REGIMES))
    counts = np.zeros(N_REGIMES)
    for i in range(len(arr) - horizon_weeks):
        r = int(arr[i])
        forward = arr[i + 1: i + 1 + horizon_weeks]
        if len(forward) < horizon_weeks:
            continue
        for rp in range(N_REGIMES):
            M[r, rp] += np.mean(forward == rp)
        counts[r] += 1
    for r in range(N_REGIMES):
        if counts[r] > 0:
            M[r] /= counts[r]
        else:
            # Regime never observed in training window - assume mean reversion to the
            # peer-empirical regime distribution.
            M[r] = np.bincount(arr, minlength=N_REGIMES) / max(1, len(arr))
    return M


def soft_current_regime_probs(reg: pd.Series, recent_weeks: int = 8) -> np.ndarray:
    """Use the recent N weeks' empirical mix as a soft probability over regimes."""
    recent = reg.dropna().iloc[-recent_weeks:].astype(int)
    if len(recent) == 0:
        return np.ones(N_REGIMES) / N_REGIMES
    counts = np.bincount(recent.values, minlength=N_REGIMES).astype(float)
    if counts.sum() == 0:
        return np.ones(N_REGIMES) / N_REGIMES
    return counts / counts.sum()


# ===================================================================
# PILLAR 1 - Regime-Conditional Forward Alpha
# ===================================================================

def per_regime_alpha(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    reg: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-regime annualised alpha for one fund. Returns (alpha_4, n_obs_4).
    Regimes with <8 weekly observations are NaN (caller imputes from peer).
    """
    aligned = pd.concat({"f": fund_ret, "b": bench_ret, "r": reg}, axis=1).dropna()
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    alphas = np.full(N_REGIMES, np.nan)
    n_obs = np.zeros(N_REGIMES, dtype=int)
    for r in range(N_REGIMES):
        sub = aligned[aligned["r"].astype(int) == r]
        n = len(sub)
        if n < 8:
            n_obs[r] = n
            continue
        x = sub["b"].values - rf_w
        y = sub["f"].values - rf_w
        var_x = np.var(x, ddof=1)
        if var_x < 1e-12:
            n_obs[r] = n
            continue
        beta = np.cov(x, y, ddof=1)[0, 1] / var_x
        alpha_w = float(np.mean(y) - beta * np.mean(x))
        # Two-stage taming for noisy per-regime alpha estimates:
        #  1) Sample-size shrink toward zero (more shrink when n is small).
        #     n=8 => ~50%, n=52 => ~14%, n=156 => ~9%.
        #  2) Hard clip to +/-50% annualised. Realistic equity alpha never
        #     reaches that range over multi-month windows.
        shrink = 1.0 / (1.0 + 4.0 / np.sqrt(n))
        annual = float(np.clip(alpha_w * WEEKS_PER_YEAR * shrink, -0.50, 0.50))
        alphas[r] = annual
        n_obs[r] = n
    return alphas, n_obs


def peer_mean_per_regime(per_fund_alphas: Dict[str, np.ndarray]) -> np.ndarray:
    """Cross-sectional mean of per-regime alphas (ignores NaN). Length 4."""
    if not per_fund_alphas:
        return np.zeros(N_REGIMES)
    stack = np.vstack(list(per_fund_alphas.values()))  # n_funds x 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(stack, axis=0)
    means[np.isnan(means)] = 0.0
    return means


def expected_forward_alpha(
    fund_alphas: np.ndarray,
    peer_alphas: np.ndarray,
    current_probs: np.ndarray,
    transition: np.ndarray,
    min_observed_regimes: int = 2,
) -> Optional[float]:
    """E[forward 1Y alpha] = (current_probs @ transition) . filled_alphas.

    Requires at least `min_observed_regimes` regimes with non-NaN fund alpha.
    Otherwise returns None (insufficient evidence for a regime-conditional
    estimate; the composite scorer will skip P1 for this fund).
    """
    n_observed = int(np.sum(~np.isnan(fund_alphas)))
    if n_observed < min_observed_regimes:
        return None
    filled = np.where(np.isnan(fund_alphas), peer_alphas, fund_alphas)
    if np.any(np.isnan(filled)):
        return None
    forward_mix = current_probs @ transition  # 4-vector (rows are stochastic)
    return float(forward_mix @ filled)


def regime_capture_spread(
    fund_ret: pd.Series, bench_ret: pd.Series, reg: pd.Series
) -> Optional[float]:
    """
    Up-capture in StrongBull+MildBull regimes minus down-capture in
    Correction+Bear regimes. A higher spread = better up/down asymmetry.
    """
    aligned = pd.concat({"f": fund_ret, "b": bench_ret, "r": reg}, axis=1).dropna()
    if len(aligned) < 30:
        return None

    up_mask = aligned["r"].astype(int).isin([0, 1])
    dn_mask = aligned["r"].astype(int).isin([2, 3])
    if up_mask.sum() < 8 or dn_mask.sum() < 8:
        return None

    bench_up = aligned.loc[up_mask, "b"].mean()
    fund_up = aligned.loc[up_mask, "f"].mean()
    bench_dn = aligned.loc[dn_mask, "b"].mean()
    fund_dn = aligned.loc[dn_mask, "f"].mean()
    if abs(bench_up) < 1e-6 or abs(bench_dn) < 1e-6:
        return None
    up_cap = fund_up / bench_up
    dn_cap = fund_dn / bench_dn
    return float(up_cap - dn_cap)


# ===================================================================
# PILLAR 2 - Bayesian Skill Persistence
# ===================================================================

def block_alphas(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    block_weeks: int = LB_6M,
    max_blocks: int = 10,
) -> np.ndarray:
    """
    Annualised alphas over non-overlapping trailing 6M blocks (most recent first).
    Returns up to `max_blocks` values; empty array if no full blocks.
    """
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < block_weeks:
        return np.array([])
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    out = []
    end = len(aligned)
    while end - block_weeks >= 0 and len(out) < max_blocks:
        sub = aligned.iloc[end - block_weeks: end]
        x = sub["b"].values - rf_w
        y = sub["f"].values - rf_w
        var_x = np.var(x, ddof=1)
        if var_x >= 1e-12:
            beta = np.cov(x, y, ddof=1)[0, 1] / var_x
            alpha_w = float(np.mean(y) - beta * np.mean(x))
            out.append(alpha_w * WEEKS_PER_YEAR)
        end -= block_weeks
    return np.array(out)


def james_stein_shrunk_alphas(
    fund_block_alphas: Dict[str, np.ndarray]
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Posterior fund alpha = peer_mean + lambda * (fund_mean - peer_mean),
    where lambda is the James-Stein shrinkage weight depending on the
    ratio of within-fund to between-fund variance.

    Returns (shrunk_alpha_per_fund, peer_mean, shrinkage_lambda_per_fund).
    """
    means: Dict[str, float] = {}
    within_var: Dict[str, float] = {}
    for fid, blocks in fund_block_alphas.items():
        if len(blocks) == 0:
            continue
        means[fid] = float(np.mean(blocks))
        within_var[fid] = float(np.var(blocks, ddof=1) / len(blocks)) if len(blocks) > 1 else 0.04 ** 2
    if not means:
        return {}, 0.0, {}

    mu_arr = np.array(list(means.values()))
    peer_mean = float(np.mean(mu_arr))
    between_var = float(np.var(mu_arr, ddof=1)) if len(mu_arr) > 1 else 0.0

    shrunk: Dict[str, float] = {}
    lams: Dict[str, float] = {}
    for fid, m in means.items():
        wv = within_var.get(fid, 0.0)
        denom = between_var + wv
        if denom < 1e-12:
            lam = 0.5
        else:
            lam = float(between_var / denom)
        lam = max(0.0, min(1.0, lam))
        shrunk[fid] = peer_mean + lam * (m - peer_mean)
        lams[fid] = lam
    return shrunk, peer_mean, lams


def rank_stability_score(blocks: np.ndarray) -> Optional[float]:
    """
    Stability of fund's own block alphas: low std-vs-mean (signed) is good.
    Returns mean / (1 + std). Higher = more consistent alpha.
    """
    if len(blocks) < 3:
        return None
    m = float(np.mean(blocks))
    s = float(np.std(blocks, ddof=1))
    return m / (1.0 + abs(s) + 0.02)


def persistence_diagnostic_ic(
    aligned_navs: Dict[str, pd.Series],
    bench_nav: pd.Series,
    eval_step: int = LB_3M,
    n_evals: int = 8,
) -> Optional[float]:
    """
    Cross-sectional Spearman IC between past-1Y rank (by simple alpha)
    and forward-1Y rank (by simple alpha), averaged across `n_evals`
    historical eval points spaced `eval_step` weeks apart.
    A diagnostic indicator of how persistent skill rankings are in this peer
    group. Returns a single float or None if data is too thin.
    """
    common_idx = bench_nav.index
    n = len(common_idx)
    if n < LB_3Y + LB_1Y + 4:
        return None

    last_eval = n - LB_1Y - 1
    first_eval = max(LB_1Y, last_eval - eval_step * (n_evals - 1))
    eval_points = list(range(first_eval, last_eval + 1, eval_step))
    if not eval_points:
        return None

    ics: List[float] = []
    bench_w = weekly_returns(bench_nav)
    for ep in eval_points[-n_evals:]:
        past_alphas: Dict[str, float] = {}
        fwd_alphas: Dict[str, float] = {}
        ep_ts = common_idx[ep]
        ep_fwd_ts = common_idx[min(ep + LB_1Y, n - 1)]
        for fid, fnav in aligned_navs.items():
            past_n = fnav.loc[fnav.index <= ep_ts]
            past_n = past_n.dropna()
            if len(past_n) < LB_1Y + 4:
                continue
            past_n = past_n.iloc[-LB_1Y - 1:]
            past_ret = weekly_returns(past_n)
            bench_past = bench_w.reindex(past_ret.index).dropna()
            past_ret = past_ret.reindex(bench_past.index).dropna()
            a, _ = alpha_beta(past_ret, bench_past)
            if a is None:
                continue
            past_alphas[fid] = a

            fwd_n = fnav.loc[(fnav.index > ep_ts) & (fnav.index <= ep_fwd_ts)]
            fwd_n = fwd_n.dropna()
            if len(fwd_n) < LB_1Y - 4:
                continue
            fwd_ret = weekly_returns(fwd_n)
            bench_fwd = bench_w.reindex(fwd_ret.index).dropna()
            fwd_ret = fwd_ret.reindex(bench_fwd.index).dropna()
            af, _ = alpha_beta(fwd_ret, bench_fwd)
            if af is None:
                continue
            fwd_alphas[fid] = af

        common = sorted(set(past_alphas) & set(fwd_alphas))
        if len(common) < 8:
            continue
        p_arr = pd.Series([past_alphas[k] for k in common])
        f_arr = pd.Series([fwd_alphas[k] for k in common])
        if p_arr.std() < 1e-9 or f_arr.std() < 1e-9:
            continue
        ic = float(p_arr.corr(f_arr, method="spearman"))
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return None
    return float(np.mean(ics))


# ===================================================================
# PILLAR 3 - SIP-Specific Forward Outcome (XIRR simulation)
# ===================================================================

def _xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    """Internal: solve for annualised XIRR via Brent's method."""
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    days = np.array([(cf[0] - t0).days for cf in cashflows], dtype=float)
    amts = np.array([cf[1] for cf in cashflows], dtype=float)
    if np.all(amts >= 0) or np.all(amts <= 0):
        return None  # need both signs

    def npv(rate: float) -> float:
        return float(np.sum(amts / (1.0 + rate) ** (days / 365.0)))

    try:
        # Bracket the root.  For sane SIP outcomes this works.
        return float(brentq(npv, -0.95, 5.0, xtol=1e-7, maxiter=200))
    except (ValueError, RuntimeError):
        # Try a wider lower bracket if needed
        try:
            return float(brentq(npv, -0.999, 20.0, xtol=1e-6, maxiter=200))
        except (ValueError, RuntimeError):
            return None


def sip_xirr_for_window(
    nav: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    monthly_amount: float = SIP_MONTHLY_AMOUNT,
) -> Optional[float]:
    """
    Simulate monthly SIP buys on the 1st of each month from `start` to `end`.
    Buy at the first available NAV on/after the 1st (no peeking). Final
    valuation uses the last NAV observed at or before `end`.
    """
    if nav is None or len(nav) == 0 or end <= start:
        return None
    buys = pd.date_range(start=start, end=end, freq="MS")
    if len(buys) < 6:
        return None

    units = 0.0
    cashflows: List[Tuple[pd.Timestamp, float]] = []
    last_buy_dt: Optional[pd.Timestamp] = None
    for buy_target in buys:
        idx = nav.index.searchsorted(buy_target, side="left")
        if idx >= len(nav):
            continue
        actual_dt = nav.index[idx]
        if actual_dt > end:
            continue
        nav_v = float(nav.iloc[idx])
        if nav_v <= 0:
            continue
        units += monthly_amount / nav_v
        cashflows.append((actual_dt, -monthly_amount))
        last_buy_dt = actual_dt

    if len(cashflows) < 6 or units <= 0:
        return None

    # Final valuation - last NAV observation at or before end
    end_idx = nav.index.searchsorted(end, side="right") - 1
    if end_idx < 0:
        return None
    final_dt = nav.index[end_idx]
    if last_buy_dt is not None and final_dt < last_buy_dt:
        # No NAV available after final buy; nothing to value forward.
        return None
    final_v = float(nav.iloc[end_idx])
    final_value = units * final_v
    # If final_dt coincides with the last buy, net the buy out of the final
    # cashflow so XIRR sees one signed cashflow at that timestamp.
    if last_buy_dt is not None and final_dt == last_buy_dt:
        # Sum of (final_value, -monthly_amount) at the same date.  Replace
        # the last buy entry with the combined net.
        net = cashflows[-1][1] + final_value  # -amount + final_value
        cashflows[-1] = (final_dt, net)
    else:
        cashflows.append((final_dt, final_value))
    return _xirr(cashflows)


def rolling_sip_xirrs(
    nav: pd.Series,
    window_years: int = 1,
    step_weeks: int = 4,
    bench_nav: Optional[pd.Series] = None,
) -> Dict[str, Optional[float]]:
    """
    Roll a 1Y SIP window forward in `step_weeks` increments through the
    available NAV history. Return median, p25, hit-rate vs benchmark SIP,
    and number of windows.
    """
    if nav is None or len(nav) < LB_1Y + 4:
        return {"sip_p50": None, "sip_p25": None, "sip_hit": None, "sip_n": 0}

    earliest = nav.index[0]
    latest = nav.index[-1]
    fund_xirrs: List[float] = []
    bench_xirrs: List[float] = []

    cursor = (earliest + pd.DateOffset(months=1)).to_period("M").to_timestamp()
    last_window_end = latest
    while True:
        end = cursor + pd.DateOffset(years=window_years)
        if end > last_window_end:
            break
        fx = sip_xirr_for_window(nav, cursor, end)
        if fx is not None:
            fund_xirrs.append(fx)
            if bench_nav is not None:
                bx = sip_xirr_for_window(bench_nav, cursor, end)
                bench_xirrs.append(bx if bx is not None else np.nan)
        cursor = cursor + pd.DateOffset(weeks=step_weeks)

    if not fund_xirrs:
        return {"sip_p50": None, "sip_p25": None, "sip_hit": None, "sip_n": 0}

    arr = np.array(fund_xirrs, dtype=float)
    p50 = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    hit = None
    if bench_xirrs:
        ba = np.array(bench_xirrs, dtype=float)
        common_mask = ~np.isnan(ba)
        if common_mask.sum() >= 3:
            hit = float(np.mean(arr[common_mask] > ba[common_mask]))
    return {
        "sip_p50": p50,
        "sip_p25": p25,
        "sip_hit": hit,
        "sip_n": int(len(arr)),
    }


# ===================================================================
# PILLAR 4 - Tail Risk & Recovery Asymmetry
# ===================================================================

def cdar_5pct(nav: pd.Series) -> Optional[float]:
    """Mean of the worst 5% of running drawdown-from-peak observations."""
    if nav is None or len(nav) < 30:
        return None
    dd = (nav / nav.cummax() - 1.0).dropna()
    if len(dd) == 0:
        return None
    n_tail = max(2, int(np.ceil(len(dd) * 0.05)))
    worst = np.sort(dd.values)[:n_tail]
    return float(np.mean(worst))


def recovery_halflife_weeks(nav: pd.Series) -> Optional[float]:
    """
    Weeks elapsed from the deepest drawdown trough until 50% of that
    drawdown has been clawed back. None if drawdown is shallow (>-5%).
    If never recovered to half, returns the remaining weeks of history.
    """
    if nav is None or len(nav) < 30:
        return None
    dd = (nav / nav.cummax() - 1.0).dropna()
    if len(dd) == 0:
        return None
    trough_ts = dd.idxmin()
    trough_val = float(dd.loc[trough_ts])
    if trough_val >= -0.05:
        return None
    target = trough_val * 0.5
    after = dd.loc[trough_ts:]
    recovered = after[after >= target]
    if len(recovered) == 0:
        return float(len(after))  # never recovered (penalising)
    rec_ts = recovered.index[0]
    delta_days = (rec_ts - trough_ts).days
    return float(max(0.0, delta_days / 7.0))


def downside_to_upside_vol(returns: pd.Series) -> Optional[float]:
    """std(returns where r<0) / std(returns where r>0). Lower = better."""
    if returns is None or len(returns) < 20:
        return None
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    if len(pos) < 5 or len(neg) < 5:
        return None
    up = pos.std(ddof=1)
    dn = neg.std(ddof=1)
    if up < 1e-12:
        return None
    return float(dn / up)


def tail_score_components(
    fund_nav: pd.Series, returns: pd.Series
) -> Dict[str, Optional[float]]:
    cdar = cdar_5pct(fund_nav)
    half = recovery_halflife_weeks(fund_nav)
    dn_up = downside_to_upside_vol(returns)
    skew = float(returns.skew()) if len(returns) >= 30 else None
    return {
        "p4_cdar5":            cdar,
        "p4_recovery_half_w":  half,
        "p4_dn_up_vol":        dn_up,
        "p4_skew":             skew,
    }


# ===================================================================
# PILLAR 5 - Active Share & Liquidity (holdings + AUM)
# ===================================================================

def aum_liquidity_haircut(aum_cr: float) -> float:
    """
    Mid-cap-specific multiplicative penalty applied to P5 score.
    AUM is in INR Crore (as provided by API).
      <= 15 000 Cr -> 1.00
      <= 25 000 Cr -> 0.95
      <= 40 000 Cr -> 0.85
      <= 60 000 Cr -> 0.75
      >  60 000 Cr -> 0.65
    Mid cap impact cost rises sharply once a fund must hold larger blocks.
    """
    if aum_cr is None or pd.isna(aum_cr) or aum_cr <= 0:
        return 1.0
    a = float(aum_cr)
    if a <= 15000:
        return 1.0
    if a <= 25000:
        return 0.95
    if a <= 40000:
        return 0.85
    if a <= 60000:
        return 0.75
    return 0.65


def holdings_metrics(holdings: List[Dict]) -> Dict[str, Optional[float]]:
    """
    Compute portfolio concentration and turnover proxies from holdings list.
    Returns all-None when holdings missing or unusable.
    """
    empty = {
        "p5_top10_conc":    None,
        "p5_n_holdings":    None,
        "p5_avg_change3m":  None,
    }
    if not holdings:
        return empty
    df = pd.DataFrame(holdings)
    if "type" in df.columns:
        eq = df[df["type"].astype(str).str.lower() == "equity"]
    else:
        eq = df
    if len(eq) == 0 or "latest" not in eq.columns:
        return empty
    alloc = pd.to_numeric(eq["latest"], errors="coerce").fillna(0.0)
    alloc = alloc[alloc > 0]
    if len(alloc) == 0:
        return empty
    top10 = float(alloc.sort_values(ascending=False).head(10).sum())
    n = int(len(alloc))
    avg_ch3m: Optional[float] = None
    if "change3m" in eq.columns:
        ch = pd.to_numeric(eq["change3m"], errors="coerce").dropna()
        if len(ch) > 0:
            avg_ch3m = float(ch.abs().mean())
    return {
        "p5_top10_conc":    top10,
        "p5_n_holdings":    n,
        "p5_avg_change3m":  avg_ch3m,
    }


# ===================================================================
# Per-Fund Analysis (collect raw metrics for all 5 pillars)
# ===================================================================

def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    bench_reg: pd.Series,
    name: str,
    aum_cr: float,
    holdings: Optional[List[Dict]],
) -> Dict:
    """Compute raw P1..P5 inputs (and CAGRs / diagnostics) for one fund."""
    n_weeks = len(fund_nav)
    first_ts = fund_nav.index.min() if n_weeks else None
    last_ts = fund_nav.index.max() if n_weeks else None
    data_days = int((last_ts - first_ts).days) + 1 if first_ts is not None else 0

    res: Dict = {
        "mfId": mf_id,
        "name": name,
        "aum": round(float(aum_cr or 0.0), 2),
        "data_days": data_days,
        "data_weeks": n_weeks,
    }

    # CAGRs
    res["cagr_1y"] = annualised_return(fund_nav, LB_1Y)
    res["cagr_3y"] = annualised_return(fund_nav, LB_3Y)
    res["cagr_5y"] = annualised_return(fund_nav, LB_5Y)

    if n_weeks < MIN_WEEKS_FOR_ANALYSIS:
        # too short - mark and bail out (will get score=0 later)
        res["_skip_reason"] = f"only {n_weeks} weekly NAV points"
        return res

    rets = weekly_returns(fund_nav)
    bench_rets = weekly_returns(bench_nav).reindex(rets.index).dropna()
    rets = rets.reindex(bench_rets.index).dropna()
    if len(rets) < MIN_WEEKS_FOR_ANALYSIS:
        res["_skip_reason"] = f"only {len(rets)} aligned weekly returns"
        return res

    res["volatility"] = annualised_volatility(rets)
    primary_cagr = res["cagr_3y"] or res["cagr_5y"] or res["cagr_1y"]
    dd = downside_deviation(rets)
    res["sortino"] = sortino_calc(primary_cagr, dd)
    a, b = alpha_beta(rets, bench_rets)
    res["alpha"] = a
    res["beta"] = b

    # P1 - per-regime fund alpha
    reg_aligned = bench_reg.reindex(rets.index)
    per_reg_alpha, per_reg_n = per_regime_alpha(rets, bench_rets, reg_aligned)
    res["_per_regime_alpha"] = per_reg_alpha   # internal, dropped before write
    res["_per_regime_n"] = per_reg_n
    res["p1_regime_capture_spread"] = regime_capture_spread(rets, bench_rets, reg_aligned)

    # P2 - block alphas (handed off for cross-sectional shrinkage later)
    blocks = block_alphas(rets, bench_rets, block_weeks=LB_6M, max_blocks=10)
    res["_block_alphas"] = blocks
    res["p2_block_alpha_n"] = int(len(blocks))
    res["p2_personal_stability"] = rank_stability_score(blocks)

    # P3 - rolling SIP XIRRs vs benchmark
    sip = rolling_sip_xirrs(fund_nav, window_years=1, step_weeks=4, bench_nav=bench_nav)
    res["p3_sip_xirr_p50"] = sip["sip_p50"]
    res["p3_sip_xirr_p25"] = sip["sip_p25"]
    res["p3_sip_hit_vs_bench"] = sip["sip_hit"]
    res["p3_sip_n_windows"] = sip["sip_n"]

    # P4 - tail metrics
    res.update(tail_score_components(fund_nav, rets))

    # P5 - holdings + AUM
    res.update(holdings_metrics(holdings))
    res["p5_aum_haircut"] = aum_liquidity_haircut(aum_cr)

    return res


# ===================================================================
# Composite Scoring
# ===================================================================

def percentile_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Cross-sectional percentile rank (0..100), preserves NaN."""
    ranked = s.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ranked * 100.0


def confidence_factor(data_days: int) -> float:
    """Graduated haircut by available history."""
    if data_days < 365:
        return 0.55
    if data_days < 2 * 365:
        return 0.75
    if data_days < 3 * 365:
        return 0.88
    if data_days < 5 * 365:
        return 0.95
    return 1.00


def compute_pillar_scores(
    df: pd.DataFrame,
    peer_per_regime: np.ndarray,
    current_regime_probs: np.ndarray,
    transition_matrix: np.ndarray,
    shrunk_alpha: Dict[str, float],
    shrunk_lambda: Dict[str, float],
) -> pd.DataFrame:
    """
    Convert raw per-fund metrics into the 5 pillar percentile scores
    (each 0-100) and a composite score.
    """
    df = df.copy()

    # ----- P1: regime-conditional forward alpha -----
    p1_alpha_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        per_reg = row.get("_per_regime_alpha")
        if per_reg is None or (isinstance(per_reg, float) and pd.isna(per_reg)):
            p1_alpha_vals.append(None)
            continue
        ev = expected_forward_alpha(
            np.asarray(per_reg, dtype=float),
            peer_per_regime,
            current_regime_probs,
            transition_matrix,
        )
        p1_alpha_vals.append(ev)
    df["p1_forward_alpha"] = p1_alpha_vals

    # P1 score requires a valid forward alpha (>=2 regimes observed). The
    # capture spread is an auxiliary signal that contributes 25% only when
    # forward alpha is also valid - this prevents very-short-history funds
    # from gaming P1 via a single-regime capture-spread.
    p1_pct_a = percentile_rank(df["p1_forward_alpha"], higher_is_better=True)
    p1_pct_c = percentile_rank(df["p1_regime_capture_spread"], higher_is_better=True)
    p1_combined = []
    for a, c in zip(p1_pct_a, p1_pct_c):
        if pd.notna(a) and pd.notna(c):
            p1_combined.append(0.75 * a + 0.25 * c)
        elif pd.notna(a):
            p1_combined.append(a)
        else:
            # forward alpha missing -> P1 not reliable -> NaN (skipped in composite)
            p1_combined.append(np.nan)
    df["p1_score"] = p1_combined

    # ----- P2: Bayesian skill posterior + rank stability -----
    df["p2_shrunk_alpha"] = df["mfId"].map(shrunk_alpha)
    df["p2_shrink_lambda"] = df["mfId"].map(shrunk_lambda)
    p2_pct_a = percentile_rank(df["p2_shrunk_alpha"], higher_is_better=True)
    p2_pct_s = percentile_rank(df["p2_personal_stability"], higher_is_better=True)
    p2_combined = []
    for a, s in zip(p2_pct_a, p2_pct_s):
        if pd.notna(a) and pd.notna(s):
            p2_combined.append(0.7 * a + 0.3 * s)
        elif pd.notna(a):
            p2_combined.append(a)
        elif pd.notna(s):
            p2_combined.append(s)
        else:
            p2_combined.append(np.nan)
    df["p2_score"] = p2_combined

    # ----- P3: SIP outcome (require >= 6 rolling SIP windows) -----
    min_sip_windows = 6
    df_p3 = df.copy()
    insufficient = df_p3["p3_sip_n_windows"].fillna(0) < min_sip_windows
    for col in ("p3_sip_xirr_p50", "p3_sip_xirr_p25", "p3_sip_hit_vs_bench"):
        df_p3.loc[insufficient, col] = np.nan
    p3_pct_p50 = percentile_rank(df_p3["p3_sip_xirr_p50"], higher_is_better=True)
    p3_pct_p25 = percentile_rank(df_p3["p3_sip_xirr_p25"], higher_is_better=True)
    p3_pct_hit = percentile_rank(df_p3["p3_sip_hit_vs_bench"], higher_is_better=True)
    p3_combined = []
    for a, b_, c in zip(p3_pct_p50, p3_pct_p25, p3_pct_hit):
        parts: List[float] = []
        weights: List[float] = []
        if pd.notna(a):
            parts.append(a); weights.append(0.45)
        if pd.notna(b_):
            parts.append(b_); weights.append(0.35)
        if pd.notna(c):
            parts.append(c); weights.append(0.20)
        if parts:
            w = np.array(weights); w /= w.sum()
            p3_combined.append(float(np.dot(np.array(parts), w)))
        else:
            p3_combined.append(np.nan)
    df["p3_score"] = p3_combined

    # ----- P4: tail risk & recovery (require >= 2 years of weekly data,
    #            otherwise the fund hasn't had time to experience a real
    #            drawdown and its CDaR-5% looks artificially benign) -----
    df_p4 = df.copy()
    too_short = df_p4["data_weeks"].fillna(0) < LB_2Y
    for col in ("p4_cdar5", "p4_recovery_half_w", "p4_dn_up_vol", "p4_skew"):
        df_p4.loc[too_short, col] = np.nan
    p4_pct_cdar = percentile_rank(df_p4["p4_cdar5"], higher_is_better=True)        # cdar is negative; less negative = better
    p4_pct_half = percentile_rank(df_p4["p4_recovery_half_w"], higher_is_better=False)  # fewer weeks = better
    p4_pct_dnup = percentile_rank(df_p4["p4_dn_up_vol"], higher_is_better=False)   # lower = better
    p4_pct_skew = percentile_rank(df_p4["p4_skew"], higher_is_better=True)         # higher (less negative) = better
    p4_combined = []
    for a, b_, c, d in zip(p4_pct_cdar, p4_pct_half, p4_pct_dnup, p4_pct_skew):
        parts: List[float] = []
        weights: List[float] = []
        if pd.notna(a):
            parts.append(a); weights.append(0.40)
        if pd.notna(b_):
            parts.append(b_); weights.append(0.25)
        if pd.notna(c):
            parts.append(c); weights.append(0.20)
        if pd.notna(d):
            parts.append(d); weights.append(0.15)
        if parts:
            w = np.array(weights); w /= w.sum()
            p4_combined.append(float(np.dot(np.array(parts), w)))
        else:
            p4_combined.append(np.nan)
    df["p4_score"] = p4_combined

    # ----- P5: active share & AUM (drops out if no holdings) -----
    p5_pct_top10 = percentile_rank(df["p5_top10_conc"], higher_is_better=False)  # lower concentration = better
    p5_pct_n = percentile_rank(df["p5_n_holdings"], higher_is_better=True)
    p5_pct_ch3m = percentile_rank(df["p5_avg_change3m"], higher_is_better=True)  # more activity = better signal of management
    p5_combined = []
    has_any_p5 = False
    for a, b_, c, h in zip(p5_pct_top10, p5_pct_n, p5_pct_ch3m, df["p5_aum_haircut"]):
        parts: List[float] = []
        weights: List[float] = []
        if pd.notna(a):
            parts.append(a); weights.append(0.45)
        if pd.notna(b_):
            parts.append(b_); weights.append(0.30)
        if pd.notna(c):
            parts.append(c); weights.append(0.25)
        if parts:
            has_any_p5 = True
            w = np.array(weights); w /= w.sum()
            base = float(np.dot(np.array(parts), w))
            p5_combined.append(base * float(h or 1.0))
        else:
            p5_combined.append(np.nan)
    df["p5_score"] = p5_combined
    df.attrs["_p5_active"] = has_any_p5

    # ----- Composite -----
    weights_used = dict(PILLAR_WEIGHTS)
    if not has_any_p5:
        # Drop P5 and renormalise the remaining four pillars.
        del weights_used["p5_active_liquidity"]
        ssum = sum(weights_used.values())
        weights_used = {k: v / ssum for k, v in weights_used.items()}

    pillar_cols = {
        "p1_regime_alpha":     "p1_score",
        "p2_skill_posterior":  "p2_score",
        "p3_sip_outcome":      "p3_score",
        "p4_tail_recovery":    "p4_score",
        "p5_active_liquidity": "p5_score",
    }
    composite_vals = []
    for _, row in df.iterrows():
        used_w = []
        used_v = []
        for k, w in weights_used.items():
            col = pillar_cols[k]
            v = row.get(col, np.nan)
            if pd.notna(v):
                used_v.append(v); used_w.append(w)
        if not used_v:
            composite_vals.append(0.0)
        else:
            wn = np.array(used_w) / sum(used_w)
            composite_vals.append(float(np.dot(np.array(used_v), wn)))
    df["raw_score"] = composite_vals

    df["confidence"] = df["data_days"].apply(confidence_factor)
    df["score"] = (df["raw_score"] * df["confidence"]).round(2)

    # Funds with too-little history get score=0 explicitly
    short_mask = df["data_weeks"] < MIN_WEEKS_FOR_ANALYSIS
    df.loc[short_mask, "score"] = 0.0
    return df


# ===================================================================
# In-Sample Diagnostic Backtest (per-pillar IC vs forward SIP XIRR)
# ===================================================================

def diagnostic_backtest(
    aligned_navs: Dict[str, pd.Series],
    bench_nav: pd.Series,
    eval_step_weeks: int = LB_6M,
    n_evals: int = 8,
) -> pd.DataFrame:
    """
    At each historical eval date, re-build a *lightweight* version of
    each pillar from trailing data only and measure Spearman rank-IC
    against the realised forward 1Y SIP XIRR.

    Lightweight pillar proxies (so backtest is feasible in reasonable time):
      P1 - trailing 1Y alpha (we already have alpha; using regime-conditional
           is similar shape since regimes shift slowly)
      P2 - mean of last 4 block alphas (lightweight version of shrunk alpha)
      P3 - median rolling 1Y SIP XIRR over trailing data
      P4 - -CDaR5 (negated so higher = better)
      P5 - skipped in backtest (holdings rarely available historically)
    """
    common_idx = bench_nav.index
    n = len(common_idx)
    if n < LB_3Y + LB_1Y + 4:
        return pd.DataFrame()

    # Pick eval indices: most-recent eval is at n - LB_1Y - 1
    last_eval = n - LB_1Y - 1
    first_eval = max(LB_2Y, last_eval - eval_step_weeks * (n_evals - 1))
    eval_indices = list(range(first_eval, last_eval + 1, eval_step_weeks))
    if not eval_indices:
        return pd.DataFrame()

    # Pre-compute weekly returns
    bench_w = weekly_returns(bench_nav)

    records: List[Dict] = []
    for ep in eval_indices[-n_evals:]:
        ep_ts = common_idx[ep]
        ep_fwd_ts = common_idx[min(ep + LB_1Y, n - 1)]

        # Forward 1Y SIP XIRR per fund (the target)
        fwd_xirrs: Dict[str, float] = {}
        # Pillar proxies per fund
        p1: Dict[str, float] = {}
        p2: Dict[str, float] = {}
        p3: Dict[str, float] = {}
        p4: Dict[str, float] = {}

        # Trailing benchmark NAV (for alpha)
        bnav_train = bench_nav.loc[bench_nav.index <= ep_ts]
        if len(bnav_train) < LB_2Y:
            continue

        for fid, fnav in aligned_navs.items():
            train = fnav.loc[fnav.index <= ep_ts].dropna()
            if len(train) < LB_1Y + 4:
                continue

            # Forward SIP XIRR (target)
            fx = sip_xirr_for_window(fnav, ep_ts, ep_fwd_ts)
            if fx is None:
                continue
            fwd_xirrs[fid] = fx

            # Train returns
            train_ret = weekly_returns(train.iloc[-LB_1Y - 1:])
            bench_train_ret = bench_w.reindex(train_ret.index).dropna()
            train_ret = train_ret.reindex(bench_train_ret.index).dropna()
            if len(train_ret) < 30:
                continue

            # P1 proxy: trailing 1Y alpha
            a, _ = alpha_beta(train_ret, bench_train_ret)
            if a is not None:
                p1[fid] = a

            # P2 proxy: mean of trailing block alphas
            blocks = block_alphas(
                weekly_returns(train),
                bench_w.reindex(weekly_returns(train).index).dropna(),
                block_weeks=LB_6M,
                max_blocks=4,
            )
            if len(blocks) > 0:
                p2[fid] = float(np.mean(blocks))

            # P3 proxy: median of rolling 1Y SIP XIRRs over trailing data
            sip_back = rolling_sip_xirrs(train, window_years=1, step_weeks=8, bench_nav=None)
            if sip_back.get("sip_p50") is not None:
                p3[fid] = sip_back["sip_p50"]

            # P4 proxy: -CDaR5 (less-negative = better -> we want positive correlation)
            cd = cdar_5pct(train)
            if cd is not None:
                p4[fid] = -cd  # invert sign to align "higher = better"

        if len(fwd_xirrs) < 8:
            continue

        common_p1 = sorted(set(fwd_xirrs) & set(p1))
        common_p2 = sorted(set(fwd_xirrs) & set(p2))
        common_p3 = sorted(set(fwd_xirrs) & set(p3))
        common_p4 = sorted(set(fwd_xirrs) & set(p4))

        def safe_ic(common_keys: List[str], pillar: Dict[str, float]) -> Optional[float]:
            if len(common_keys) < 6:
                return None
            x = pd.Series([pillar[k] for k in common_keys])
            y = pd.Series([fwd_xirrs[k] for k in common_keys])
            if x.std() < 1e-9 or y.std() < 1e-9:
                return None
            ic = float(x.corr(y, method="spearman"))
            return ic if not np.isnan(ic) else None

        records.append({
            "eval_date":     ep_ts.date(),
            "n_funds":       len(fwd_xirrs),
            "ic_p1":         safe_ic(common_p1, p1),
            "ic_p2":         safe_ic(common_p2, p2),
            "ic_p3":         safe_ic(common_p3, p3),
            "ic_p4":         safe_ic(common_p4, p4),
            "fwd_xirr_med":  float(np.median(list(fwd_xirrs.values()))),
        })
    return pd.DataFrame(records)


# ===================================================================
# Output Formatters
# ===================================================================

def _pct(v) -> str:
    return f"{float(v) * 100:.2f}" if v is not None and pd.notna(v) else ""


def _ratio(v) -> str:
    return f"{float(v):.3f}" if v is not None and pd.notna(v) else ""


def _num(v) -> str:
    return f"{float(v):.2f}" if v is not None and pd.notna(v) else ""


def _int(v) -> str:
    return f"{int(v)}" if v is not None and pd.notna(v) else ""


# ===================================================================
# Main entry point
# ===================================================================

def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 80)
    print("  MID CAP MUTUAL FUND SCORING - CLAUDE (Regime-Conditional Predictive)")
    print(f"  Benchmark : Nifty Midcap 150  ({BENCHMARK})")
    print("  Pillars   : P1 RegimeAlpha 35% | P2 SkillPosterior 20%")
    print("              P3 SIPOutcome  20% | P4 TailRecovery   15% | P5 ActiveLiquidity 10%")
    print("=" * 80)

    provider = MfDataProvider(date=date)

    # ---- Benchmark ----
    logger.info("Loading benchmark index data...")
    bench_df = provider.get_index_chart(BENCHMARK)
    bench_nav = clean_nav_to_series(bench_df)
    if len(bench_nav) < LB_2Y:
        logger.error(
            "Benchmark history too short (%d weeks). Aborting.", len(bench_nav)
        )
        sys.exit(1)
    print(
        f"\n  Benchmark data : {len(bench_nav)} weeks  "
        f"({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})"
    )

    # ---- Regime backbone ----
    reg = regime_series(bench_nav)
    cur_probs = soft_current_regime_probs(reg, recent_weeks=8)
    transition = empirical_forward_mix(reg, horizon_weeks=LB_1Y)
    forward_mix = cur_probs @ transition
    print("\n  Current regime mix (last 8w):  " + ", ".join(
        f"{REGIMES[i]} {cur_probs[i] * 100:4.1f}%" for i in range(N_REGIMES)
    ))
    print("  Implied forward 1Y mix:        " + ", ".join(
        f"{REGIMES[i]} {forward_mix[i] * 100:4.1f}%" for i in range(N_REGIMES)
    ))

    # ---- Fund universe ----
    df_all = provider.list_all_mf()
    universe = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Mid Cap funds  : {len(universe)}")

    # ---- Per-fund analysis ----
    raw_results: List[Dict] = []
    aligned_navs: Dict[str, pd.Series] = {}
    holdings_count = 0
    for _, row in universe.iterrows():
        mf_id = str(row["mfId"])
        name = str(row["name"])
        aum = float(row.get("aum", 0) or 0)
        try:
            chart = provider.get_mf_chart(mf_id)
            fund_nav = clean_nav_to_series(chart)
            if len(fund_nav) < 4:
                logger.warning("Skip %s (%s): only %d NAV points", mf_id, name, len(fund_nav))
                continue
            try:
                holdings = provider.read_mf_holdings(mf_id)
            except Exception:
                holdings = []
            if holdings:
                holdings_count += 1

            res = analyse_fund(mf_id, fund_nav, bench_nav, reg, name, aum, holdings)
            raw_results.append(res)

            # store aligned NAV for diagnostic backtest
            aligned_navs[mf_id] = fund_nav.reindex(bench_nav.index).ffill()
        except Exception as e:
            logger.error("Error analysing %s (%s): %s", mf_id, name, e)

    if not raw_results:
        logger.error("No funds analysed. Exiting.")
        sys.exit(1)

    print(f"  Funds analysed : {len(raw_results)}  (with holdings: {holdings_count})")

    df_raw = pd.DataFrame(raw_results)

    # ---- Cross-sectional inputs to the pillar score functions ----
    # 1) Peer-mean per-regime alpha (for filling in regimes a fund hasn't seen)
    per_regime_dict: Dict[str, np.ndarray] = {
        r["mfId"]: np.asarray(r.get("_per_regime_alpha", [np.nan] * N_REGIMES), dtype=float)
        for r in raw_results
        if r.get("_per_regime_alpha") is not None
    }
    peer_per_regime = peer_mean_per_regime(per_regime_dict)
    print("\n  Peer mean per-regime alpha (annualised):")
    for i in range(N_REGIMES):
        print(f"    {REGIMES[i]:10s}: {peer_per_regime[i] * 100:+6.2f}%")

    # 2) Bayesian shrunk alpha
    block_dict: Dict[str, np.ndarray] = {
        r["mfId"]: np.asarray(r.get("_block_alphas", []), dtype=float)
        for r in raw_results
        if r.get("_block_alphas") is not None
    }
    shrunk_alpha, peer_mean_alpha, shrink_lambda = james_stein_shrunk_alphas(block_dict)
    print(f"\n  Bayesian peer-mean block alpha: {peer_mean_alpha * 100:+6.2f}%  "
          f"(median shrink lambda: {np.median(list(shrink_lambda.values())) if shrink_lambda else 0:.2f})")

    # ---- Composite scoring ----
    df_scored = compute_pillar_scores(
        df_raw,
        peer_per_regime=peer_per_regime,
        current_regime_probs=cur_probs,
        transition_matrix=transition,
        shrunk_alpha=shrunk_alpha,
        shrunk_lambda=shrink_lambda,
    )
    p5_active = bool(df_scored.attrs.get("_p5_active", False))
    print(f"  P5 (Active/Liquidity) active : {p5_active}")

    df_scored["rank"] = df_scored["score"].rank(ascending=False, method="min").astype(int)
    df_scored = df_scored.sort_values("rank")

    # ---- Diagnostic backtest ----
    logger.info("Running diagnostic backtest (per-pillar IC)...")
    bt = diagnostic_backtest(aligned_navs, bench_nav, eval_step_weeks=LB_6M, n_evals=8)
    persist_ic = persistence_diagnostic_ic(aligned_navs, bench_nav, eval_step=LB_3M, n_evals=8)

    # ---- Build output CSV ----
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
    output["sortino"] = df_scored["sortino"].apply(_ratio)
    output["alpha"] = df_scored["alpha"].apply(_pct)
    output["beta"] = df_scored["beta"].apply(_ratio)

    # Pillar diagnostic columns
    output["p1_forward_alpha"] = df_scored["p1_forward_alpha"].apply(_pct)
    output["p1_regime_capture_spread"] = df_scored["p1_regime_capture_spread"].apply(_ratio)
    output["p1_score"] = df_scored["p1_score"].apply(_num)
    output["p2_shrunk_alpha"] = df_scored["p2_shrunk_alpha"].apply(_pct)
    output["p2_shrink_lambda"] = df_scored["p2_shrink_lambda"].apply(_ratio)
    output["p2_personal_stability"] = df_scored["p2_personal_stability"].apply(_ratio)
    output["p2_block_alpha_n"] = df_scored["p2_block_alpha_n"].apply(_int)
    output["p2_score"] = df_scored["p2_score"].apply(_num)
    output["p3_sip_xirr_p50"] = df_scored["p3_sip_xirr_p50"].apply(_pct)
    output["p3_sip_xirr_p25"] = df_scored["p3_sip_xirr_p25"].apply(_pct)
    output["p3_sip_hit_vs_bench"] = df_scored["p3_sip_hit_vs_bench"].apply(_pct)
    output["p3_sip_n_windows"] = df_scored["p3_sip_n_windows"].apply(_int)
    output["p3_score"] = df_scored["p3_score"].apply(_num)
    output["p4_cdar5"] = df_scored["p4_cdar5"].apply(_pct)
    output["p4_recovery_half_w"] = df_scored["p4_recovery_half_w"].apply(_num)
    output["p4_dn_up_vol"] = df_scored["p4_dn_up_vol"].apply(_ratio)
    output["p4_skew"] = df_scored["p4_skew"].apply(_ratio)
    output["p4_score"] = df_scored["p4_score"].apply(_num)
    output["p5_top10_conc"] = df_scored["p5_top10_conc"].apply(_pct)
    output["p5_n_holdings"] = df_scored["p5_n_holdings"].apply(_int)
    output["p5_avg_change3m"] = df_scored["p5_avg_change3m"].apply(_ratio)
    output["p5_aum_haircut"] = df_scored["p5_aum_haircut"].apply(_ratio)
    output["p5_score"] = df_scored["p5_score"].apply(_num)
    output["aum"] = df_scored["aum"]
    output["data_weeks"] = df_scored["data_weeks"]
    output["confidence"] = df_scored["confidence"].apply(_ratio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    logger.info("Results saved to %s", OUTPUT_FILE)

    # ---- Console summary ----
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC BACKTEST (per-pillar Spearman IC vs forward 1Y SIP XIRR)")
    print("=" * 80)
    if bt.empty:
        print("  Insufficient history for diagnostic backtest.")
    else:
        with pd.option_context("display.float_format", "{:.3f}".format):
            print("\n" + bt.to_string(index=False))
        print("\n  Mean IC by pillar:")
        for col in ("ic_p1", "ic_p2", "ic_p3", "ic_p4"):
            v = bt[col].dropna()
            if len(v) > 0:
                print(f"    {col:8s}: {v.mean():+.3f}  (n={len(v)})")
            else:
                print(f"    {col:8s}: n/a")
    if persist_ic is not None:
        print(f"\n  Cross-sectional persistence IC (past 1Y -> fwd 1Y): {persist_ic:+.3f}")
    else:
        print("\n  Cross-sectional persistence IC: n/a (insufficient data)")

    print("\n" + "=" * 80)
    print(f"  TOP 15 MID CAP FUNDS - {SECTOR}_Claude")
    print("=" * 80 + "\n")
    show_cols = [
        "rank", "name", "score",
        "cagr_3y", "cagr_5y",
        "p1_score", "p2_score", "p3_score", "p4_score",
    ]
    show_cols = [c for c in show_cols if c in output.columns]
    print(output.head(15)[show_cols].to_string(index=False))

    print(f"\n  Full results ({len(output)} funds) -> {OUTPUT_FILE}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mid Cap MF screener (Claude - regime-conditional)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
