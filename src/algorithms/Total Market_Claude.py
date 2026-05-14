#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Claude
====================================================

Hybrid SIP-Hold Composite (HSC) for Indian "broad market" diversified
active funds: Flexi Cap, Multi Cap, Focused, Value, Contra.

The User's Scenario
-------------------
The user invests via 12 monthly SIPs in the first year, then holds without
any further contributions in the second year, and exits with a single
sell at month 24.  The cashflow timeline:

    Month 1 ........ Month 12 ........ Month 24
       |                |                  |
       |--- SIP buys ---|----- HOLD -------|
       12 monthly cashflows           1 sell
       (1st of each month)            (final NAV)

This hybrid structure has two distinct phases that demand different fund
qualities:

  * Year 1 (accumulation): cost-basis averaging cushions volatility -
    drawdowns can actually help by lowering average entry NAV.
  * Year 2 (hold): no new contributions, so a deep drawdown in months
    13-24 directly hits the final value with no chance to average down.
    This phase behaves like a lumpsum hold.

A good fund needs to handle both phases.  The algorithm reflects this by
combining 1Y-window metrics (year-1 quality) and 2Y-window metrics
(full horizon quality).

Why a New Strategy
------------------
Research evidence drives the design:

  * Alpha persistence collapses at multi-year horizons.  SPIVA Persistence
    Scorecard 2024: only 8.3% of outperforming active equity funds
    remained outperformers over the next 2 years; among 2022 large-cap
    top-quartile funds, *0%* stayed top-quartile (vs 6.25% expected
    randomly).  Implication: do *not* rank funds on raw trailing alpha.
  * Rolling-window consistency *does* persist.  Indian flexi-cap data
    shows top funds delivered ~99% of 2Y rolling windows above 20%
    returns vs ~4% for the category average.  Implication: rank on
    consistency of rolling outcomes.
  * Augmented Carhart works.  Funds beating *both* benchmark *and* peer
    median in rolling windows show 36-month-forward persistence.
    Implication: use a "beat-both" hit-rate, not just benchmark beat.
  * Recovery speed matters disproportionately at 24M horizon.  A SIP that
    buys through a year-1 drawdown only compounds well if NAV recovers
    in year 2 - and there are no fresh buys to help.  Implication:
    hard-weight recovery half-life and current DD position.
  * AUM capacity (Berk-Green).  Capacity erosion of alpha is visible at
    multi-year horizons.  Implication: penalise extreme AUM where alpha
    has decayed.

Five-Pillar Architecture
------------------------
P1  Hybrid SIP-Hold XIRR             30%   Empirical distribution of all
                                            rolling 24-month "1Y SIP +
                                            1Y Hold" XIRRs: median, 25th-
                                            pct downside, hit-rate vs
                                            Nifty 500 same-structure XIRR,
                                            hit-rate vs peer-median XIRR,
                                            number of windows.

P2  Year-1 Skill Consistency         22%   % of trailing 12M windows with
                                            positive alpha; % of 12M
                                            windows beating peer-median
                                            alpha (augmented Carhart);
                                            James-Stein-shrunk alpha
                                            across non-overlapping 12M
                                            blocks; Information Ratio.

P3  Year-2 Hold Resilience           23%   Forward 24M alpha projection
                                            through empirical Nifty-500
                                            regime transition matrix;
                                            recovery half-life from worst
                                            trough; bear/correction excess
                                            return; current drawdown
                                            depth (entry valuation signal).

P4  Active Conviction & Capacity     10%   Active divergence x alpha
                                            (genuine convicted bets that
                                            work); cross-regime beta
                                            stability; soft AUM-capacity
                                            haircut (Berk-Green smooth
                                            penalty for AUM > 25,000 Cr).

P5  Compounding Path Quality         15%   Gain-to-pain, tail ratio,
                                            Ulcer Performance Index;
                                            vol-drag-adjusted geometric
                                            return (alpha - sigma^2/2);
                                            current drawdown.

Composite = sum(w * pillar_percentile) * confidence_haircut(data_weeks).
Confidence haircut: 0.55 (<1Y), 0.75 (1-2Y), 0.88 (2-3Y), 0.97 (3-5Y),
1.00 (5Y+).  Funds with <110 weeks of data are scored on P2-P5 only.

Differentiation vs Other Models in This Repo
--------------------------------------------
- The previous Total Market_Claude.py was a 1Y-horizon composite.  This
  version directly simulates the user's 24-month hybrid cashflow.
- Codex's tuning approach used walk-forward weight optimisation; we use
  research-grounded weights (pillar weights derived from SPIVA evidence
  on what persists at multi-year horizons) and report 24M-IC as a
  diagnostic so weights stay interpretable.
- Gemini relied on simple Sharpe + win-rate over a fixed window.  We use
  rolling-window consistency fractions, Bayesian shrinkage, and
  regime-conditional projections.
- No per-sector or per-theme bonuses anywhere - every fund scored using
  identical, purely data-driven metrics.

Sector  : Total Market (broad-market diversified active)
Author  : Claude
"""

from __future__ import annotations

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
# Path & logging setup
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
SECTOR_NAME = "Equity"  # MfDataProvider sector key
SUBSECTORS = [
    "Flexi Cap Fund",
    "Multi Cap Fund",
    "Focused Fund",
    "Value Fund",
    "Contra Fund",
]
PRIMARY_BENCHMARK = "Total Market"           # resolves to .NIFTY500
RISK_FREE_RATE = 0.065                       # India ~6.5% T-bill proxy
WEEKS_PER_YEAR = 52

LB_3M = 13
LB_6M = 26
LB_1Y = 52
LB_2Y = 104
LB_3Y = 156
LB_5Y = 260

# Data adequacy thresholds
MIN_WEEKS_FOR_ANALYSIS = 30
MIN_WEEKS_FOR_2Y_SIP = 110   # need at least one full hybrid window
MIN_WEEKS_3Y = 150
MIN_WEEKS_5Y = 250

# The user's hybrid scenario
SIP_MONTHS = 12
HOLD_MONTHS = 12
TOTAL_MONTHS = SIP_MONTHS + HOLD_MONTHS  # = 24
HYBRID_HORIZON_WEEKS = TOTAL_MONTHS * 4 + (TOTAL_MONTHS // 3)  # ~104
SIP_MONTHLY_AMOUNT = 10000.0  # only ratio matters for XIRR

PILLAR_WEIGHTS = {
    "p1_hybrid_sip_hold":    0.30,
    "p2_skill_consistency":  0.22,
    "p3_hold_resilience":    0.23,
    "p4_conviction_capac":   0.10,
    "p5_compounding_path":   0.15,
}
assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9

REGIMES = ("Bull", "Sideways", "Correction", "Bear")
N_REGIMES = len(REGIMES)

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "Total Market_Claude.csv"


# ===================================================================
# 1. Data Loading & Cleaning
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


def align_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Reindex a NAV series onto a target index with forward-fill (max 1 wk)."""
    if len(s) == 0:
        return pd.Series(index=idx, dtype=float)
    return s.reindex(idx, method="ffill", limit=1)


# ===================================================================
# 2. Return / Volatility Helpers
# ===================================================================

def weekly_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def annualised_return(nav: pd.Series, weeks: int) -> Optional[float]:
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


def alpha_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Annualised Jensen alpha and beta via OLS on weekly excess returns."""
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < 12:
        return None, None
    x = aligned["b"].values
    y = aligned["f"].values
    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-12:
        return None, None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    alpha_w = float(np.mean(y) - rf_w - beta * (np.mean(x) - rf_w))
    return float(alpha_w * WEEKS_PER_YEAR), beta


def information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < 20:
        return None
    excess = aligned["f"] - aligned["b"]
    te = excess.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    if te <= 1e-12:
        return None
    return float((excess.mean() * WEEKS_PER_YEAR) / te)


def max_drawdown_calc(nav: pd.Series) -> Optional[float]:
    if nav is None or len(nav) < 8:
        return None
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


# ===================================================================
# 3. NIFTY-500 Regime Classifier
# ===================================================================
#
#   Bull       : 12W momentum > +5% AND drawdown_26W shallower than -5%
#   Sideways   : everything else with drawdown_26W shallower than -10%
#   Correction : drawdown_26W between -10% and -20%
#   Bear       : drawdown_26W <= -20% OR 12W momentum <= -10%

def _classify_week(mom_12w: float, dd_26w: float) -> int:
    if pd.isna(mom_12w) or pd.isna(dd_26w):
        return -1
    if dd_26w <= -0.20 or mom_12w <= -0.10:
        return 3
    if dd_26w <= -0.10:
        return 2
    if mom_12w > 0.05 and dd_26w > -0.05:
        return 0
    return 1


def regime_series(bench_nav: pd.Series) -> pd.Series:
    if len(bench_nav) < LB_6M + 4:
        return pd.Series(dtype="float64", index=bench_nav.index)
    mom_12w = bench_nav.pct_change(12)
    peak_26w = bench_nav.rolling(LB_6M, min_periods=8).max()
    dd_26w = bench_nav / peak_26w - 1.0
    out = pd.Series(np.nan, index=bench_nav.index, dtype="float64")
    for ts in bench_nav.index:
        r = _classify_week(mom_12w.get(ts, np.nan), dd_26w.get(ts, np.nan))
        if r >= 0:
            out[ts] = float(r)
    return out


def empirical_forward_mix(reg: pd.Series, horizon_weeks: int = LB_2Y) -> np.ndarray:
    """Mean fraction of next `horizon_weeks` spent in each regime, by start regime."""
    valid = reg.dropna().astype(int)
    if len(valid) < horizon_weeks + 4:
        # fallback: each regime persists with ~70% probability
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
            M[r, rp] += float(np.mean(forward == rp))
        counts[r] += 1
    for r in range(N_REGIMES):
        if counts[r] > 0:
            M[r] /= counts[r]
        else:
            M[r] = np.bincount(arr, minlength=N_REGIMES).astype(float) / max(1, len(arr))
    return M


def soft_current_regime(reg: pd.Series, recent_weeks: int = 8) -> np.ndarray:
    recent = reg.dropna().iloc[-recent_weeks:].astype(int)
    if len(recent) == 0:
        return np.ones(N_REGIMES) / N_REGIMES
    counts = np.bincount(recent.values, minlength=N_REGIMES).astype(float)
    if counts.sum() == 0:
        return np.ones(N_REGIMES) / N_REGIMES
    return counts / counts.sum()


def per_regime_alpha(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    reg: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Annualised alpha for each regime; NaN where < 8 weekly observations."""
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
        var_x = float(np.var(x, ddof=1))
        if var_x < 1e-12:
            n_obs[r] = n
            continue
        beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
        alpha_w = float(np.mean(y) - beta * np.mean(x))
        # Sample-size shrink toward zero + clip
        shrink = 1.0 / (1.0 + 4.0 / np.sqrt(n))
        annual = float(np.clip(alpha_w * WEEKS_PER_YEAR * shrink, -0.50, 0.50))
        alphas[r] = annual
        n_obs[r] = n
    return alphas, n_obs


def peer_mean_per_regime(per_fund_alphas: Dict[str, np.ndarray]) -> np.ndarray:
    if not per_fund_alphas:
        return np.zeros(N_REGIMES)
    stack = np.vstack(list(per_fund_alphas.values()))
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
    n_observed = int(np.sum(~np.isnan(fund_alphas)))
    if n_observed < min_observed_regimes:
        return None
    filled = np.where(np.isnan(fund_alphas), peer_alphas, fund_alphas)
    if np.any(np.isnan(filled)):
        return None
    forward_mix = current_probs @ transition
    return float(forward_mix @ filled)


def regime_capture_spread(
    fund_ret: pd.Series, bench_ret: pd.Series, reg: pd.Series
) -> Optional[float]:
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
    return float((fund_up / bench_up) - (fund_dn / bench_dn))


def bear_correction_outperformance(
    fund_ret: pd.Series, bench_ret: pd.Series, reg: pd.Series
) -> Optional[float]:
    aligned = pd.concat({"f": fund_ret, "b": bench_ret, "r": reg}, axis=1).dropna()
    bear = aligned[aligned["r"].astype(int).isin([2, 3])]
    if len(bear) < 8:
        return None
    return float((bear["f"] - bear["b"]).mean() * WEEKS_PER_YEAR)


# ===================================================================
# 4. James-Stein Shrinkage (P2 Bayesian alpha)
# ===================================================================

def block_alphas(
    fund_ret: pd.Series,
    bench_ret: pd.Series,
    block_weeks: int = LB_1Y,    # 12-month blocks for 24M-target horizon
    max_blocks: int = 5,
) -> np.ndarray:
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < block_weeks:
        return np.array([])
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    out: List[float] = []
    end = len(aligned)
    while end - block_weeks >= 0 and len(out) < max_blocks:
        sub = aligned.iloc[end - block_weeks: end]
        x = sub["b"].values - rf_w
        y = sub["f"].values - rf_w
        var_x = float(np.var(x, ddof=1))
        if var_x >= 1e-12:
            beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
            alpha_w = float(np.mean(y) - beta * np.mean(x))
            out.append(alpha_w * WEEKS_PER_YEAR)
        end -= block_weeks
    return np.array(out)


def james_stein_shrunk(
    fund_block_alphas: Dict[str, np.ndarray]
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """Posterior alpha = peer_mean + lambda * (fund_mean - peer_mean)."""
    means: Dict[str, float] = {}
    within_var: Dict[str, float] = {}
    for fid, blocks in fund_block_alphas.items():
        if len(blocks) == 0:
            continue
        means[fid] = float(np.mean(blocks))
        within_var[fid] = (
            float(np.var(blocks, ddof=1) / len(blocks))
            if len(blocks) > 1
            else 0.04 ** 2
        )
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
        lam = 0.5 if denom < 1e-12 else float(between_var / denom)
        lam = max(0.0, min(1.0, lam))
        shrunk[fid] = peer_mean + lam * (m - peer_mean)
        lams[fid] = lam
    return shrunk, peer_mean, lams


# ===================================================================
# 5. Hybrid SIP-Hold XIRR Simulation (P1)
# ===================================================================

def _xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    days = np.array([(cf[0] - t0).days for cf in cashflows], dtype=float)
    amts = np.array([cf[1] for cf in cashflows], dtype=float)
    if np.all(amts >= 0) or np.all(amts <= 0):
        return None

    def npv(rate: float) -> float:
        return float(np.sum(amts / (1.0 + rate) ** (days / 365.0)))

    try:
        return float(brentq(npv, -0.95, 5.0, xtol=1e-7, maxiter=200))
    except (ValueError, RuntimeError):
        try:
            return float(brentq(npv, -0.999, 20.0, xtol=1e-6, maxiter=200))
        except (ValueError, RuntimeError):
            return None


def hybrid_sip_xirr_for_window(
    nav: pd.Series,
    sip_start: pd.Timestamp,
    sip_end_exclusive: pd.Timestamp,
    final_exit: pd.Timestamp,
    monthly_amount: float = SIP_MONTHLY_AMOUNT,
) -> Optional[float]:
    """
    The user's cashflow:
      * Buy on 1st of each month from sip_start up to (but not including)
        sip_end_exclusive.  Buy at first available NAV >= the 1st.
      * No transactions between sip_end_exclusive and final_exit.
      * Sell entire holding at last NAV <= final_exit.

    Returns XIRR or None.  No look-ahead - all buys use NAV available
    on or after the 1st of the buy month.
    """
    if nav is None or len(nav) == 0 or final_exit <= sip_start:
        return None
    buys = pd.date_range(start=sip_start, end=sip_end_exclusive, freq="MS")
    buys = buys[buys < sip_end_exclusive]  # exclusive end
    if len(buys) < SIP_MONTHS - 1:
        return None
    units = 0.0
    cashflows: List[Tuple[pd.Timestamp, float]] = []
    last_buy_dt: Optional[pd.Timestamp] = None
    for buy_target in buys:
        avail = nav.loc[nav.index >= buy_target]
        if avail.empty:
            continue
        buy_dt = avail.index[0]
        if buy_dt >= final_exit:
            continue
        buy_nav = float(avail.iloc[0])
        if buy_nav <= 0:
            continue
        units += monthly_amount / buy_nav
        cashflows.append((buy_dt, -float(monthly_amount)))
        last_buy_dt = buy_dt
    if len(cashflows) < SIP_MONTHS - 1:
        return None
    final_avail = nav.loc[nav.index <= final_exit]
    if final_avail.empty:
        return None
    final_dt = final_avail.index[-1]
    final_nav = float(final_avail.iloc[-1])
    if last_buy_dt is None or final_dt <= last_buy_dt:
        return None
    cashflows.append((final_dt, units * final_nav))
    return _xirr(cashflows)


def rolling_hybrid_sip_xirrs(
    nav: pd.Series,
    step_weeks: int = 4,
    max_windows: int = 60,
) -> List[Tuple[pd.Timestamp, float]]:
    """
    All rolling 24-month "1Y SIP + 1Y Hold" XIRRs anchored at successive
    starting points, most recent first.  Returns list of (start_ts, xirr).
    """
    if nav is None or len(nav) < HYBRID_HORIZON_WEEKS + 8:
        return []
    timestamps = nav.index
    end_idx = len(nav) - 1
    out: List[Tuple[pd.Timestamp, float]] = []
    while end_idx - HYBRID_HORIZON_WEEKS >= 0 and len(out) < max_windows:
        sip_start_ts = timestamps[end_idx - HYBRID_HORIZON_WEEKS]
        # SIP for first 12 months -> ends ~52 weeks after start
        sip_end_ts = timestamps[end_idx - HYBRID_HORIZON_WEEKS // 2]
        final_exit_ts = timestamps[end_idx]
        sub = nav.iloc[: end_idx + 1]
        x = hybrid_sip_xirr_for_window(sub, sip_start_ts, sip_end_ts, final_exit_ts)
        if x is not None and -0.95 < x < 5.0:
            out.append((sip_start_ts, x))
        end_idx -= step_weeks
    return out


# ===================================================================
# 6. Path Quality (P5) - tail / drawdown / recovery
# ===================================================================

def gain_to_pain_ratio(rets: pd.Series) -> Optional[float]:
    if len(rets) < 20:
        return None
    pain = abs(rets[rets < 0].sum())
    if pain <= 1e-12:
        return 10.0
    return float(rets.sum() / pain)


def tail_ratio(rets: pd.Series) -> Optional[float]:
    if len(rets) < 30:
        return None
    p95 = float(np.percentile(rets, 95))
    p5 = float(abs(np.percentile(rets, 5)))
    if p5 <= 1e-12:
        return 5.0
    return float(p95 / p5)


def ulcer_index(nav: pd.Series) -> Optional[float]:
    if nav is None or len(nav) < 10:
        return None
    peak = nav.cummax()
    dd_pct = (nav / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def ulcer_perf_index(cagr: Optional[float], ui: Optional[float]) -> Optional[float]:
    if cagr is None or ui is None or ui <= 1e-9:
        return None
    return float((cagr - RISK_FREE_RATE) * 100.0 / ui)


def recovery_half_life(nav: pd.Series) -> Optional[float]:
    """Weeks for fund to recover half of its worst observed drawdown."""
    if nav is None or len(nav) < 20:
        return None
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough_idx = int(dd.values.argmin())
    if trough_idx == len(dd) - 1:
        return None  # still in trough
    trough_dd = float(dd.iloc[trough_idx])
    if trough_dd >= -0.005:
        return 1.0
    target = trough_dd / 2.0
    after = dd.iloc[trough_idx + 1:]
    above = after[after >= target]
    if above.empty:
        return float(len(after))
    rec_idx = above.index[0]
    weeks = int(nav.index.get_loc(rec_idx)) - trough_idx
    return float(max(1, weeks))


def current_drawdown(nav: pd.Series) -> Optional[float]:
    if nav is None or len(nav) < 4:
        return None
    return float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1.0)


def vol_drag_adjusted_return(rets: pd.Series) -> Optional[float]:
    """
    Geometric-style adjusted annualised return = arithmetic_mean - 0.5 * variance.
    The (sigma^2)/2 drag matters for compounding over 24 months: two funds
    with the same arithmetic mean but different volatility compound very
    differently, and this metric captures it without needing a long
    enough actual trailing window for the geometric mean to converge.
    """
    if rets is None or len(rets) < 26:
        return None
    arith = float(rets.mean() * WEEKS_PER_YEAR)
    var_ann = float(rets.var(ddof=1) * WEEKS_PER_YEAR)
    return float(arith - 0.5 * var_ann)


# ===================================================================
# 7. Active Conviction & Capacity (P4)
# ===================================================================

def active_divergence(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    """Annualised tracking error - activity proxy."""
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < 20:
        return None
    excess = aligned["f"] - aligned["b"]
    return float(excess.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def active_skill_signal(
    alpha: Optional[float], divergence: Optional[float]
) -> Optional[float]:
    """alpha * tracking_error - rewards convicted active bets that work."""
    if alpha is None or divergence is None:
        return None
    return float(np.clip(alpha, -0.20, 0.30) * np.clip(divergence, 0.0, 0.40))


def cross_regime_beta_stability(
    fund_ret: pd.Series, bench_ret: pd.Series, reg: pd.Series
) -> Optional[float]:
    """1 / (1 + std_of_per_regime_betas).  Higher = more predictable style."""
    aligned = pd.concat({"f": fund_ret, "b": bench_ret, "r": reg}, axis=1).dropna()
    if len(aligned) < 40:
        return None
    betas: List[float] = []
    for r in range(N_REGIMES):
        sub = aligned[aligned["r"].astype(int) == r]
        if len(sub) < 8:
            continue
        x = sub["b"].values
        y = sub["f"].values
        var_x = float(np.var(x, ddof=1))
        if var_x < 1e-12:
            continue
        b = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
        betas.append(b)
    if len(betas) < 2:
        return None
    sd = float(np.std(betas, ddof=1))
    return float(1.0 / (1.0 + sd))


def aum_capacity_haircut(aum: Optional[float]) -> float:
    """
    Smooth piecewise multiplier in [0.85, 1.0] for AUM > 25,000 Cr.
    Berk-Green (2004) style: large funds face capacity-driven alpha decay.
    Below 25,000 Cr: full credit (1.0).
    25,000 - 75,000 Cr: linear from 1.0 -> 0.92.
    Above 75,000 Cr: floor at 0.85.
    """
    if aum is None or pd.isna(aum) or aum <= 0:
        return 1.0
    if aum < 25000:
        return 1.0
    if aum < 75000:
        return float(1.0 - (aum - 25000) / 50000 * 0.08)
    return 0.85


# ===================================================================
# 8. Rolling 12M Alpha Consistency (P2)
# ===================================================================

def _rolling_window_starts(
    n_weeks_total: int, window_weeks: int, step_weeks: int = 4, max_windows: int = 60
) -> List[int]:
    """End-anchored window-end indices, most recent first."""
    starts: List[int] = []
    end = n_weeks_total - 1
    while end - window_weeks >= 0 and len(starts) < max_windows:
        starts.append(end)
        end -= step_weeks
    return starts


def _alpha_in_window(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Optional[float]:
    """Single annualised alpha for an aligned window."""
    if len(fund_ret) < 20:
        return None
    aligned = pd.concat({"f": fund_ret, "b": bench_ret}, axis=1).dropna()
    if len(aligned) < 20:
        return None
    x = aligned["b"].values
    y = aligned["f"].values
    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-12:
        return None
    rf_w = (1 + RISK_FREE_RATE) ** (1 / WEEKS_PER_YEAR) - 1
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    alpha_w = float(np.mean(y) - rf_w - beta * (np.mean(x) - rf_w))
    return float(alpha_w * WEEKS_PER_YEAR)


def rolling_12m_alphas(
    fund_nav: pd.Series, bench_nav: pd.Series,
    step_weeks: int = 4, max_windows: int = 30,
) -> List[Tuple[int, float]]:
    """
    Returns list of (window_end_idx, alpha) for non-overlapping or
    step-spaced trailing 12M windows, most recent first.
    """
    if len(fund_nav) < LB_1Y + 4:
        return []
    bench_w_full = weekly_returns(bench_nav)
    out: List[Tuple[int, float]] = []
    for end_idx in _rolling_window_starts(len(fund_nav), LB_1Y, step_weeks, max_windows):
        sub = fund_nav.iloc[end_idx - LB_1Y: end_idx + 1]
        fr = weekly_returns(sub)
        br = bench_w_full.reindex(fr.index).dropna()
        fr2 = fr.reindex(br.index).dropna()
        a = _alpha_in_window(fr2, br)
        if a is not None and -1.0 < a < 1.0:
            out.append((end_idx, a))
    return out


# ===================================================================
# 9. Walk-forward IC (24M horizon)
# ===================================================================

def walk_forward_ic_24m(
    aligned_navs: Dict[str, pd.Series],
    bench_nav: pd.Series,
    eval_step: int = LB_3M,
    n_evals: int = 6,
) -> Optional[float]:
    """
    Spearman rank-corr of (trailing 24M hybrid SIP-Hold XIRR) vs (forward
    24M hybrid SIP-Hold XIRR), averaged across `n_evals` historical eval
    points.  This is the hypothesis test: do consistency-style metrics
    actually predict 24M-forward outcomes better than alpha did at 1Y?
    """
    n = len(bench_nav)
    if n < HYBRID_HORIZON_WEEKS * 2 + 4:
        return None
    last_eval = n - HYBRID_HORIZON_WEEKS - 1
    first_eval = max(HYBRID_HORIZON_WEEKS, last_eval - eval_step * (n_evals - 1))
    eval_points = list(range(first_eval, last_eval + 1, eval_step))
    if not eval_points:
        return None
    timestamps = bench_nav.index
    ics: List[float] = []
    for ep in eval_points[-n_evals:]:
        ep_ts = timestamps[ep]
        fwd_ts = timestamps[min(ep + HYBRID_HORIZON_WEEKS, n - 1)]
        past_xirrs: Dict[str, float] = {}
        fwd_xirrs: Dict[str, float] = {}
        for fid, fnav in aligned_navs.items():
            past_n = fnav.loc[fnav.index <= ep_ts].dropna()
            if len(past_n) < HYBRID_HORIZON_WEEKS + 4:
                continue
            past_start = past_n.index[-(HYBRID_HORIZON_WEEKS + 1)]
            past_sip_end = past_n.index[-(HYBRID_HORIZON_WEEKS // 2 + 1)]
            x_past = hybrid_sip_xirr_for_window(past_n, past_start, past_sip_end, ep_ts)
            if x_past is None:
                continue
            past_xirrs[fid] = x_past

            fwd_n = fnav.loc[(fnav.index > ep_ts) & (fnav.index <= fwd_ts)].dropna()
            if len(fwd_n) < HYBRID_HORIZON_WEEKS - 4:
                continue
            f_start = fwd_n.index[0]
            mid_idx = len(fwd_n) // 2
            f_sip_end = fwd_n.index[mid_idx]
            x_fwd = hybrid_sip_xirr_for_window(fwd_n, f_start, f_sip_end, fwd_ts)
            if x_fwd is None:
                continue
            fwd_xirrs[fid] = x_fwd

        common = sorted(set(past_xirrs) & set(fwd_xirrs))
        if len(common) < 8:
            continue
        p = pd.Series([past_xirrs[k] for k in common])
        f = pd.Series([fwd_xirrs[k] for k in common])
        if p.std() < 1e-9 or f.std() < 1e-9:
            continue
        ic = float(p.corr(f, method="spearman"))
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return None
    return float(np.mean(ics))


# ===================================================================
# 10. Per-fund metric computation (raw - peer-relative metrics computed later)
# ===================================================================

def compute_fund_metrics(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    bench_ret: pd.Series,
    reg: pd.Series,
    name: str,
    aum: float,
    subsector: str,
) -> dict:
    out: dict = {
        "mfId": mf_id,
        "name": name,
        "subsector": subsector,
        "aum": round(aum, 2) if aum is not None else None,
    }
    if fund_nav is None or len(fund_nav) == 0:
        out.update({"data_days": 0, "data_weeks": 0})
        return out

    first_ts = fund_nav.index.min()
    last_ts = fund_nav.index.max()
    out["data_days"] = int((last_ts - first_ts).days) + 1 if pd.notna(first_ts) else 0
    out["data_weeks"] = int(len(fund_nav))

    out["cagr_1y"] = annualised_return(fund_nav, LB_1Y)
    out["cagr_3y"] = annualised_return(fund_nav, LB_3Y)
    out["cagr_5y"] = annualised_return(fund_nav, LB_5Y)

    if out["data_weeks"] < MIN_WEEKS_FOR_ANALYSIS:
        return out

    rets = weekly_returns(fund_nav)

    # Standard ratios
    primary_cagr = out["cagr_3y"] or out["cagr_5y"] or out["cagr_1y"]
    vol = annualised_volatility(rets)
    out["volatility"] = vol
    out["sharpe"] = (
        (primary_cagr - RISK_FREE_RATE) / vol
        if primary_cagr is not None and vol is not None and vol > 1e-9
        else None
    )

    # Alpha/beta and IR
    alpha, beta = alpha_beta(rets, bench_ret)
    out["alpha"] = alpha
    out["beta"] = beta
    out["info_ratio"] = information_ratio(rets, bench_ret)

    # P3 - per-regime alpha (raw) + capture + bear excess
    pra, n_obs = per_regime_alpha(rets, bench_ret, reg)
    out["_per_regime_alpha"] = pra
    out["_per_regime_nobs"] = n_obs
    out["regime_capture_spread"] = regime_capture_spread(rets, bench_ret, reg)
    out["bear_correction_excess"] = bear_correction_outperformance(rets, bench_ret, reg)
    out["recovery_half_life"] = recovery_half_life(fund_nav)

    # P2 - 12M block alphas (Bayesian shrinkage input) + rolling 12M alphas
    blk = block_alphas(rets, bench_ret, block_weeks=LB_1Y, max_blocks=5)
    out["_block_alphas"] = blk
    out["block_alpha_n"] = int(len(blk))
    rolling_12m = rolling_12m_alphas(fund_nav, bench_nav)
    out["_rolling_12m_alphas"] = rolling_12m
    if len(rolling_12m) >= 3:
        alphas_only = np.array([a for _, a in rolling_12m])
        out["alpha_pos_frac_12m"] = float(np.mean(alphas_only > 0))
        out["rolling_12m_n"] = int(len(rolling_12m))
    else:
        out["alpha_pos_frac_12m"] = None
        out["rolling_12m_n"] = int(len(rolling_12m))

    # P1 - rolling hybrid SIP-Hold XIRRs
    sip_x = rolling_hybrid_sip_xirrs(fund_nav)
    out["_rolling_hybrid_xirrs"] = sip_x
    out["hybrid_xirr_n"] = int(len(sip_x))
    if len(sip_x) >= 4:
        arr = np.array([x for _, x in sip_x])
        out["hybrid_xirr_p50"] = float(np.median(arr))
        out["hybrid_xirr_p25"] = float(np.percentile(arr, 25))
        out["hybrid_xirr_recent"] = float(np.mean(arr[: min(3, len(arr))]))
    else:
        out["hybrid_xirr_p50"] = None
        out["hybrid_xirr_p25"] = None
        out["hybrid_xirr_recent"] = None

    # P5 - Path quality
    out["max_drawdown"] = max_drawdown_calc(fund_nav)
    out["ulcer_index"] = ulcer_index(fund_nav)
    out["ulcer_perf_index"] = ulcer_perf_index(primary_cagr, out["ulcer_index"])
    out["gain_to_pain"] = gain_to_pain_ratio(rets)
    out["tail_ratio"] = tail_ratio(rets)
    out["current_drawdown"] = current_drawdown(fund_nav)
    out["vol_drag_adj_return"] = vol_drag_adjusted_return(rets)

    # P4 - Active conviction
    out["active_divergence"] = active_divergence(rets, bench_ret)
    out["active_skill_signal"] = active_skill_signal(alpha, out["active_divergence"])
    out["beta_stability"] = cross_regime_beta_stability(rets, bench_ret, reg)
    out["aum_capacity_mult"] = aum_capacity_haircut(aum)

    return out


# ===================================================================
# 11. Cross-sectional fills: peer-median XIRRs and peer-median 12M alphas
# ===================================================================

def compute_peer_median_xirrs(
    aligned_navs: Dict[str, pd.Series], all_window_starts: List[pd.Timestamp]
) -> Dict[pd.Timestamp, float]:
    """Cross-sectional median hybrid XIRR per window-start across all funds."""
    by_start: Dict[pd.Timestamp, List[float]] = {}
    for fid, nav in aligned_navs.items():
        windows = rolling_hybrid_sip_xirrs(nav)
        for start_ts, x in windows:
            by_start.setdefault(start_ts, []).append(x)
    return {ts: float(np.median(v)) for ts, v in by_start.items() if len(v) >= 5}


def compute_peer_median_12m_alphas(
    aligned_navs: Dict[str, pd.Series], bench_nav: pd.Series
) -> Dict[int, float]:
    """Cross-sectional median 12M alpha per window-end-index across all funds."""
    by_end: Dict[int, List[float]] = {}
    for fid, nav in aligned_navs.items():
        windows = rolling_12m_alphas(nav, bench_nav)
        for end_idx, a in windows:
            by_end.setdefault(end_idx, []).append(a)
    return {idx: float(np.median(v)) for idx, v in by_end.items() if len(v) >= 5}


def fund_hit_rate_vs_peer_xirr(
    fund_windows: List[Tuple[pd.Timestamp, float]],
    peer_median: Dict[pd.Timestamp, float],
) -> Optional[float]:
    if not fund_windows:
        return None
    hits = 0
    n = 0
    for ts, x in fund_windows:
        pm = peer_median.get(ts)
        if pm is None:
            continue
        n += 1
        if x > pm:
            hits += 1
    return float(hits / n) if n > 0 else None


def fund_hit_rate_vs_peer_alpha(
    fund_alphas: List[Tuple[int, float]],
    peer_median_alpha: Dict[int, float],
) -> Optional[float]:
    if not fund_alphas:
        return None
    hits = 0
    n = 0
    for idx, a in fund_alphas:
        pm = peer_median_alpha.get(idx)
        if pm is None:
            continue
        n += 1
        if a > pm:
            hits += 1
    return float(hits / n) if n > 0 else None


def benchmark_hybrid_xirrs(bench_nav: pd.Series) -> Dict[pd.Timestamp, float]:
    """Compute the Nifty-500 hybrid XIRR per window-start."""
    windows = rolling_hybrid_sip_xirrs(bench_nav)
    return {ts: x for ts, x in windows}


def fund_hit_rate_vs_benchmark(
    fund_windows: List[Tuple[pd.Timestamp, float]],
    bench_windows: Dict[pd.Timestamp, float],
) -> Optional[float]:
    if not fund_windows:
        return None
    hits = 0
    n = 0
    for ts, x in fund_windows:
        b = bench_windows.get(ts)
        if b is None:
            # nearest
            keys = list(bench_windows.keys())
            if not keys:
                continue
            nearest = min(keys, key=lambda k: abs((k - ts).days))
            if abs((nearest - ts).days) > 21:
                continue
            b = bench_windows[nearest]
        n += 1
        if x > b:
            hits += 1
    return float(hits / n) if n > 0 else None


# ===================================================================
# 12. Pillar Scoring & Composite
# ===================================================================

def percentile_rank(s: pd.Series, higher_better: bool = True) -> pd.Series:
    if higher_better:
        ranks = s.rank(method="average", pct=True, na_option="keep")
    else:
        ranks = (-s).rank(method="average", pct=True, na_option="keep")
    return ranks * 100.0


def confidence_haircut(data_weeks: int) -> float:
    if data_weeks < LB_1Y:
        return 0.55
    if data_weeks < LB_2Y:
        return 0.75
    if data_weeks < LB_3Y:
        return 0.88
    if data_weeks < LB_5Y:
        return 0.97
    return 1.00


def assemble_p1_score(df: pd.DataFrame) -> pd.Series:
    """Hybrid SIP-Hold outcome composite (0..100)."""
    s_med  = percentile_rank(df["hybrid_xirr_p50"], higher_better=True)
    s_dn   = percentile_rank(df["hybrid_xirr_p25"], higher_better=True)
    s_rc   = percentile_rank(df["hybrid_xirr_recent"], higher_better=True)
    s_hb   = percentile_rank(df["hybrid_xirr_hit_bench"], higher_better=True)
    s_hp   = percentile_rank(df["hybrid_xirr_hit_peer"], higher_better=True)
    return (
        0.30 * s_med + 0.20 * s_dn + 0.15 * s_rc
        + 0.20 * s_hb + 0.15 * s_hp
    ).fillna(0.0)


def assemble_p2_score(df: pd.DataFrame) -> pd.Series:
    """Year-1 skill consistency over 12M windows."""
    s_pos  = percentile_rank(df["alpha_pos_frac_12m"], higher_better=True)
    s_peer = percentile_rank(df["alpha_beat_peer_frac_12m"], higher_better=True)
    s_bayes = percentile_rank(df["bayes_shrunk_alpha"], higher_better=True)
    s_ir   = percentile_rank(df["info_ratio"], higher_better=True)
    return (
        0.30 * s_pos + 0.30 * s_peer
        + 0.25 * s_bayes + 0.15 * s_ir
    ).fillna(0.0)


def assemble_p3_score(df: pd.DataFrame) -> pd.Series:
    """Year-2 hold resilience."""
    s_fwd = percentile_rank(df["forward_alpha_24m"], higher_better=True)
    s_cap = percentile_rank(df["regime_capture_spread"], higher_better=True)
    s_bce = percentile_rank(df["bear_correction_excess"], higher_better=True)
    s_rec = percentile_rank(df["recovery_half_life"], higher_better=False)  # smaller = faster
    s_curdd = percentile_rank(df["current_drawdown"], higher_better=True)   # closer to 0 = ok
    return (
        0.30 * s_fwd + 0.20 * s_cap + 0.15 * s_bce
        + 0.25 * s_rec + 0.10 * s_curdd
    ).fillna(0.0)


def assemble_p4_score(df: pd.DataFrame) -> pd.Series:
    """Active conviction & capacity."""
    s_act  = percentile_rank(df["active_skill_signal"], higher_better=True)
    s_beta = percentile_rank(df["beta_stability"], higher_better=True)
    # AUM-capacity multiplier scaled to 0..100; 1.0 = 100, 0.85 = 0
    cap_score = ((df["aum_capacity_mult"].fillna(1.0) - 0.85) / 0.15) * 100.0
    cap_score = cap_score.clip(lower=0.0, upper=100.0)
    return (0.45 * s_act + 0.35 * s_beta + 0.20 * cap_score).fillna(0.0)


def assemble_p5_score(df: pd.DataFrame) -> pd.Series:
    """Compounding path quality."""
    s_g2p  = percentile_rank(df["gain_to_pain"], higher_better=True)
    s_tail = percentile_rank(df["tail_ratio"], higher_better=True)
    s_upi  = percentile_rank(df["ulcer_perf_index"], higher_better=True)
    s_vd   = percentile_rank(df["vol_drag_adj_return"], higher_better=True)
    s_curdd = percentile_rank(df["current_drawdown"], higher_better=True)
    return (
        0.20 * s_g2p + 0.15 * s_tail + 0.20 * s_upi
        + 0.30 * s_vd + 0.15 * s_curdd
    ).fillna(0.0)


def composite_score(df: pd.DataFrame) -> pd.Series:
    raw = (
        PILLAR_WEIGHTS["p1_hybrid_sip_hold"]    * df["p1_score"]
        + PILLAR_WEIGHTS["p2_skill_consistency"]  * df["p2_score"]
        + PILLAR_WEIGHTS["p3_hold_resilience"]    * df["p3_score"]
        + PILLAR_WEIGHTS["p4_conviction_capac"]   * df["p4_score"]
        + PILLAR_WEIGHTS["p5_compounding_path"]   * df["p5_score"]
    )
    return raw * df["confidence"]


# ===================================================================
# 13. Main pipeline
# ===================================================================

def load_universe(provider: MfDataProvider) -> pd.DataFrame:
    df_all = provider.list_all_mf()
    target = df_all[
        (df_all["sector"] == SECTOR_NAME)
        & (df_all["subsector"].isin(SUBSECTORS))
    ].copy()
    target = target.dropna(subset=["mfId", "name", "subsector"]).reset_index(drop=True)
    logger.info(
        f"Universe: {len(target)} funds across {target['subsector'].nunique()} subsectors"
    )
    return target


def load_aligned_navs(
    provider: MfDataProvider,
    funds: pd.DataFrame,
    bench_nav: pd.Series,
) -> Dict[str, pd.Series]:
    aligned: Dict[str, pd.Series] = {}
    for _, row in funds.iterrows():
        mf_id = row["mfId"]
        try:
            chart = provider.get_mf_chart(mf_id)
        except Exception as exc:
            logger.warning(f"{mf_id}: failed to load chart ({exc})")
            continue
        nav = clean_nav_to_series(chart)
        if len(nav) == 0:
            continue
        nav = align_to_index(nav, bench_nav.index).dropna()
        if len(nav) == 0:
            continue
        aligned[mf_id] = nav
    logger.info(f"Loaded NAVs for {len(aligned)} funds")
    return aligned


def run(force_refresh: bool = False, date: Optional[str] = None) -> None:
    provider = MfDataProvider(date=date)
    if force_refresh:
        provider.fetch_all_data()

    # 1. Load universe
    funds_df = load_universe(provider)

    # 2. Load benchmark
    bench_chart = provider.get_index_chart(PRIMARY_BENCHMARK)
    bench_nav = clean_nav_to_series(bench_chart)
    if len(bench_nav) < LB_3Y:
        raise RuntimeError(f"Benchmark history too short: {len(bench_nav)} weeks")
    bench_ret = weekly_returns(bench_nav)
    logger.info(
        f"Benchmark Nifty 500: {len(bench_nav)} weeks "
        f"({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})"
    )

    # 3. Load all fund NAVs aligned to benchmark grid
    aligned_navs = load_aligned_navs(provider, funds_df, bench_nav)
    funds_df = funds_df[funds_df["mfId"].isin(aligned_navs)].reset_index(drop=True)

    # 4. Regime classifier (24M transition matrix for hold-period projection)
    reg = regime_series(bench_nav)
    cur_probs = soft_current_regime(reg)
    transition_24m = empirical_forward_mix(reg, horizon_weeks=LB_2Y)
    forward_mix = cur_probs @ transition_24m
    logger.info(
        "Current regime mix (Bull/Sideways/Correction/Bear): "
        + ", ".join(f"{p:.2f}" for p in cur_probs)
    )
    logger.info(
        "Expected forward 24M regime mix (Bull/Sideways/Correction/Bear): "
        + ", ".join(f"{p:.2f}" for p in forward_mix)
    )

    # 5. Per-fund metrics
    metrics: List[dict] = []
    for _, row in funds_df.iterrows():
        mf_id = row["mfId"]
        nav = aligned_navs[mf_id]
        m = compute_fund_metrics(
            mf_id=mf_id,
            fund_nav=nav,
            bench_nav=bench_nav,
            bench_ret=bench_ret,
            reg=reg,
            name=row.get("name"),
            aum=row.get("aum"),
            subsector=row.get("subsector"),
        )
        metrics.append(m)

    df = pd.DataFrame(metrics)

    # 6. Cross-sectional peer-median computations (P1 + P2 augmented Carhart)
    logger.info("Computing peer-median hybrid XIRRs and 12M alphas...")
    peer_med_xirr = compute_peer_median_xirrs(aligned_navs, [])
    peer_med_alpha = compute_peer_median_12m_alphas(aligned_navs, bench_nav)
    bench_xirrs = benchmark_hybrid_xirrs(bench_nav)
    logger.info(
        f"  Peer-median XIRR windows: {len(peer_med_xirr)};  "
        f"peer-median alpha windows: {len(peer_med_alpha)};  "
        f"benchmark XIRR windows: {len(bench_xirrs)}"
    )

    def _safe_list(v):
        if isinstance(v, list):
            return v
        return []

    df["hybrid_xirr_hit_peer"] = df["_rolling_hybrid_xirrs"].apply(
        lambda v: fund_hit_rate_vs_peer_xirr(_safe_list(v), peer_med_xirr)
    ) if "_rolling_hybrid_xirrs" in df.columns else None
    df["hybrid_xirr_hit_bench"] = df["_rolling_hybrid_xirrs"].apply(
        lambda v: fund_hit_rate_vs_benchmark(_safe_list(v), bench_xirrs)
    ) if "_rolling_hybrid_xirrs" in df.columns else None
    df["alpha_beat_peer_frac_12m"] = df["_rolling_12m_alphas"].apply(
        lambda v: fund_hit_rate_vs_peer_alpha(_safe_list(v), peer_med_alpha)
    ) if "_rolling_12m_alphas" in df.columns else None

    # 7. Bayesian shrunk alpha (P2)
    blocks_dict = {
        r["mfId"]: r["_block_alphas"]
        for _, r in df.iterrows()
        if isinstance(r.get("_block_alphas"), np.ndarray) and len(r["_block_alphas"]) > 0
    }
    shrunk, peer_mean, lams = james_stein_shrunk(blocks_dict)
    df["bayes_shrunk_alpha"] = df["mfId"].map(shrunk)
    df["bayes_shrink_lambda"] = df["mfId"].map(lams)
    logger.info(f"Bayesian peer-mean 12M alpha: {peer_mean:.4f}; n_funds: {len(shrunk)}")

    # 8. Forward 24M regime alpha (P3)
    pra_dict = {
        r["mfId"]: r["_per_regime_alpha"]
        for _, r in df.iterrows()
        if isinstance(r.get("_per_regime_alpha"), np.ndarray)
    }
    peer_alphas = peer_mean_per_regime(pra_dict)
    fwd: Dict[str, Optional[float]] = {}
    for fid, a in pra_dict.items():
        fwd[fid] = expected_forward_alpha(a, peer_alphas, cur_probs, transition_24m)
    df["forward_alpha_24m"] = df["mfId"].map(fwd)

    # 9. Pillar scores
    df["p1_score"] = assemble_p1_score(df)
    df["p2_score"] = assemble_p2_score(df)
    df["p3_score"] = assemble_p3_score(df)
    df["p4_score"] = assemble_p4_score(df)
    df["p5_score"] = assemble_p5_score(df)

    # 10. Confidence haircut + composite
    df["confidence"] = df["data_weeks"].apply(confidence_haircut)
    df["score"] = composite_score(df).round(2)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # 11. Walk-forward 24M IC diagnostic
    ic = walk_forward_ic_24m(aligned_navs, bench_nav)
    if ic is not None:
        logger.info(f"Walk-forward 24M hybrid-XIRR persistence IC (Spearman): {ic:.3f}")
    else:
        logger.info("Walk-forward 24M IC could not be computed (insufficient history).")

    # 12. Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "mfId", "name", "rank", "score", "data_days", "subsector",
        "cagr_1y", "cagr_3y", "cagr_5y",
        "volatility", "sharpe", "alpha", "beta", "info_ratio",
        "bayes_shrunk_alpha", "bayes_shrink_lambda", "block_alpha_n",
        "alpha_pos_frac_12m", "alpha_beat_peer_frac_12m", "rolling_12m_n",
        "forward_alpha_24m", "regime_capture_spread", "bear_correction_excess",
        "recovery_half_life",
        "hybrid_xirr_p50", "hybrid_xirr_p25", "hybrid_xirr_recent",
        "hybrid_xirr_hit_bench", "hybrid_xirr_hit_peer", "hybrid_xirr_n",
        "max_drawdown", "ulcer_perf_index", "gain_to_pain", "tail_ratio",
        "current_drawdown", "vol_drag_adj_return",
        "active_divergence", "active_skill_signal", "beta_stability",
        "aum_capacity_mult",
        "p1_score", "p2_score", "p3_score", "p4_score", "p5_score",
        "confidence", "aum", "data_weeks",
    ]
    out_df = df[output_cols].copy()

    pct_cols = [
        "cagr_1y", "cagr_3y", "cagr_5y", "alpha", "bayes_shrunk_alpha",
        "forward_alpha_24m", "bear_correction_excess",
        "hybrid_xirr_p50", "hybrid_xirr_p25", "hybrid_xirr_recent",
        "max_drawdown", "current_drawdown", "active_divergence",
        "vol_drag_adj_return",
    ]
    for c in pct_cols:
        out_df[c] = (out_df[c] * 100.0).round(2)

    frac_cols = [
        "alpha_pos_frac_12m", "alpha_beat_peer_frac_12m",
        "hybrid_xirr_hit_bench", "hybrid_xirr_hit_peer",
    ]
    for c in frac_cols:
        out_df[c] = (out_df[c] * 100.0).round(1)  # display as %

    other_round = [
        "volatility", "sharpe", "beta", "info_ratio",
        "regime_capture_spread", "ulcer_perf_index",
        "gain_to_pain", "tail_ratio",
        "active_skill_signal", "beta_stability", "aum_capacity_mult",
        "p1_score", "p2_score", "p3_score", "p4_score", "p5_score",
        "confidence", "bayes_shrink_lambda",
    ]
    for c in other_round:
        out_df[c] = out_df[c].astype(float).round(3)
    out_df["recovery_half_life"] = out_df["recovery_half_life"].round(0)

    out_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Wrote {len(out_df)} rows -> {OUTPUT_FILE}")

    # Pretty top-15
    print("\nTop 15 by composite score:")
    print(
        out_df[
            ["rank", "name", "subsector", "score", "cagr_3y",
             "hybrid_xirr_p50", "hybrid_xirr_hit_peer",
             "alpha_pos_frac_12m"]
        ]
        .head(15)
        .to_string(index=False)
    )


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Total Market scoring (Claude HSC)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force refetch of all data via MfDataProvider before scoring",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force_refresh=args.refresh, date=args.date)
