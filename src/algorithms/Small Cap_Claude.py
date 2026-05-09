#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm — SIP-XIRR-first Self-Backtested Model
==============================================================================

Objective
---------
Predict the next 1-year MONTHLY SIP XIRR (1st-of-month buys for 12 months,
redemption at month 12) for every Indian Small Cap mutual fund and rank
accordingly.

Distinctive approach
--------------------
Instead of weighting features by intuition, the model SELF-BACKTESTS each
candidate signal against realized forward 1Y SIP XIRR over the available NAV
history, and uses the resulting Spearman correlations as adaptive weights for
the cross-sectional composite. Features whose past correlation is non-positive
are dropped, so only evidence-supported signals contribute. The directionality
of every signal is learned from the data, not asserted.

Pipeline
--------
1. Load NAV history for every Small Cap fund + Nifty SmallCap 250 benchmark.
2. Resample to month-start NAV (the SIP buy NAV; a SIP investor buys 1 unit
   on the 1st of each month).
3. Compute the rolling 1Y SIP XIRR series for each fund and the benchmark via
   NPV bisection.
4. For every historical month t with 12+ months of forward data:
     - compute features using only data up to t (look-ahead safe)
     - record the realized forward 1Y SIP XIRR starting at t
5. Compute per-feature Spearman correlation with forward SIP XIRR across the
   pooled (fund x time) panel. These correlations become adaptive weights.
6. At "now" (the latest NAV date) z-score each feature cross-sectionally,
   sign-correct using the learned correlation sign, weight, and convert to a
   0-100 percentile score. Apply a track-record haircut for funds with <3Y
   of data.

Author : Claude
Sector : Small Cap Fund
"""

import argparse
import logging
import sys
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

from mf_data_provider import MfDataProvider  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
BENCHMARK_INDEX = "Small Cap"          # resolves to .NISM250
RISK_FREE_RATE = 0.065                 # ~6.5 % Indian T-bill proxy

SIP_MONTHS = 12                        # task: monthly SIP for 1 year
SIP_AMOUNT = 1.0                       # arbitrary; XIRR is amount-invariant

WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 365.25

MIN_MONTHS_FOR_SIP = SIP_MONTHS + 1    # 12 buys + 1 redemption point
MIN_MONTHS_FOR_BACKTEST = SIP_MONTHS + 1
MIN_DATA_DAYS_FULL_CONFIDENCE = int(3 * DAYS_PER_YEAR)
TRACK_RECORD_HAIRCUT = 0.85
NOISE_FLOOR_RHO = 0.05                 # drop features with |rho| < this

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_Claude.csv"


# ===================================================================
# SIP XIRR machinery
# ===================================================================

def _to_month_start_nav(nav_series: pd.Series) -> pd.Series:
    """
    Build a clean month-start NAV series.

    Tickertape NAVs are roughly weekly. We forward-fill to daily so a SIP
    investor can "buy" on the 1st of each month using the most recent
    available NAV (real-world behaviour: buy at the next NAV after the SIP
    instruction date).
    """
    if nav_series.empty:
        return nav_series
    daily = nav_series.resample("D").ffill()
    monthly = daily.resample("MS").first().dropna()
    return monthly


def _xirr_bisect(
    cashflows: List[float],
    times_years: List[float],
    lo: float = -0.99,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Solve NPV(r) = sum(cf_i / (1+r)^t_i) = 0 via bisection.

    Returns None if the NPV does not change sign on [lo, hi] or if any
    intermediate value is non-finite.
    """
    def npv(r: float) -> float:
        try:
            return float(sum(cf / ((1.0 + r) ** t) for cf, t in zip(cashflows, times_years)))
        except (OverflowError, ZeroDivisionError):
            return float("nan")

    f_lo = npv(lo)
    f_hi = npv(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return None
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if not np.isfinite(f_mid):
            return None
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def sip_xirr_at(monthly_nav: pd.Series, start_idx: int, months: int = SIP_MONTHS) -> Optional[float]:
    """
    Annualised XIRR of a monthly SIP starting at monthly_nav[start_idx].

    Cashflows: -SIP_AMOUNT at each of the next `months` month-starts, then
    +SIP_AMOUNT * sum(1/nav_i) * nav_redeem at the redemption month.
    """
    end_idx = start_idx + months
    if start_idx < 0 or end_idx >= len(monthly_nav):
        return None

    invest_navs = monthly_nav.iloc[start_idx:end_idx]
    redeem_nav = monthly_nav.iloc[end_idx]
    if (invest_navs <= 0).any() or redeem_nav <= 0:
        return None

    invest_dates = monthly_nav.index[start_idx:end_idx]
    redeem_date = monthly_nav.index[end_idx]

    units = SIP_AMOUNT / invest_navs.values
    redeem_value = float(units.sum() * redeem_nav)

    cashflows = [-float(SIP_AMOUNT)] * months + [redeem_value]
    times = [(d - invest_dates[0]).days / DAYS_PER_YEAR for d in invest_dates]
    times.append((redeem_date - invest_dates[0]).days / DAYS_PER_YEAR)

    return _xirr_bisect(cashflows, times)


def rolling_sip_xirr_series(monthly_nav: pd.Series, months: int = SIP_MONTHS) -> pd.Series:
    """
    Series of realized 1Y SIP XIRRs.

    Indexed by SIP START date (first cashflow). Value at index t is the
    XIRR of a SIP that started at month t and redeemed at month t+months.
    """
    if len(monthly_nav) < months + 1:
        return pd.Series(dtype=float)

    out = {}
    for i in range(len(monthly_nav) - months):
        r = sip_xirr_at(monthly_nav, i, months=months)
        if r is not None and np.isfinite(r):
            out[monthly_nav.index[i]] = r
    return pd.Series(out, dtype=float).sort_index()


# ===================================================================
# Feature functions (all look-ahead safe — operate on data up to t only)
# ===================================================================

def _annualised_cagr(nav_series: pd.Series, years: float) -> Optional[float]:
    """CAGR over the trailing `years` years of `nav_series` (assumes index sorted)."""
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=int(years * DAYS_PER_YEAR))
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 5 or window.iloc[0] <= 0:
        return None
    actual_years = (window.index[-1] - window.index[0]).days / DAYS_PER_YEAR
    if actual_years < years * 0.85:  # require at least 85 % of requested span
        return None
    return float((window.iloc[-1] / window.iloc[0]) ** (1.0 / actual_years) - 1.0)


def _drawdown_depth(nav_series: pd.Series, lookback_years: float = 3.0) -> Optional[float]:
    """Current NAV relative to trailing peak (negative if in drawdown)."""
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=int(lookback_years * DAYS_PER_YEAR))
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 5:
        return None
    peak = window.max()
    if peak <= 0:
        return None
    return float(window.iloc[-1] / peak - 1.0)


def _recovery_slope(nav_series: pd.Series, days: int = 90) -> Optional[float]:
    """
    Slope of log(NAV) regressed on time over the last `days` days,
    expressed as an annualised rate (positive = uptrend).
    """
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=days)
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 5 or (window <= 0).any():
        return None
    t = np.array([(d - window.index[0]).days for d in window.index], dtype=float)
    y = np.log(window.values)
    if t[-1] - t[0] < days * 0.5:
        return None
    slope, _ = np.polyfit(t, y, 1)         # log-units per day
    return float(slope * DAYS_PER_YEAR)    # annualised


def _late_cycle_momentum(nav_series: pd.Series, days: int = 180) -> Optional[float]:
    """Return over the last `days` days (raw, not annualised)."""
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=days)
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 5 or window.iloc[0] <= 0:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def _weekly_returns(nav_series: pd.Series) -> pd.Series:
    """Return a clean weekly-return series (pct_change of the raw weekly NAV)."""
    return nav_series.pct_change().dropna()


def _aligned_weekly_returns(
    fund_nav: pd.Series, bench_nav: pd.Series, lookback_years: float
) -> Tuple[pd.Series, pd.Series]:
    """Return weekly fund/bench returns aligned by inner-join, restricted to lookback."""
    end_date = fund_nav.index[-1]
    start_date = end_date - pd.Timedelta(days=int(lookback_years * DAYS_PER_YEAR))
    f = fund_nav[fund_nav.index >= start_date]
    b = bench_nav[bench_nav.index >= start_date]
    f_ret = _weekly_returns(f)
    b_ret = _weekly_returns(b)
    aligned = pd.concat([f_ret.rename("f"), b_ret.rename("b")], axis=1, join="inner").dropna()
    return aligned["f"], aligned["b"]


def _capture_ratios(
    fund_nav: pd.Series, bench_nav: pd.Series, lookback_years: float = 2.0
) -> Tuple[Optional[float], Optional[float]]:
    """Up- and down-capture ratios on weekly returns over the trailing window."""
    f, b = _aligned_weekly_returns(fund_nav, bench_nav, lookback_years)
    if len(f) < 20:
        return None, None

    up = b > 0
    down = b < 0

    up_cap = None
    if up.sum() >= 5:
        b_up_mean = b[up].mean()
        if abs(b_up_mean) > 1e-12:
            up_cap = float(f[up].mean() / b_up_mean)

    down_cap = None
    if down.sum() >= 5:
        b_down_mean = b[down].mean()
        if abs(b_down_mean) > 1e-12:
            down_cap = float(f[down].mean() / b_down_mean)

    return up_cap, down_cap


def _info_ratio(
    fund_nav: pd.Series, bench_nav: pd.Series, lookback_years: float = 2.0
) -> Optional[float]:
    """Annualised information ratio over the trailing window."""
    f, b = _aligned_weekly_returns(fund_nav, bench_nav, lookback_years)
    if len(f) < 20:
        return None
    active = f - b
    te = active.std()
    if te is None or te == 0 or not np.isfinite(te):
        return None
    return float((active.mean() * WEEKS_PER_YEAR) / (te * np.sqrt(WEEKS_PER_YEAR)))


def _sortino(nav_series: pd.Series, lookback_years: float = 3.0) -> Optional[float]:
    """Annualised Sortino ratio using weekly returns over the trailing window."""
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=int(lookback_years * DAYS_PER_YEAR))
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 30:
        return None
    rets = _weekly_returns(window)
    if len(rets) < 20:
        return None
    weekly_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / WEEKS_PER_YEAR) - 1.0
    excess = rets - weekly_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    dd = float(np.sqrt((downside ** 2).mean()) * np.sqrt(WEEKS_PER_YEAR))
    if dd == 0:
        return None
    return float((excess.mean() * WEEKS_PER_YEAR) / dd)


def _trend_residual(nav_series: pd.Series, lookback_years: float = 5.0) -> Optional[float]:
    """
    z-scored residual of current log(NAV) vs a fitted log-linear 5Y trend.
    Negative => below trend (mean-reversion candidate); positive => above trend.
    """
    if nav_series.empty:
        return None
    end_date = nav_series.index[-1]
    start_date = end_date - pd.Timedelta(days=int(lookback_years * DAYS_PER_YEAR))
    window = nav_series[nav_series.index >= start_date]
    if len(window) < 30 or (window <= 0).any():
        return None
    t = np.array([(d - window.index[0]).days for d in window.index], dtype=float)
    y = np.log(window.values)
    slope, intercept = np.polyfit(t, y, 1)
    residuals = y - (slope * t + intercept)
    sd = residuals.std()
    if sd == 0 or not np.isfinite(sd):
        return None
    return float(residuals[-1] / sd)


def _sip_history_features(
    fund_sip: pd.Series, bench_sip: pd.Series
) -> Dict[str, Optional[float]]:
    """
    Summary statistics of the past rolling 1Y SIP XIRR series.

    `fund_sip` and `bench_sip` are indexed by SIP-start date, value = realized
    1Y SIP XIRR. Both must already exclude the future (we only look at SIPs
    that have already redeemed by `now`).
    """
    out: Dict[str, Optional[float]] = {
        "sip_1y_median": None,
        "sip_1y_p20": None,
        "sip_alpha_median": None,
        "sip_consistency": None,
    }
    if fund_sip.empty:
        return out

    out["sip_1y_median"] = float(fund_sip.median())
    out["sip_1y_p20"] = float(fund_sip.quantile(0.20))

    if not bench_sip.empty:
        aligned = pd.concat(
            [fund_sip.rename("f"), bench_sip.rename("b")], axis=1, join="inner"
        ).dropna()
        if len(aligned) >= 6:
            alpha = aligned["f"] - aligned["b"]
            out["sip_alpha_median"] = float(alpha.median())
            out["sip_consistency"] = float((alpha > 0).mean())

    return out


# ===================================================================
# Feature panel builder (look-ahead safe)
# ===================================================================

FEATURE_COLS = [
    "sip_1y_median",
    "sip_1y_p20",
    "sip_alpha_median",
    "sip_consistency",
    "dd_depth",
    "bench_dd_depth",
    "recovery_slope",
    "late_mom_6m",
    "up_capture_2y",
    "down_capture_2y",
    "info_ratio_2y",
    "nav_trend_residual",
    "bench_trend_residual",
    "sortino_3y",
    "cagr_3y",
    "cagr_5y",
]

# Directional priors:
#   +1  : higher is unambiguously better (locked sign)
#   -1  : lower is unambiguously better (locked sign)
#    0  : genuinely two-sided -> data drives the sign
#
# A locked feature is only used if the self-backtested cross-sectional
# correlation AGREES with the prior; otherwise it is dropped (we refuse to
# bet against a financial-fundamentals signal on the basis of a noisy weak
# negative correlation). Locked-feature weight is |rho|, same as before.
FEATURE_PRIORS: Dict[str, int] = {
    # SIP-quality signals: higher SIP outcome / consistency / alpha = better.
    "sip_1y_median":        +1,
    "sip_1y_p20":           +1,
    "sip_alpha_median":     +1,
    "sip_consistency":      +1,
    # Risk-adjusted skill metrics: higher = better.
    "info_ratio_2y":        +1,
    "sortino_3y":           +1,
    "up_capture_2y":        +1,
    "recovery_slope":       +1,
    # Down-capture: lower = better (less hurt during benchmark drawdowns).
    "down_capture_2y":      -1,
    # Genuinely two-sided -- could go either way and we WANT to learn:
    "cagr_3y":               0,   # past CAGR may persist or mean-revert
    "cagr_5y":               0,
    "late_mom_6m":           0,   # momentum vs. mean-reversion
    "dd_depth":              0,   # depressed price -> rebound, or weak fund
    "bench_dd_depth":        0,   # regime gauge
    "nav_trend_residual":    0,   # above/below trend -- mean-reversion signal
    "bench_trend_residual":  0,
}


def _features_at(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    fund_sip_completed: pd.Series,
    bench_sip_completed: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    Compute every feature using only the data passed in (which the caller has
    already truncated to <= t). The SIP series passed in must contain only
    rolling 1Y SIPs that have ALREADY REDEEMED by t (no leakage).
    """
    feats: Dict[str, Optional[float]] = {}

    feats.update(_sip_history_features(fund_sip_completed, bench_sip_completed))

    feats["dd_depth"] = _drawdown_depth(fund_nav, lookback_years=3.0)
    feats["bench_dd_depth"] = _drawdown_depth(bench_nav, lookback_years=3.0)
    feats["recovery_slope"] = _recovery_slope(fund_nav, days=90)
    feats["late_mom_6m"] = _late_cycle_momentum(fund_nav, days=180)

    up_cap, down_cap = _capture_ratios(fund_nav, bench_nav, lookback_years=2.0)
    feats["up_capture_2y"] = up_cap
    feats["down_capture_2y"] = down_cap
    feats["info_ratio_2y"] = _info_ratio(fund_nav, bench_nav, lookback_years=2.0)

    feats["nav_trend_residual"] = _trend_residual(fund_nav, lookback_years=5.0)
    feats["bench_trend_residual"] = _trend_residual(bench_nav, lookback_years=5.0)

    feats["sortino_3y"] = _sortino(fund_nav, lookback_years=3.0)
    feats["cagr_3y"] = _annualised_cagr(fund_nav, years=3.0)
    feats["cagr_5y"] = _annualised_cagr(fund_nav, years=5.0)

    return feats


def build_history_panel(
    fund_data: Dict[str, Dict[str, pd.Series]],
    bench_nav: pd.Series,
    bench_monthly: pd.Series,
    bench_sip: pd.Series,
) -> pd.DataFrame:
    """
    Build a (fund, sip_start_date) -> features + realized_forward_sip_xirr panel.

    For each fund and each candidate SIP-start month t with 12+ months of
    forward data, we compute features using only data up to (and including) t,
    and pair them with the realized 1Y SIP XIRR starting at t.
    """
    rows: List[Dict] = []

    for mf_id, payload in fund_data.items():
        fund_nav: pd.Series = payload["nav"]
        fund_monthly: pd.Series = payload["monthly"]
        fund_sip: pd.Series = payload["sip"]

        if len(fund_monthly) < 2 * SIP_MONTHS + 1:
            # Need at least one historical SIP that has redeemed before any
            # forward SIP would have started. Otherwise the panel is empty.
            continue

        # Iterate over candidate SIP-start dates t. We need:
        #   - forward 1Y SIP from t available (i.e. fund_sip[t] exists)
        #   - some completed past SIP history strictly before t
        for t in fund_sip.index:
            redeem_date = t + pd.DateOffset(months=SIP_MONTHS)

            fund_nav_at_t = fund_nav[fund_nav.index <= t]
            bench_nav_at_t = bench_nav[bench_nav.index <= t]
            if len(fund_nav_at_t) < 30 or len(bench_nav_at_t) < 30:
                continue

            # SIPs that REDEEMED on or before t (their start <= t - SIP_MONTHS)
            cutoff_start = t - pd.DateOffset(months=SIP_MONTHS)
            fund_sip_done = fund_sip[fund_sip.index <= cutoff_start]
            bench_sip_done = bench_sip[bench_sip.index <= cutoff_start]

            feats = _features_at(
                fund_nav_at_t, bench_nav_at_t, fund_sip_done, bench_sip_done
            )

            row = {"mfId": mf_id, "t": t, "forward_sip_xirr": float(fund_sip.loc[t])}
            row.update(feats)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["mfId", "t", "forward_sip_xirr"] + FEATURE_COLS)

    panel = pd.DataFrame(rows)
    return panel


# ===================================================================
# Self-backtest -> adaptive weights
# ===================================================================

def _cross_sectional_rank_corr(panel: pd.DataFrame, feature_col: str) -> Optional[float]:
    """
    Cross-sectional rank correlation pooled across time.

    For each time slice t we rank funds by `feature_col` and by
    `forward_sip_xirr` (both expressed as cross-sectional percentile ranks),
    then compute Pearson correlation on the pooled (rank, rank) pairs. This
    is exactly what the score will exploit cross-sectionally: "does ranking
    by X at time t predict the ranking of forward outcomes at time t?"

    This sidesteps time-regime pooling effects: e.g. ALL funds doing well
    in a bull-market window otherwise inflates feature/target co-movement
    that doesn't translate to better cross-sectional picks.
    """
    if feature_col not in panel.columns:
        return None
    sub = panel[["t", feature_col, "forward_sip_xirr"]].dropna()
    if sub.empty:
        return None
    sub = sub.copy()
    # Require at least 5 funds per time slice to rank meaningfully
    counts = sub.groupby("t").size()
    valid_ts = counts[counts >= 5].index
    sub = sub[sub["t"].isin(valid_ts)]
    if len(sub) < 50:
        return None
    sub["x_rank"] = sub.groupby("t")[feature_col].rank(pct=True)
    sub["y_rank"] = sub.groupby("t")["forward_sip_xirr"].rank(pct=True)
    if sub["x_rank"].nunique() < 2 or sub["y_rank"].nunique() < 2:
        return None
    rho = sub["x_rank"].corr(sub["y_rank"])
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def compute_feature_weights(
    panel: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float], Dict[str, str]]:
    """
    Returns (weights, signs, correlations, statuses).

    weights        : non-negative, sum to 1. Final composite weight per feature.
    signs          : resolved sign (+1 or -1) used by the score, AFTER applying
                     the prior + data agreement check.
    correlations   : raw cross-sectional Spearman rho per feature (informational).
    statuses       : human-readable status per feature for the diagnostic table.
                     One of: "locked", "drop_disagree", "data", "drop_noise",
                     "drop_no_data".

    Resolution rules:
      - Feature with prior +/- and data agrees (sign(rho) == prior, or |rho|
        below noise so prior wins): keep, sign = prior, weight ~ |rho|.
      - Feature with prior +/- and data DISAGREES with magnitude above the
        noise floor: drop. We refuse to bet against a fundamentally directed
        signal on the basis of a noisy negative correlation.
      - Feature with prior 0 (two-sided): keep iff |rho| >= NOISE_FLOOR_RHO,
        sign = sign(rho), weight ~ |rho|.
    """
    correlations: Dict[str, float] = {}
    statuses: Dict[str, str] = {}
    raw_weights: Dict[str, float] = {}
    signs: Dict[str, int] = {}

    for col in FEATURE_COLS:
        rho = _cross_sectional_rank_corr(panel, col)
        if rho is None:
            statuses[col] = "drop_no_data"
            continue
        correlations[col] = rho
        prior = FEATURE_PRIORS.get(col, 0)

        if prior != 0:
            # Locked-direction feature.
            if (rho * prior) >= 0 or abs(rho) < NOISE_FLOOR_RHO:
                # Data agrees (or is too noisy to disagree) -> keep, sign = prior.
                # Weight is |rho| but with a tiny floor so a strong-prior feature
                # with rho ~= 0 still contributes a sliver (otherwise the prior
                # is wasted whenever the panel produces near-zero correlations).
                w = max(abs(rho), NOISE_FLOOR_RHO)
                raw_weights[col] = w
                signs[col] = prior
                statuses[col] = "locked"
            else:
                # Data meaningfully contradicts the prior -> drop the feature.
                statuses[col] = "drop_disagree"
        else:
            # Data-driven feature.
            if abs(rho) >= NOISE_FLOOR_RHO:
                raw_weights[col] = abs(rho)
                signs[col] = 1 if rho >= 0 else -1
                statuses[col] = "data"
            else:
                statuses[col] = "drop_noise"

    total = sum(raw_weights.values())
    if total == 0:
        # Degenerate: nothing usable. Fall back to equal weight on the most
        # robust prior-positive SIP-quality features so we still produce a
        # sensible ranking.
        fallback = ["sip_alpha_median", "sip_1y_median",
                    "info_ratio_2y", "sortino_3y", "sip_consistency"]
        present = [c for c in fallback if FEATURE_PRIORS.get(c, 0) == 1]
        if not present:
            return {}, {}, correlations, statuses
        w = 1.0 / len(present)
        weights = {c: w for c in present}
        signs = {c: 1 for c in present}
        for c in present:
            statuses[c] = "fallback"
        return weights, signs, correlations, statuses

    weights = {c: w / total for c, w in raw_weights.items()}
    return weights, signs, correlations, statuses


# ===================================================================
# Cross-sectional scoring at "now"
# ===================================================================

def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; NaN preserved."""
    mu = s.mean()
    sd = s.std()
    if sd is None or sd == 0 or not np.isfinite(sd):
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return (s - mu) / sd


def composite_score(
    features_now: pd.DataFrame,
    weights: Dict[str, float],
    signs: Dict[str, int],
) -> pd.Series:
    """
    Compute the composite raw score per fund.

    For each weighted feature:
      z = cross-sectional z-score
      contribution = w * sign * z, where `sign` is the resolved direction
      (prior or data-driven) from compute_feature_weights().
    Funds with NaN on a feature have its weight redistributed (so absent data
    does not zero them out).
    """
    contribs = pd.DataFrame(index=features_now.index, dtype=float)
    weight_panel = pd.DataFrame(index=features_now.index, dtype=float)

    for col, w in weights.items():
        if col not in features_now.columns:
            continue
        z = _zscore(features_now[col].astype(float))
        sign = float(signs.get(col, 1))
        contribs[col] = w * sign * z
        weight_panel[col] = np.where(z.notna(), w, np.nan)

    if contribs.empty:
        return pd.Series(0.0, index=features_now.index)

    raw_sum = contribs.sum(axis=1, skipna=True)
    used_weight = weight_panel.sum(axis=1, skipna=True)
    # Renormalise so a fund missing a feature isn't penalised (other than the
    # track-record haircut applied later).
    raw = np.where(used_weight > 0, raw_sum / used_weight, 0.0)
    return pd.Series(raw, index=features_now.index)


# ===================================================================
# Main pipeline
# ===================================================================

def _load_nav(provider: MfDataProvider, mf_id: str) -> Optional[pd.Series]:
    chart = provider.get_mf_chart(mf_id)
    if chart is None or len(chart) < 30:
        return None
    chart = chart.copy()
    chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
    chart = chart.sort_values("timestamp").drop_duplicates("timestamp")
    nav = chart.set_index("timestamp")["nav"].astype(float)
    nav = nav[nav > 0]
    if len(nav) < 30:
        return None
    return nav


def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 72)
    print("  SMALL CAP MUTUAL FUND SCORING - CLAUDE MODEL")
    print("  Objective: predict next-1Y monthly-SIP XIRR")
    print(f"  Benchmark: Nifty SmallCap 250 ({BENCHMARK_INDEX})")
    print("=" * 72)

    provider = MfDataProvider(date=date)

    # ---- Benchmark ----
    bench_df = provider.get_index_chart(BENCHMARK_INDEX)
    bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"], utc=True)
    bench_df = bench_df.sort_values("timestamp").drop_duplicates("timestamp")
    bench_nav = bench_df.set_index("timestamp")["nav"].astype(float)
    bench_nav = bench_nav[bench_nav > 0]
    bench_monthly = _to_month_start_nav(bench_nav)
    bench_sip = rolling_sip_xirr_series(bench_monthly)
    print(
        f"  Benchmark: {len(bench_nav)} weekly NAVs "
        f"({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})  "
        f"| {len(bench_sip)} historical 1Y SIP outcomes"
    )

    # ---- Fund universe ----
    df_all = provider.list_all_mf()
    funds_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    print(f"  Small Cap funds in universe: {len(funds_df)}")

    # ---- Load all funds + precompute monthly + SIP series ----
    fund_data: Dict[str, Dict[str, pd.Series]] = {}
    fund_meta: Dict[str, Dict[str, object]] = {}

    for _, row in funds_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = float(row.get("aum", 0) or 0)

        try:
            nav = _load_nav(provider, mf_id)
            if nav is None:
                logger.warning(f"Skipping {mf_id} ({name}): insufficient NAV data")
                continue
            monthly = _to_month_start_nav(nav)
            if len(monthly) < MIN_MONTHS_FOR_SIP:
                logger.warning(
                    f"Skipping {mf_id} ({name}): only {len(monthly)} monthly points"
                )
                continue
            sip = rolling_sip_xirr_series(monthly)

            fund_data[mf_id] = {"nav": nav, "monthly": monthly, "sip": sip}
            fund_meta[mf_id] = {
                "name": name,
                "aum": round(aum, 2),
                "data_days": int((nav.index.max() - nav.index.min()).days + 1),
                "n_sip_windows": len(sip),
            }
        except Exception as e:
            logger.error(f"Error loading {mf_id} ({name}): {e}")
            continue

    print(f"  Funds with usable NAV history: {len(fund_data)}")
    if not fund_data:
        logger.error("No usable funds. Exiting.")
        sys.exit(1)

    # ---- Build (look-ahead safe) historical panel for self-backtest ----
    print("\n  Building historical (fund x time) feature panel...")
    panel = build_history_panel(fund_data, bench_nav, bench_monthly, bench_sip)
    print(f"  Panel rows: {len(panel)} across {panel['mfId'].nunique() if not panel.empty else 0} funds")

    # ---- Self-backtest: learn feature weights ----
    weights, signs, correlations, statuses = compute_feature_weights(panel)

    print("\n  Self-backtested feature signal vs forward 1Y SIP XIRR")
    print("  " + "-" * 76)
    print(f"  {'feature':<24}{'prior':>7}{'rho':>10}{'sign':>7}{'weight':>14}{'status':>16}")
    print("  " + "-" * 76)
    prior_label = {1: "+", -1: "-", 0: "data"}
    for col in FEATURE_COLS:
        prior = FEATURE_PRIORS.get(col, 0)
        rho = correlations.get(col)
        w = weights.get(col, 0.0)
        sgn = signs.get(col)
        status = statuses.get(col, "")
        rho_str = f"{rho:+.3f}" if rho is not None else "  n/a"
        sgn_str = ("+" if sgn == 1 else "-") if sgn is not None else " "
        print(
            f"  {col:<24}{prior_label[prior]:>7}{rho_str:>10}{sgn_str:>7}"
            f"{w*100:>12.2f}%{status:>16}"
        )
    print("  " + "-" * 76)

    # ---- Compute features at "now" for every fund ----
    print("\n  Computing features at latest date for ranking...")
    now_rows: List[Dict] = []
    for mf_id, payload in fund_data.items():
        fund_nav = payload["nav"]
        fund_monthly = payload["monthly"]
        fund_sip = payload["sip"]

        # All historical SIPs for this fund have already redeemed by "now"
        # (they were computed from past data), so use the entire series.
        feats = _features_at(fund_nav, bench_nav, fund_sip, bench_sip)
        row = {"mfId": mf_id}
        row.update(feats)
        now_rows.append(row)

    features_now = pd.DataFrame(now_rows).set_index("mfId")
    raw_score = composite_score(features_now, weights, signs)

    # ---- Convert raw -> 0-100 percentile -> apply track-record haircut ----
    pctl = raw_score.rank(pct=True, na_option="keep") * 100.0
    score = pctl.fillna(0.0)

    haircut = pd.Series(1.0, index=score.index)
    for mf_id in score.index:
        days = fund_meta[mf_id]["data_days"]
        if days < MIN_DATA_DAYS_FULL_CONFIDENCE:
            haircut[mf_id] = TRACK_RECORD_HAIRCUT
    score = (score * haircut).round(2)

    # ---- Assemble output ----
    out = features_now.copy()
    out["score"] = score
    out["raw_score"] = raw_score
    out["name"] = [fund_meta[m]["name"] for m in out.index]
    out["aum"] = [fund_meta[m]["aum"] for m in out.index]
    out["data_days"] = [fund_meta[m]["data_days"] for m in out.index]
    out["n_sip_windows"] = [fund_meta[m]["n_sip_windows"] for m in out.index]

    out = out.reset_index()
    out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)

    # ---- Format columns ----
    pct_fmt = lambda v: f"{v * 100:.2f}" if pd.notna(v) else ""
    ratio_fmt = lambda v: f"{v:.3f}" if pd.notna(v) else ""

    output = pd.DataFrame()
    output["mfId"] = out["mfId"]
    output["name"] = out["name"]
    output["rank"] = out["rank"]
    output["score"] = out["score"]
    output["data_days"] = out["data_days"]
    output["cagr_3y"] = out["cagr_3y"].apply(pct_fmt)
    output["cagr_5y"] = out["cagr_5y"].apply(pct_fmt)
    output["sip_1y_median"] = out["sip_1y_median"].apply(pct_fmt)
    output["sip_1y_p20"] = out["sip_1y_p20"].apply(pct_fmt)
    output["sip_alpha_median"] = out["sip_alpha_median"].apply(pct_fmt)
    output["sip_consistency"] = out["sip_consistency"].apply(ratio_fmt)
    output["dd_depth_now"] = out["dd_depth"].apply(pct_fmt)
    output["recovery_slope"] = out["recovery_slope"].apply(pct_fmt)
    output["late_mom_6m"] = out["late_mom_6m"].apply(pct_fmt)
    output["info_ratio_2y"] = out["info_ratio_2y"].apply(ratio_fmt)
    output["up_capture_2y"] = out["up_capture_2y"].apply(ratio_fmt)
    output["down_capture_2y"] = out["down_capture_2y"].apply(ratio_fmt)
    output["nav_trend_residual"] = out["nav_trend_residual"].apply(ratio_fmt)
    output["sortino_3y"] = out["sortino_3y"].apply(ratio_fmt)
    output["aum"] = out["aum"]
    output["n_sip_windows"] = out["n_sip_windows"]

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # ---- Top 15 ----
    print("\n" + "=" * 72)
    print("  TOP 15 SMALL CAP FUNDS - CLAUDE MODEL")
    print("=" * 72 + "\n")
    cols = ["rank", "name", "score", "cagr_3y", "sip_1y_median",
            "sip_alpha_median", "dd_depth_now", "aum"]
    print(output.head(15)[cols].to_string(index=False))

    print(f"\n  Full results ({len(output)} funds) -> {OUTPUT_FILE}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Small Cap MF screener (Claude)")
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=p.parse_args().date)
