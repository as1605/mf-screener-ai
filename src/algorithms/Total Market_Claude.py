#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - Claude

An adaptive multi-horizon conviction model for scoring Indian Total Market
mutual funds (Contra, Flexi Cap, Focused, Multi Cap, Value).

Total Market funds enjoy maximum portfolio flexibility — they can invest across
market caps, sectors, and styles.  The key challenge is identifying which fund
managers consistently convert that freedom into alpha.  This model focuses on
aspects that are uniquely predictive for this diversified category:

- Skill persistence via excess-return autocorrelation and rolling IC
- Path quality through gain-to-pain ratios, tail analysis, and CVaR
- Asymmetric participation in up/down market movements
- Cross-horizon rank consistency (reliable across timeframes)
- Tempered momentum with acceleration detection

Architecture
------------
~33 metrics per fund across seven research-backed categories, normalised to
peer-group percentiles and combined via fixed weights:

  1. Skill & Alpha Quality        25%   Alpha, IR, T-M decomposition,
                                         excess-return autocorrelation, active divergence
  2. Path Quality & Tail Risk     16%   Gain-to-pain, tail ratio, CVaR,
                                         Ulcer PI, Calmar
  3. Drawdown Resilience          14%   Max DD, pain index, current DD,
                                         recovery speed, avg DD duration
  4. Regime Adaptability          14%   Capture spread, transition alpha,
                                         beta asymmetry, bear outperformance, T-M gamma
  5. Consistency & Stability      13%   Rolling beat%, cross-horizon rank
                                         consistency, sortino stability, hit rate
  6. Momentum & Acceleration      13%   Relative momentum, acceleration,
                                         vol-normalised momentum, DD-adjusted momentum
  7. Distribution Quality          5%   Return skewness, excess kurtosis

Sector  : Total Market
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
SECTOR = "Total Market"
SUBSECTORS = [
    "Contra Fund",
    "Flexi Cap Fund",
    "Focused Fund",
    "Multi Cap Fund",
    "Value Fund",
]
BENCHMARK_INDEX = "Total Market"
RISK_FREE_RATE = 0.065
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
    Treynor-Mazuy market timing model separates stock-picking skill
    (alpha) from market timing (gamma).
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


def active_divergence(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Optional[float]:
    """
    Tracking error as a proxy for active management intensity.
    Higher tracking error implies the fund is taking more active bets.
    Combined with positive alpha, this signals genuine skill.
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 20:
        return None
    excess = aligned["fund"] - aligned["bench"]
    return float(excess.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


# ===================================================================
# Novel Metrics: Path Quality & Tail Risk
# ===================================================================

def gain_to_pain_ratio(rets: pd.Series) -> Optional[float]:
    """
    Schwager's Gain-to-Pain Ratio: sum of all returns / abs(sum of
    negative returns). Captures the asymmetry of the return stream.
    Values > 1.0 indicate attractive return profile.
    """
    if len(rets) < 20:
        return None
    total_return = rets.sum()
    total_pain = abs(rets[rets < 0].sum())
    if total_pain <= 1e-12:
        return 10.0
    return float(total_return / total_pain)


def tail_ratio(rets: pd.Series) -> Optional[float]:
    """
    Ratio of the 95th-percentile return to the absolute value of the
    5th-percentile return. Values > 1 mean the right tail is fatter
    than the left — a desirable property for fund selection.
    """
    if len(rets) < 30:
        return None
    p95 = float(np.percentile(rets, 95))
    p5 = float(abs(np.percentile(rets, 5)))
    if p5 <= 1e-12:
        return 5.0
    return float(p95 / p5)


def conditional_value_at_risk(rets: pd.Series, alpha: float = 0.05) -> Optional[float]:
    """
    CVaR (Expected Shortfall) at the given alpha level.
    Measures the expected loss in the worst alpha% of weeks.
    Returned as annualised negative value (more negative = worse).
    """
    if len(rets) < 30:
        return None
    cutoff = np.percentile(rets, alpha * 100)
    tail_losses = rets[rets <= cutoff]
    if len(tail_losses) == 0:
        return 0.0
    return float(tail_losses.mean() * WEEKS_PER_YEAR)


def excess_return_autocorrelation(
    fund_ret: pd.Series, bench_ret: pd.Series, lag: int = 4
) -> Optional[float]:
    """
    Lag-N autocorrelation of weekly excess returns over benchmark.
    Positive autocorrelation implies the fund's alpha-generation process
    has momentum / persistence — a strong predictor of future returns
    as documented in Carhart (1997) and Bollen & Busse (2005).
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 40:
        return None
    excess = aligned["fund"] - aligned["bench"]
    if excess.std() <= 1e-12:
        return None
    return float(excess.autocorr(lag=lag))


# ===================================================================
# Capture Ratios & Regime Analysis
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


def regime_transition_alpha(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    Alpha during regime transition periods — defined as weeks where the
    benchmark's trailing 3M return changes sign vs its trailing 6M return.
    Funds that navigate transitions well (bull→bear, bear→bull) tend to
    outperform in the following year, as they demonstrate adaptive skill.
    """
    if len(fund_nav) < LB_1Y or len(bench_nav) < LB_1Y:
        return None

    bench_3m = bench_nav.pct_change(LB_3M)
    bench_6m = bench_nav.pct_change(LB_6M)

    transition_mask = (
        ((bench_3m > 0) & (bench_6m < 0)) |
        ((bench_3m < 0) & (bench_6m > 0))
    )
    transition_dates = transition_mask[transition_mask].index

    fund_ret = weekly_returns(fund_nav)
    bench_ret = weekly_returns(bench_nav)
    common = fund_ret.index.intersection(bench_ret.index).intersection(transition_dates)

    if len(common) < 12:
        return None

    alpha, _ = compute_alpha_beta(fund_ret.loc[common], bench_ret.loc[common])
    return alpha


def bear_market_outperformance(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    Average weekly excess return during periods where the benchmark
    has negative trailing 6M returns.
    """
    if len(fund_nav) < LB_1Y or len(bench_nav) < LB_1Y:
        return None

    fund_ret = weekly_returns(fund_nav)
    bench_ret = weekly_returns(bench_nav)
    bench_6m = bench_nav.pct_change(LB_6M)
    bear_dates = bench_6m[bench_6m < 0].index

    common = fund_ret.index.intersection(bench_ret.index).intersection(bear_dates)
    if len(common) < 10:
        return None

    excess = fund_ret.loc[common] - bench_ret.loc[common]
    return float(excess.mean() * WEEKS_PER_YEAR)


# ===================================================================
# Drawdown Analysis
# ===================================================================

def max_drawdown_calc(nav: pd.Series) -> float:
    if len(nav) < 10:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def pain_index(nav: pd.Series) -> Optional[float]:
    """
    Average drawdown across the entire observation period.
    Unlike max drawdown which captures only the worst episode, pain index
    reflects the typical underwater experience. Lower (closer to 0) is better.
    """
    if len(nav) < 20:
        return None
    dd = nav / nav.cummax() - 1.0
    return float(dd.mean())


def ulcer_index(nav: pd.Series) -> float:
    """sqrt(mean(drawdown_pct^2)). Captures both depth and duration."""
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
    Average inverse recovery ratio for drawdowns deeper than threshold.
    Higher = faster recovery = better.
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

    if in_drawdown:
        time_in_dd = len(dd) - 1 - dd_start
        time_to_trough = max(dd_trough - dd_start, 1)
        estimated_ratio = max(time_in_dd / time_to_trough, 2.0)
        recovery_ratios.append(estimated_ratio)

    if not recovery_ratios:
        return None

    avg_ratio = float(np.mean(recovery_ratios))
    return float(1.0 / (avg_ratio + 0.1))


def avg_drawdown_duration(nav: pd.Series, threshold: float = -0.03) -> Optional[float]:
    """
    Average number of weeks spent in drawdowns deeper than threshold.
    Shorter durations suggest the fund recovers quickly from losses.
    Returned as inverse so higher = better (shorter drawdowns).
    """
    if len(nav) < LB_1Y:
        return None

    dd = nav / nav.cummax() - 1.0
    in_drawdown = False
    dd_start = 0
    durations: List[int] = []

    for i in range(len(dd)):
        val = float(dd.iloc[i])
        if not in_drawdown and val < threshold:
            in_drawdown = True
            dd_start = i
        elif in_drawdown and val >= 0:
            durations.append(i - dd_start)
            in_drawdown = False

    if not durations:
        return None

    avg_dur = float(np.mean(durations))
    return float(1.0 / (avg_dur + 1.0))


# ===================================================================
# Consistency & Stability Metrics
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


def cross_horizon_rank_consistency(
    fund_nav: pd.Series, bench_nav: pd.Series, peer_navs: Dict[str, pd.Series]
) -> Optional[float]:
    """
    Measure how consistent a fund's peer-relative rank is across multiple
    time horizons (3M, 6M, 1Y, 3Y). A fund that ranks well across all
    horizons demonstrates reliable, non-fluky performance. Returns a
    score in [0, 1] where 1 = perfectly consistent ranking.

    This is a novel metric not commonly found in standard MF analysis.
    It captures the idea that truly skilled funds perform well regardless
    of the measurement window, while lucky streaks show up as inconsistency.
    """
    horizons = [LB_3M, LB_6M, LB_1Y, LB_3Y]
    peer_ids = list(peer_navs.keys())
    if len(peer_ids) < 5:
        return None

    fund_ranks: List[Optional[float]] = []
    for h in horizons:
        fund_ret = annualised_return(fund_nav, h)
        if fund_ret is None:
            fund_ranks.append(None)
            continue

        peer_rets = []
        for pid in peer_ids:
            pr = annualised_return(peer_navs[pid], h)
            if pr is not None:
                peer_rets.append(pr)

        if len(peer_rets) < 5:
            fund_ranks.append(None)
            continue

        n_better = sum(1 for pr in peer_rets if pr > fund_ret)
        percentile = 1.0 - (n_better / len(peer_rets))
        fund_ranks.append(percentile)

    valid_ranks = [r for r in fund_ranks if r is not None]
    if len(valid_ranks) < 3:
        return None

    arr = np.array(valid_ranks)
    mean_rank = float(np.mean(arr))
    std_rank = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    consistency = mean_rank * (1.0 / (1.0 + std_rank * 5.0))
    return float(np.clip(consistency, 0, 1))


def rolling_sortino_stability(
    fund_nav: pd.Series, window: int = LB_1Y, step: int = ROLLING_STEP
) -> Optional[float]:
    """
    Rolling Sortino ratio analysis. Returns a stability score that
    combines mean Sortino level with low variance across windows.
    Funds with consistently high Sortino across time are preferred.
    """
    rets = weekly_returns(fund_nav)
    if len(rets) < window + 10:
        return None

    sortinos: List[float] = []
    for start in range(0, len(rets) - window, step):
        sub = rets.iloc[start:start + window]
        dd = downside_deviation(sub)
        nav_window = fund_nav.iloc[start:start + window + 1]
        cagr = annualised_return(nav_window, window)
        if cagr is not None:
            s = sortino_ratio_calc(cagr, dd)
            if s is not None:
                sortinos.append(s)

    if len(sortinos) < 3:
        return None

    mean_s = float(np.mean(sortinos))
    std_s = float(np.std(sortinos, ddof=1))

    return float(0.5 * np.clip(mean_s, -1, 3) + 0.5 / (1.0 + std_s))


def excess_return_hit_rate(
    fund_ret: pd.Series, bench_ret: pd.Series
) -> Optional[float]:
    """
    Fraction of weeks where fund return exceeds benchmark return.
    A hit rate significantly above 50% indicates consistent alpha.
    """
    aligned = pd.concat(
        [fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1
    ).dropna()
    if len(aligned) < 20:
        return None
    return float((aligned["fund"] > aligned["bench"]).mean())


# ===================================================================
# Momentum & Acceleration
# ===================================================================

def relative_momentum(
    fund_nav: pd.Series, bench_nav: pd.Series, weeks: int
) -> Optional[float]:
    f = annualised_return(fund_nav, weeks)
    b = annualised_return(bench_nav, weeks)
    if f is None or b is None:
        return None
    return float(f - b)


def vol_normalised_momentum(
    fund_nav: pd.Series, bench_nav: pd.Series, weeks: int = LB_6M
) -> Optional[float]:
    """
    Relative momentum normalised by the fund's recent volatility.
    This penalises momentum that comes from high-vol bets (which are
    less persistent) and rewards momentum earned with lower risk.
    """
    rel_mom = relative_momentum(fund_nav, bench_nav, weeks)
    if rel_mom is None:
        return None

    rets = weekly_returns(fund_nav)
    recent_rets = rets.tail(weeks)
    if len(recent_rets) < 10:
        return None

    vol = annualised_volatility(recent_rets)
    if pd.isna(vol) or vol <= 1e-12:
        return None

    return float(rel_mom / vol)


def momentum_acceleration(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    Change in relative momentum: compares 3M relative momentum to 6M
    relative momentum. Positive acceleration means the fund is gaining
    strength relative to benchmark — a forward-looking signal.
    """
    mom_3m = relative_momentum(fund_nav, bench_nav, LB_3M)
    mom_6m = relative_momentum(fund_nav, bench_nav, LB_6M)
    if mom_3m is None or mom_6m is None:
        return None
    return float(mom_3m - mom_6m)


# ===================================================================
# Novel Metrics: Current State & Path-Adjusted Momentum
# ===================================================================

def current_drawdown_depth(nav: pd.Series) -> Optional[float]:
    """
    Current distance from all-time high NAV.  Returns 0 when the fund
    is at its peak, and negative values when below.  Less negative
    (closer to 0) is better — the fund is near its highs.

    This captures real-time fund health that purely historical metrics
    miss: a fund sitting at -20% from peak right now faces different
    forward dynamics than one at ATH, regardless of historical stats.
    """
    if len(nav) < 20:
        return None
    return float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1.0)


def drawdown_adjusted_momentum(
    fund_nav: pd.Series, bench_nav: pd.Series
) -> Optional[float]:
    """
    Relative momentum adjusted for path quality over the measurement
    period.  Rewards momentum achieved through smooth, low-drawdown
    paths and penalises momentum earned via volatile recovery-crash
    cycles.

    The intuition: 10% relative outperformance earned via a smooth
    uptrend is far more likely to persist than 10% earned through a
    sharp V-recovery.  The path quality multiplier (based on the
    average drawdown during the momentum window) provides this
    differentiation.
    """
    mom = relative_momentum(fund_nav, bench_nav, LB_6M)
    if mom is None:
        return None

    recent_nav = fund_nav.iloc[-(LB_6M + 1):]
    if len(recent_nav) < 20:
        return None

    dd = recent_nav / recent_nav.cummax() - 1.0
    avg_pain = abs(float(dd.mean()))
    path_quality = 1.0 / (1.0 + avg_pain * 15)

    return float(mom * path_quality)


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
    gp = gain_to_pain_ratio(rets)
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
    add(gp, 1.0)
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
    Score funds using trailing data, then measure actual forward 1Y
    excess returns. Reports Spearman rank correlation and top-5 portfolio
    performance per period.
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
    # --- Skill & Alpha Quality (25%) ---
    "alpha":                    (True,  0.07),
    "info_ratio":               (True,  0.05),
    "tm_alpha":                 (True,  0.04),
    "excess_autocorr":          (True,  0.04),
    "active_divergence_score":  (True,  0.05),

    # --- Path Quality & Tail Risk (16%) ---
    "gain_to_pain":             (True,  0.05),
    "tail_ratio":               (True,  0.03),
    "cvar":                     (True,  0.03),
    "ulcer_perf_index":         (True,  0.03),
    "calmar":                   (True,  0.02),

    # --- Drawdown Resilience (14%) ---
    # NOTE: max_drawdown & pain_index are NEGATIVE values (less negative = better)
    # so higher_is_better=True is correct for them.
    "max_drawdown":             (True,  0.04),
    "pain_index":               (True,  0.03),
    "recovery_speed":           (True,  0.03),
    "avg_dd_duration":          (True,  0.02),
    "current_drawdown":         (True,  0.02),

    # --- Regime Adaptability (14%) ---
    "capture_spread":           (True,  0.03),
    "transition_alpha":         (True,  0.03),
    "beta_asymmetry":           (True,  0.02),
    "bear_outperformance":      (True,  0.02),
    "tm_gamma":                 (True,  0.04),

    # --- Consistency & Stability (13%) ---
    "rolling_1y_beat_pct":      (True,  0.04),
    "cross_horizon_consistency":(True,  0.03),
    "sortino_stability":        (True,  0.03),
    "hit_rate":                 (True,  0.03),

    # --- Momentum & Acceleration (13%) ---
    "momentum_6m":              (True,  0.04),
    "vol_norm_momentum":        (True,  0.04),
    "acceleration":             (True,  0.03),
    "dd_adj_momentum":          (True,  0.02),

    # --- Distribution Quality (5%) ---
    "skewness":                 (True,  0.03),
    "kurtosis":                 (False, 0.02),
}


def analyse_fund(
    mf_id: str,
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    name: str,
    aum: float,
    subsector: str,
    peer_navs: Dict[str, pd.Series],
) -> dict:
    """Compute all metrics for a single fund."""

    n_weeks = len(fund_nav)
    result: dict = {
        "mfId": mf_id,
        "name": name,
        "aum": round(aum, 2),
        "subsector": subsector,
    }

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

    primary_cagr = (
        result["cagr_3y"] if result["cagr_3y"] is not None
        else result["cagr_5y"] if result["cagr_5y"] is not None
        else result["cagr_1y"]
    )

    # ---- Returns & Volatility ----
    rets = weekly_returns(fund_nav)
    if len(rets) < 20:
        logger.warning(f"{mf_id}: insufficient data ({len(rets)} weeks)")
        return result

    vol = annualised_volatility(rets)
    dd = downside_deviation(rets)
    result["volatility"] = vol
    result["downside_dev"] = dd

    # ---- Standard Ratios ----
    result["sharpe"] = sharpe_ratio(primary_cagr, vol)
    result["sortino"] = sortino_ratio_calc(primary_cagr, dd)
    result["calmar"] = calmar_ratio_calc(primary_cagr, max_drawdown_calc(fund_nav))

    # ---- Benchmark-Relative ----
    bench_rets = weekly_returns(bench_nav)
    alpha, beta = compute_alpha_beta(rets, bench_rets)
    result["alpha"] = alpha
    result["beta"] = beta
    result["info_ratio"] = information_ratio(rets, bench_rets)

    tm_a, tm_g = treynor_mazuy_decomposition(rets, bench_rets)
    result["tm_alpha"] = tm_a
    result["tm_gamma"] = tm_g

    result["active_divergence"] = active_divergence(rets, bench_rets)
    ad = result["active_divergence"]
    if ad is not None and alpha is not None:
        result["active_divergence_score"] = float(
            np.clip(alpha, -0.2, 0.2) * np.clip(ad, 0, 0.3)
        )
    else:
        result["active_divergence_score"] = None

    # ---- Novel: Path Quality & Tail Risk ----
    result["gain_to_pain"] = gain_to_pain_ratio(rets)
    result["tail_ratio"] = tail_ratio(rets)
    result["cvar"] = conditional_value_at_risk(rets)

    mdd = max_drawdown_calc(fund_nav)
    result["max_drawdown"] = mdd
    ui = ulcer_index(fund_nav)
    result["ulcer_index"] = ui
    result["ulcer_perf_index"] = ulcer_performance_index(primary_cagr, ui)

    # ---- Novel: Excess Return Autocorrelation ----
    result["excess_autocorr"] = excess_return_autocorrelation(rets, bench_rets)

    # ---- Drawdown Resilience ----
    result["pain_index"] = pain_index(fund_nav)
    result["recovery_speed"] = recovery_speed_score(fund_nav)
    result["avg_dd_duration"] = avg_drawdown_duration(fund_nav)

    # ---- Regime Adaptability ----
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

    result["transition_alpha"] = regime_transition_alpha(fund_nav, bench_nav)
    result["bear_outperformance"] = bear_market_outperformance(fund_nav, bench_nav)

    # ---- Consistency & Stability ----
    result["rolling_1y_beat_pct"] = rolling_benchmark_beat_pct(fund_nav, bench_nav)
    result["cross_horizon_consistency"] = cross_horizon_rank_consistency(
        fund_nav, bench_nav, peer_navs
    )
    result["sortino_stability"] = rolling_sortino_stability(fund_nav)
    result["hit_rate"] = excess_return_hit_rate(rets, bench_rets)

    # ---- Momentum & Acceleration ----
    result["momentum_6m"] = relative_momentum(fund_nav, bench_nav, LB_6M)
    result["momentum_12m"] = relative_momentum(fund_nav, bench_nav, LB_1Y)
    result["vol_norm_momentum"] = vol_normalised_momentum(fund_nav, bench_nav)
    result["acceleration"] = momentum_acceleration(fund_nav, bench_nav)
    result["dd_adj_momentum"] = drawdown_adjusted_momentum(fund_nav, bench_nav)

    # ---- Current State ----
    result["current_drawdown"] = current_drawdown_depth(fund_nav)

    # ---- Distribution Quality ----
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

    def _confidence(days: int) -> float:
        if days < 365:
            return 0.55
        if days < 2 * 365:
            return 0.72
        if days < 3 * 365:
            return 0.85
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
    print("\n" + "=" * 84)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM — CLAUDE")
    print(f"  Benchmark : Nifty 500 ({BENCHMARK_INDEX})")
    print("  Subsectors: " + ", ".join(SUBSECTORS))
    print("  Model     : Adaptive multi-horizon conviction + path quality")
    print("=" * 84)

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
    sector_df = df_all[df_all["subsector"].isin(SUBSECTORS)].copy()
    print(f"  Total Market funds: {len(sector_df)}")
    for sub in SUBSECTORS:
        n = (sector_df["subsector"] == sub).sum()
        print(f"    {sub:20s}: {n}")

    # --- Pre-load all fund NAVs for cross-horizon consistency ---
    logger.info("Loading all fund NAV data...")
    fund_navs: Dict[str, pd.Series] = {}
    fund_meta: Dict[str, dict] = {}

    for _, row in sector_df.iterrows():
        mf_id = row["mfId"]
        name = row["name"]
        aum = float(row.get("aum", 0) or 0)
        subsector = row["subsector"]

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

            fund_navs[mf_id] = fund_nav
            fund_meta[mf_id] = {
                "name": name,
                "aum": aum,
                "subsector": subsector,
            }
        except Exception as e:
            logger.error(f"Error loading {mf_id} ({name}): {e}")

    logger.info(f"Loaded NAV data for {len(fund_navs)} funds")

    # --- Analyse each fund ---
    logger.info("Analysing individual funds...")
    results: List[dict] = []
    aligned_navs: Dict[str, pd.Series] = {}

    for mf_id, fund_nav in fund_navs.items():
        meta = fund_meta[mf_id]
        try:
            peer_navs = {k: v for k, v in fund_navs.items() if k != mf_id}

            metrics = analyse_fund(
                mf_id=mf_id,
                fund_nav=fund_nav,
                bench_nav=bench_nav,
                name=meta["name"],
                aum=meta["aum"],
                subsector=meta["subsector"],
                peer_navs=peer_navs,
            )
            results.append(metrics)
            aligned_navs[mf_id] = fund_nav.reindex(bench_nav.index).ffill()

        except Exception as e:
            logger.error(f"Error analysing {mf_id} ({meta['name']}): {e}")

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
    output["subsector"] = df_scored["subsector"]
    output["cagr_1y"] = df_scored["cagr_1y"].apply(_pct)
    output["cagr_3y"] = df_scored["cagr_3y"].apply(_pct)
    output["cagr_5y"] = df_scored["cagr_5y"].apply(_pct)
    output["volatility"] = df_scored["volatility"].apply(_pct)
    output["sharpe"] = df_scored["sharpe"].apply(_ratio)
    output["sortino"] = df_scored["sortino"].apply(_ratio)
    output["alpha"] = df_scored["alpha"].apply(_pct)
    output["beta"] = df_scored["beta"].apply(_ratio)
    output["info_ratio"] = df_scored["info_ratio"].apply(_ratio)
    output["tm_alpha"] = df_scored["tm_alpha"].apply(_pct)
    output["tm_gamma"] = df_scored["tm_gamma"].apply(_ratio)
    output["excess_autocorr"] = df_scored["excess_autocorr"].apply(_ratio)
    output["active_divergence"] = df_scored["active_divergence"].apply(_pct)
    output["gain_to_pain"] = df_scored["gain_to_pain"].apply(_num)
    output["tail_ratio"] = df_scored["tail_ratio"].apply(_num)
    output["cvar"] = df_scored["cvar"].apply(_pct)
    output["max_drawdown"] = df_scored["max_drawdown"].apply(_pct)
    output["pain_index"] = df_scored["pain_index"].apply(_pct)
    output["ulcer_index"] = df_scored["ulcer_index"].apply(_num)
    output["ulcer_perf_index"] = df_scored["ulcer_perf_index"].apply(_num)
    output["recovery_speed"] = df_scored["recovery_speed"].apply(_num)
    output["avg_dd_duration"] = df_scored["avg_dd_duration"].apply(_num)
    output["up_capture"] = df_scored["up_capture"].apply(_ratio)
    output["down_capture"] = df_scored["down_capture"].apply(_ratio)
    output["capture_spread"] = df_scored["capture_spread"].apply(_ratio)
    output["beta_asymmetry"] = df_scored["beta_asymmetry"].apply(_ratio)
    output["transition_alpha"] = df_scored["transition_alpha"].apply(_pct)
    output["bear_outperformance"] = df_scored["bear_outperformance"].apply(_pct)
    output["rolling_1y_beat_pct"] = df_scored["rolling_1y_beat_pct"].apply(_pct)
    output["cross_horizon_consistency"] = df_scored["cross_horizon_consistency"].apply(_num)
    output["sortino_stability"] = df_scored["sortino_stability"].apply(_num)
    output["hit_rate"] = df_scored["hit_rate"].apply(_pct)
    output["momentum_6m"] = df_scored["momentum_6m"].apply(_pct)
    output["momentum_12m"] = df_scored["momentum_12m"].apply(_pct)
    output["vol_norm_momentum"] = df_scored["vol_norm_momentum"].apply(_ratio)
    output["acceleration"] = df_scored["acceleration"].apply(_pct)
    output["dd_adj_momentum"] = df_scored["dd_adj_momentum"].apply(_ratio)
    output["current_drawdown"] = df_scored["current_drawdown"].apply(_pct)
    output["skewness"] = df_scored["skewness"].apply(_ratio)
    output["kurtosis"] = df_scored["kurtosis"].apply(_ratio)
    output["aum"] = df_scored["aum"]
    output["data_weeks"] = df_scored["data_weeks"]
    output["confidence"] = df_scored["confidence"].apply(_ratio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # --- Console summary ---
    print("\n" + "=" * 84)
    print("  SCORING MODEL SUMMARY")
    print("=" * 84)

    categories = {
        "Skill & Alpha Quality": [
            "alpha", "info_ratio", "tm_alpha", "excess_autocorr",
            "active_divergence_score",
        ],
        "Path Quality & Tail Risk": [
            "gain_to_pain", "tail_ratio", "cvar", "ulcer_perf_index", "calmar",
        ],
        "Drawdown Resilience": [
            "max_drawdown", "pain_index", "recovery_speed",
            "avg_dd_duration", "current_drawdown",
        ],
        "Regime Adaptability": [
            "capture_spread", "transition_alpha", "beta_asymmetry",
            "bear_outperformance", "tm_gamma",
        ],
        "Consistency & Stability": [
            "rolling_1y_beat_pct", "cross_horizon_consistency",
            "sortino_stability", "hit_rate",
        ],
        "Momentum & Acceleration": [
            "momentum_6m", "vol_norm_momentum", "acceleration",
            "dd_adj_momentum",
        ],
        "Distribution Quality": [
            "skewness", "kurtosis",
        ],
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

    print("\n" + "=" * 84)
    print("  TOP 20 TOTAL MARKET FUNDS BY COMPOSITE SCORE")
    print("=" * 84 + "\n")

    display_cols = [
        "rank", "name", "score", "subsector", "cagr_5y", "alpha",
        "gain_to_pain", "max_drawdown", "current_drawdown",
        "rolling_1y_beat_pct", "capture_spread", "aum",
    ]
    display_cols = [c for c in display_cols if c in output.columns]
    print(output.head(20)[display_cols].to_string(index=False))

    print(f"\n  Full results ({len(output)} funds) → {OUTPUT_FILE}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
