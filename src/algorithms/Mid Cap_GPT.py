#!/usr/bin/env python3
"""
Mid Cap Mutual Fund Scoring Algorithm - GPT
===========================================

Rank Indian Mid Cap mutual funds for the next 1-year monthly SIP outcome using
only NAV history, benchmark NAV history, and fund metadata.

2027-2028 market prior
----------------------
The mid-cap opportunity looks more earnings-led than rerating-led: rate support,
domestic consumption, manufacturing/capex, financials, autos, e-commerce, and
healthcare can help FY27/FY28 profit growth, while valuations and liquidity keep
the downside asymmetric. The NAV translation is therefore:

- favor funds with persistent rolling SIP alpha versus Nifty Midcap 150,
- reward recovery participation after mid-cap corrections,
- prefer downside capture discipline over raw high beta,
- penalize parabolic momentum when it is not backed by rolling alpha,
- shrink short-history or low-confidence signals.

Feature weights are learned from historical, look-ahead-safe 12-month SIP
windows and then shrunk toward the prior above to avoid overfitting.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


SECTOR = "Mid Cap"
SUBSECTOR = "Mid Cap Fund"
BENCHMARK_INDEX = "Mid Cap"

SIP_MONTHS = 12
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 365.25
RISK_FREE_RATE = 0.065
WEEKLY_RF = (1.0 + RISK_FREE_RATE) ** (1.0 / WEEKS_PER_YEAR) - 1.0

MIN_HISTORY_WEEKS = 52
MIN_TRAINING_MONTHS = 30
MIN_TRAINING_OBS = 60
MIN_CORR = 0.025
MAX_FEATURE_WEIGHT = 0.16

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_GPT.csv"


FEATURES = [
    "sip_xirr_median",
    "sip_xirr_p25",
    "sip_alpha_hit_rate",
    "alpha_consistency",
    "information_ratio",
    "up_capture",
    "down_capture",
    "downside_beta",
    "drawdown_control",
    "recovery_strength",
    "correction_defense",
    "momentum_quality",
    "omega_ratio",
    "ulcer_index",
    "cdar_5",
    "style_purity",
    "aum_score",
]

# 1 means higher raw values are better; -1 means lower raw values are better.
FEATURE_DIRECTIONS = pd.Series(
    {
        "sip_xirr_median": 1,
        "sip_xirr_p25": 1,
        "sip_alpha_hit_rate": 1,
        "alpha_consistency": 1,
        "information_ratio": 1,
        "up_capture": 1,
        "down_capture": -1,
        "downside_beta": -1,
        "drawdown_control": 1,
        "recovery_strength": 1,
        "correction_defense": 1,
        "momentum_quality": 1,
        "omega_ratio": 1,
        "ulcer_index": -1,
        "cdar_5": 1,
        "style_purity": 1,
        "aum_score": 1,
    },
    dtype=float,
)

PRIOR_WEIGHTS = pd.Series(
    {
        "sip_xirr_median": 0.085,
        "sip_xirr_p25": 0.100,
        "sip_alpha_hit_rate": 0.090,
        "alpha_consistency": 0.075,
        "information_ratio": 0.060,
        "up_capture": 0.050,
        "down_capture": 0.065,
        "downside_beta": 0.060,
        "drawdown_control": 0.075,
        "recovery_strength": 0.085,
        "correction_defense": 0.080,
        "momentum_quality": 0.060,
        "omega_ratio": 0.040,
        "ulcer_index": 0.045,
        "cdar_5": 0.040,
        "style_purity": 0.050,
        "aum_score": 0.040,
    },
    dtype=float,
)
PRIOR_WEIGHTS = PRIOR_WEIGHTS / PRIOR_WEIGHTS.sum()


def clean_nav_chart(df: pd.DataFrame) -> pd.Series:
    """Return a sorted positive NAV series indexed by timezone-naive timestamp."""
    if df is None or df.empty or "timestamp" not in df or "nav" not in df:
        return pd.Series(dtype=float)

    out = df[["timestamp", "nav"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["timestamp", "nav"])
    out = out[out["nav"] > 0]
    if out.empty:
        return pd.Series(dtype=float)

    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    series = out.set_index("timestamp")["nav"].astype(float)
    series.name = "nav"
    return series


def to_daily(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.sort_index().resample("D").ffill().dropna()


def to_weekly(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.sort_index().resample("W-FRI").last().dropna()


def to_month_start(nav: pd.Series) -> pd.Series:
    daily = to_daily(nav)
    if daily.empty:
        return daily
    return daily.resample("MS").first().dropna()


def slice_to(nav: pd.Series, end_date: Optional[pd.Timestamp]) -> pd.Series:
    if end_date is None or nav.empty:
        return nav
    return nav[nav.index <= pd.Timestamp(end_date)]


def finite_or_nan(value: float) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def simple_return(nav: pd.Series, periods: int) -> float:
    if len(nav) <= periods:
        return np.nan
    start = nav.iloc[-periods - 1]
    end = nav.iloc[-1]
    if start <= 0 or end <= 0:
        return np.nan
    return float(end / start - 1.0)


def calendar_cagr(nav: pd.Series, years: float) -> float:
    if nav.empty:
        return np.nan
    daily = to_daily(nav)
    if daily.empty:
        return np.nan
    cutoff = daily.index[-1] - pd.Timedelta(days=int(years * DAYS_PER_YEAR))
    window = daily[daily.index >= cutoff]
    if len(window) < 2:
        return np.nan
    elapsed = (window.index[-1] - window.index[0]).days / DAYS_PER_YEAR
    if elapsed <= 0 or elapsed < years * 0.82 or window.iloc[0] <= 0:
        return np.nan
    return float((window.iloc[-1] / window.iloc[0]) ** (1.0 / elapsed) - 1.0)


def weekly_returns(nav: pd.Series) -> pd.Series:
    weekly = to_weekly(nav)
    if weekly.empty:
        return pd.Series(dtype=float)
    return weekly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def align_returns(*navs: pd.Series) -> pd.DataFrame:
    frames = []
    for i, nav in enumerate(navs):
        ret = weekly_returns(nav)
        if ret.empty:
            return pd.DataFrame()
        frames.append(ret.rename(f"r{i}"))
    return pd.concat(frames, axis=1, join="inner").dropna()


def annualized_excess_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns.mean()) ** WEEKS_PER_YEAR - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    daily = to_daily(nav)
    if daily.empty:
        return np.nan
    drawdown = daily / daily.cummax() - 1.0
    return float(drawdown.min())


def current_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    daily = to_daily(nav)
    if daily.empty:
        return np.nan
    return float(daily.iloc[-1] / daily.cummax().iloc[-1] - 1.0)


def ulcer_index(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    daily = to_daily(nav)
    if daily.empty:
        return np.nan
    drawdown_pct = (daily / daily.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(np.minimum(drawdown_pct, 0.0)))))


def conditional_drawdown_at_risk(nav: pd.Series, q: float = 0.05) -> float:
    if nav.empty:
        return np.nan
    daily = to_daily(nav)
    if daily.empty:
        return np.nan
    drawdown = daily / daily.cummax() - 1.0
    n_tail = max(1, int(np.ceil(len(drawdown) * q)))
    return float(drawdown.nsmallest(n_tail).mean())


def sortino_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    downside = returns[returns < WEEKLY_RF] - WEEKLY_RF
    if downside.empty:
        return np.nan
    downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(WEEKS_PER_YEAR))
    if downside_dev <= 1e-12:
        return np.nan
    excess = (1.0 + returns.mean()) ** WEEKS_PER_YEAR - 1.0 - RISK_FREE_RATE
    return float(excess / downside_dev)


def omega_ratio(returns: pd.Series, hurdle: float = WEEKLY_RF) -> float:
    if returns.empty:
        return np.nan
    excess = returns - hurdle
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses <= 1e-12:
        return np.nan
    return float(gains / losses)


def monthly_irr(cashflows: Iterable[float]) -> float:
    flows = [float(x) for x in cashflows]
    if not flows or max(flows) <= 0 or min(flows) >= 0:
        return np.nan

    def npv(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** i) for i, cf in enumerate(flows))

    low = -0.95
    high = 5.0
    low_npv = npv(low)
    high_npv = npv(high)
    if not np.isfinite(low_npv) or not np.isfinite(high_npv) or low_npv * high_npv > 0:
        invested = -sum(cf for cf in flows if cf < 0)
        final = sum(cf for cf in flows if cf > 0)
        if invested <= 0:
            return np.nan
        return float((final / invested) ** (12.0 / max(1, len(flows) - 1)) - 1.0)

    for _ in range(80):
        mid = (low + high) / 2.0
        mid_npv = npv(mid)
        if abs(mid_npv) < 1e-7:
            break
        if low_npv * mid_npv <= 0:
            high = mid
            high_npv = mid_npv
        else:
            low = mid
            low_npv = mid_npv

    monthly = (low + high) / 2.0
    if monthly <= -1.0:
        return np.nan
    return float((1.0 + monthly) ** 12.0 - 1.0)


def sip_xirr_at(monthly_nav: pd.Series, start_pos: int, months: int = SIP_MONTHS) -> float:
    if monthly_nav.empty or start_pos < 0 or start_pos + months >= len(monthly_nav):
        return np.nan
    buys = monthly_nav.iloc[start_pos : start_pos + months]
    exit_nav = monthly_nav.iloc[start_pos + months]
    if buys.empty or (buys <= 0).any() or exit_nav <= 0:
        return np.nan
    units = float((1.0 / buys).sum())
    final_value = units * float(exit_nav)
    return monthly_irr([-1.0] * months + [final_value])


def rolling_sip_xirrs(nav: pd.Series, months: int = SIP_MONTHS) -> pd.Series:
    monthly = to_month_start(nav)
    if len(monthly) <= months:
        return pd.Series(dtype=float)
    values: Dict[pd.Timestamp, float] = {}
    for pos in range(0, len(monthly) - months):
        values[monthly.index[pos]] = sip_xirr_at(monthly, pos, months)
    out = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    out.name = "sip_xirr"
    return out


def forward_sip_xirr(nav: pd.Series, start_date: pd.Timestamp, months: int = SIP_MONTHS) -> float:
    monthly = to_month_start(nav)
    if len(monthly) <= months:
        return np.nan
    start_date = pd.Timestamp(start_date)
    positions = np.flatnonzero(monthly.index >= start_date)
    if len(positions) == 0:
        return np.nan
    return sip_xirr_at(monthly, int(positions[0]), months)


def capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    joined = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1, join="inner").dropna()
    if len(joined) < 12:
        return np.nan, np.nan

    up = joined[joined["bench"] > 0]
    down = joined[joined["bench"] < 0]

    up_capture = np.nan
    down_capture = np.nan
    if len(up) >= 4 and abs(up["bench"].sum()) > 1e-12:
        up_capture = float(up["fund"].sum() / up["bench"].sum())
    if len(down) >= 4 and abs(down["bench"].sum()) > 1e-12:
        down_capture = float(down["fund"].sum() / down["bench"].sum())
    return up_capture, down_capture


def downside_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    joined = pd.concat([fund_ret.rename("fund"), bench_ret.rename("bench")], axis=1, join="inner").dropna()
    joined = joined[joined["bench"] < 0]
    if len(joined) < 8 or joined["bench"].var() <= 1e-12:
        return np.nan
    return float(joined["fund"].cov(joined["bench"]) / joined["bench"].var())


def information_ratio(fund_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(active) < 12:
        return np.nan
    diff = active.iloc[:, 0] - active.iloc[:, 1]
    tracking = diff.std()
    if tracking <= 1e-12 or not np.isfinite(tracking):
        return np.nan
    return float(diff.mean() / tracking * np.sqrt(WEEKS_PER_YEAR))


def rolling_alpha_consistency(fund_ret: pd.Series, bench_ret: pd.Series, window: int = 26) -> float:
    active = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(active) < window + 4:
        return np.nan
    diff = active.iloc[:, 0] - active.iloc[:, 1]
    rolling = diff.rolling(window).sum().dropna()
    if rolling.empty:
        return np.nan
    return float((rolling > 0).mean())


def correction_defense_and_recovery(fund_nav: pd.Series, bench_nav: pd.Series) -> Tuple[float, float]:
    aligned = align_returns(fund_nav, bench_nav)
    if len(aligned) < 30:
        return np.nan, np.nan
    fund = aligned["r0"]
    bench = aligned["r1"]
    bench_weekly_nav = to_weekly(bench_nav).reindex(aligned.index).ffill().dropna()
    bench_dd = bench_weekly_nav / bench_weekly_nav.cummax() - 1.0
    bench_dd = bench_dd.reindex(aligned.index).ffill()

    stress_mask = (bench < -0.015) | ((bench_dd < -0.06) & (bench < 0))
    if stress_mask.sum() >= 4:
        defense = annualized_excess_return(fund[stress_mask] - bench[stress_mask])
    else:
        defense = np.nan

    prior_4w = bench.rolling(4).sum().shift(1)
    rebound_mask = ((prior_4w < -0.035) | (bench_dd.shift(1) < -0.07)) & (bench > 0.006)
    if rebound_mask.sum() >= 4:
        recovery = annualized_excess_return(fund[rebound_mask] - bench[rebound_mask])
    else:
        recovery = np.nan
    return defense, recovery


def momentum_quality(fund_nav: pd.Series, bench_nav: pd.Series) -> float:
    fund_w = to_weekly(fund_nav)
    bench_w = to_weekly(bench_nav)
    joined = pd.concat([fund_w.rename("fund"), bench_w.rename("bench")], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return np.nan

    fund = joined["fund"]
    bench = joined["bench"]
    rel_13 = simple_return(fund, 13) - simple_return(bench, 13)
    rel_26 = simple_return(fund, 26) - simple_return(bench, 26)
    rel_52 = simple_return(fund, 52) - simple_return(bench, 52)
    rel_4 = simple_return(fund, 4) - simple_return(bench, 4)
    rel_terms = [x for x in [rel_13, rel_26, rel_52] if np.isfinite(x)]
    if not rel_terms:
        return np.nan

    fund_ret = fund.pct_change().dropna()
    vol_26 = float(fund_ret.tail(26).std() * np.sqrt(WEEKS_PER_YEAR)) if len(fund_ret) >= 26 else np.nan
    ma_40 = fund.rolling(40).mean().iloc[-1] if len(fund) >= 40 else np.nan
    overextension = max(0.0, float(fund.iloc[-1] / ma_40 - 1.12)) if np.isfinite(ma_40) and ma_40 > 0 else 0.0
    short_reversal = max(0.0, rel_4) if np.isfinite(rel_4) else 0.0

    score = 0.25 * finite_or_nan(rel_13) + 0.45 * finite_or_nan(rel_26) + 0.30 * finite_or_nan(rel_52)
    if np.isfinite(vol_26):
        score = score / max(0.08, vol_26)
    score -= 0.65 * overextension
    score -= 0.20 * short_reversal
    return finite_or_nan(score)


def style_purity_score(fund_nav: pd.Series, index_navs: Dict[str, pd.Series]) -> float:
    mid = index_navs.get("Mid Cap", pd.Series(dtype=float))
    large = index_navs.get("Large Cap", pd.Series(dtype=float))
    small = index_navs.get("Small Cap", pd.Series(dtype=float))
    if mid.empty or large.empty or small.empty:
        return np.nan

    joined = align_returns(fund_nav, large, mid, small).tail(156)
    if len(joined) < 52:
        return np.nan

    y = joined["r0"].to_numpy(dtype=float)
    x = joined[["r1", "r2", "r3"]].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0][1:]
    except np.linalg.LinAlgError:
        return np.nan

    large_beta, mid_beta, small_beta = beta
    gross = abs(large_beta) + abs(mid_beta) + abs(small_beta)
    if gross <= 1e-12:
        return np.nan

    mid_share = mid_beta / gross
    small_leakage = max(0.0, small_beta - 0.35)
    excessive_beta = max(0.0, gross - 1.35)
    score = mid_share - 0.55 * small_leakage - 0.25 * excessive_beta
    return finite_or_nan(score)


def aum_quality_score(aum: float) -> float:
    aum = finite_or_nan(aum)
    if not np.isfinite(aum) or aum <= 0:
        return 0.55

    log_aum = np.log(aum)
    small_conf = 1.0 / (1.0 + np.exp(-(log_aum - np.log(700.0)) * 2.0))
    capacity_drag = 1.0 / (1.0 + np.exp(-(log_aum - np.log(65000.0)) * 3.0))
    sweet_spot = np.exp(-0.5 * ((log_aum - np.log(12000.0)) / 1.45) ** 2)
    score = 0.55 + 0.35 * small_conf + 0.20 * sweet_spot - 0.18 * capacity_drag
    return float(np.clip(score, 0.35, 1.0))


def feature_snapshot(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    index_navs: Dict[str, pd.Series],
    aum_score: float,
    end_date: Optional[pd.Timestamp] = None,
    compute_style: bool = True,
) -> Dict[str, float]:
    fund = slice_to(fund_nav, end_date)
    bench = slice_to(bench_nav, end_date)
    sliced_indices = {name: slice_to(nav, end_date) for name, nav in index_navs.items()}

    fund_w = to_weekly(fund)
    bench_w = to_weekly(bench)
    if len(fund_w) < MIN_HISTORY_WEEKS or len(bench_w) < MIN_HISTORY_WEEKS:
        return {}

    aligned = align_returns(fund, bench)
    if len(aligned) < 26:
        return {}
    fund_ret = aligned["r0"]
    bench_ret = aligned["r1"]

    fund_sip = rolling_sip_xirrs(fund)
    bench_sip = rolling_sip_xirrs(bench)
    sip_joined = pd.concat([fund_sip.rename("fund"), bench_sip.rename("bench")], axis=1, join="inner").dropna()
    sip_alpha = sip_joined["fund"] - sip_joined["bench"] if not sip_joined.empty else pd.Series(dtype=float)

    lookback_nav = fund[fund.index >= fund.index[-1] - pd.Timedelta(days=int(3 * DAYS_PER_YEAR))]
    if len(to_weekly(lookback_nav)) < 40:
        lookback_nav = fund

    up_cap, down_cap = capture_ratios(fund_ret, bench_ret)
    defense, recovery = correction_defense_and_recovery(fund, bench)
    mdd = max_drawdown(lookback_nav)
    cdar = conditional_drawdown_at_risk(lookback_nav, q=0.05)

    result = {
        "sip_xirr_median": float(sip_joined["fund"].median()) if len(sip_joined) else np.nan,
        "sip_xirr_p25": float(sip_joined["fund"].quantile(0.25)) if len(sip_joined) else np.nan,
        "sip_alpha_hit_rate": float((sip_alpha > 0).mean()) if len(sip_alpha) else np.nan,
        "alpha_consistency": rolling_alpha_consistency(fund_ret, bench_ret),
        "information_ratio": information_ratio(fund_ret, bench_ret),
        "up_capture": up_cap,
        "down_capture": down_cap,
        "downside_beta": downside_beta(fund_ret, bench_ret),
        "drawdown_control": 1.0 + mdd if np.isfinite(mdd) else np.nan,
        "recovery_strength": recovery,
        "correction_defense": defense,
        "momentum_quality": momentum_quality(fund, bench),
        "omega_ratio": omega_ratio(fund_ret),
        "ulcer_index": ulcer_index(lookback_nav),
        "cdar_5": cdar,
        "style_purity": style_purity_score(fund, sliced_indices) if compute_style else np.nan,
        "aum_score": aum_score,
        "sortino_3y": sortino_ratio(weekly_returns(lookback_nav)),
        "max_drawdown_3y": mdd,
        "current_drawdown": current_drawdown(fund),
        "n_sip_windows": int(len(sip_joined)),
        "latest_sip_xirr": float(sip_joined["fund"].iloc[-1]) if len(sip_joined) else np.nan,
    }
    return {key: finite_or_nan(value) for key, value in result.items()}


def build_training_panel(
    nav_by_fund: Dict[str, pd.Series],
    bench_nav: pd.Series,
    index_navs: Dict[str, pd.Series],
    aum_scores: Dict[str, float],
) -> pd.DataFrame:
    bench_months = to_month_start(bench_nav)
    if len(bench_months) < MIN_TRAINING_MONTHS + SIP_MONTHS:
        return pd.DataFrame()

    start = max(MIN_TRAINING_MONTHS, SIP_MONTHS + 6)
    stop = len(bench_months) - SIP_MONTHS - 1
    # Quarterly-ish snapshots keep the training look-ahead-safe without spending
    # most runtime recomputing expensive rolling path metrics on overlapping dates.
    snapshot_dates = list(bench_months.index[start:stop:4])[-24:]
    rows: List[Dict[str, float]] = []

    for end_date in snapshot_dates:
        bench_target = forward_sip_xirr(bench_nav, end_date)
        if not np.isfinite(bench_target):
            continue

        for mf_id, nav in nav_by_fund.items():
            fund_target = forward_sip_xirr(nav, end_date)
            if not np.isfinite(fund_target):
                continue

            features = feature_snapshot(
                nav,
                bench_nav,
                index_navs,
                aum_scores.get(mf_id, 0.60),
                end_date=end_date - pd.Timedelta(days=1),
                compute_style=False,
            )
            if not features:
                continue

            row = {feature: features.get(feature, np.nan) for feature in FEATURES}
            row["target_sip_alpha"] = fund_target - bench_target
            rows.append(row)

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel = panel.replace([np.inf, -np.inf], np.nan)
    keep = panel[FEATURES].notna().sum(axis=1) >= max(5, len(FEATURES) // 2)
    return panel.loc[keep].reset_index(drop=True)


def learn_feature_weights(panel: pd.DataFrame) -> pd.Series:
    if panel.empty or len(panel) < MIN_TRAINING_OBS or "target_sip_alpha" not in panel:
        return PRIOR_WEIGHTS.copy()

    target = panel["target_sip_alpha"].replace([np.inf, -np.inf], np.nan)
    learned = pd.Series(0.0, index=FEATURES, dtype=float)

    for feature in FEATURES:
        series = panel[feature].replace([np.inf, -np.inf], np.nan)
        direction = FEATURE_DIRECTIONS.get(feature, 1.0)
        valid = pd.concat([(series * direction).rename("x"), target.rename("y")], axis=1).dropna()
        if len(valid) < MIN_TRAINING_OBS or valid["x"].nunique() < 4:
            continue
        corr = valid["x"].corr(valid["y"], method="spearman")
        if np.isfinite(corr) and corr > MIN_CORR:
            learned[feature] = min(float(corr), MAX_FEATURE_WEIGHT)

    if learned.sum() <= 0:
        return PRIOR_WEIGHTS.copy()

    learned = learned / learned.sum()
    blended = 0.58 * PRIOR_WEIGHTS + 0.42 * learned
    blended = blended.clip(upper=MAX_FEATURE_WEIGHT)
    return blended / blended.sum()


def percentile_scores(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=values.index)
    score_input = numeric if higher_is_better else -numeric
    ranks = score_input.rank(method="average", pct=True)
    return ranks.fillna(0.45).clip(0.0, 1.0)


def score_funds(current: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    scored = current.copy()
    normalized = pd.DataFrame(index=scored.index)

    for feature in FEATURES:
        direction = FEATURE_DIRECTIONS.get(feature, 1.0)
        normalized[feature] = percentile_scores(scored[feature], higher_is_better=direction > 0)

    base = normalized.mul(weights.reindex(FEATURES).fillna(0.0), axis=1).sum(axis=1)
    confidence = pd.to_numeric(scored["confidence"], errors="coerce").fillna(0.70).clip(0.45, 1.0)
    sip_windows = pd.to_numeric(scored.get("n_sip_windows", 0), errors="coerce").fillna(0.0)
    sip_evidence = (0.55 + 0.45 * (sip_windows / 24.0).clip(0.0, 1.0)).clip(0.55, 1.0)
    scored["raw_score"] = base
    scored["score"] = (
        100.0
        * (0.88 * base + 0.12 * confidence)
        * (0.78 + 0.22 * confidence)
        * sip_evidence
    ).clip(0, 100)
    scored = scored.sort_values(["score", "sip_xirr_p25", "sip_alpha_hit_rate"], ascending=[False, False, False])
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def load_funds(provider: MfDataProvider) -> pd.DataFrame:
    all_funds = provider.list_all_mf()
    mask = (all_funds["sector"] == "Equity") & (all_funds["subsector"] == SUBSECTOR)
    funds = all_funds.loc[mask, ["mfId", "name", "aum", "sector", "subsector"]].copy()
    funds["aum"] = pd.to_numeric(funds["aum"], errors="coerce")
    return funds.sort_values("aum", ascending=False, na_position="last").reset_index(drop=True)


def load_index_navs(provider: MfDataProvider) -> Dict[str, pd.Series]:
    index_navs: Dict[str, pd.Series] = {}
    for name in ["Large Cap", "Mid Cap", "Small Cap", "Total Market"]:
        try:
            index_navs[name] = clean_nav_chart(provider.get_index_chart(name))
        except Exception as exc:  # noqa: BLE001 - missing auxiliary indices should not stop scoring
            logger.warning("Unable to load %s index: %s", name, exc)
            index_navs[name] = pd.Series(dtype=float)
    return index_navs


def build_current_rows(
    provider: MfDataProvider,
    funds: pd.DataFrame,
    bench_nav: pd.Series,
    index_navs: Dict[str, pd.Series],
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, float]]:
    aum_scores = {str(row.mfId): aum_quality_score(row.aum) for row in funds.itertuples(index=False)}
    rows: List[Dict[str, float]] = []
    nav_by_fund: Dict[str, pd.Series] = {}

    for fund in funds.itertuples(index=False):
        mf_id = str(fund.mfId)
        name = str(fund.name)
        try:
            nav = clean_nav_chart(provider.get_mf_chart(mf_id))
        except Exception as exc:  # noqa: BLE001 - continue ranking other funds
            logger.warning("Skipping %s (%s): %s", name, mf_id, exc)
            continue

        if len(to_weekly(nav)) < MIN_HISTORY_WEEKS:
            logger.info("Skipping %s (%s): insufficient history", name, mf_id)
            continue

        aum_score = aum_scores.get(mf_id, 0.60)
        features = feature_snapshot(nav, bench_nav, index_navs, aum_score)
        if not features:
            logger.info("Skipping %s (%s): insufficient aligned benchmark data", name, mf_id)
            continue

        nav_by_fund[mf_id] = nav
        data_days = int((nav.index[-1] - nav.index[0]).days + 1)
        history_years = data_days / DAYS_PER_YEAR
        history_confidence = np.clip(history_years / 4.0, 0.50, 1.0)
        sip_confidence = np.clip(features.get("n_sip_windows", 0) / 48.0, 0.45, 1.0)
        confidence = float(np.clip(0.58 * history_confidence + 0.24 * sip_confidence + 0.18 * aum_score, 0.45, 1.0))

        row = {
            "mfId": mf_id,
            "name": name,
            "aum": finite_or_nan(fund.aum),
            "data_days": data_days,
            "history_years": history_years,
            "confidence": confidence,
            "cagr_3y": calendar_cagr(nav, 3.0),
            "cagr_5y": calendar_cagr(nav, 5.0),
        }
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows), nav_by_fund, aum_scores


def format_output(ranked: pd.DataFrame) -> pd.DataFrame:
    output_cols = [
        "mfId",
        "name",
        "rank",
        "score",
        "data_days",
        "cagr_3y",
        "cagr_5y",
        "sip_xirr_median",
        "sip_xirr_p25",
        "latest_sip_xirr",
        "sip_alpha_hit_rate",
        "alpha_consistency",
        "information_ratio",
        "up_capture",
        "down_capture",
        "downside_beta",
        "drawdown_control",
        "recovery_strength",
        "correction_defense",
        "momentum_quality",
        "omega_ratio",
        "sortino_3y",
        "ulcer_index",
        "cdar_5",
        "max_drawdown_3y",
        "current_drawdown",
        "style_purity",
        "aum_score",
        "confidence",
        "aum",
        "history_years",
        "n_sip_windows",
    ]
    for col in output_cols:
        if col not in ranked:
            ranked[col] = np.nan

    out = ranked[output_cols].copy()
    pct_cols = [
        "cagr_3y",
        "cagr_5y",
        "sip_xirr_median",
        "sip_xirr_p25",
        "latest_sip_xirr",
        "drawdown_control",
        "recovery_strength",
        "correction_defense",
        "cdar_5",
        "max_drawdown_3y",
        "current_drawdown",
    ]
    ratio_cols = [
        "sip_alpha_hit_rate",
        "alpha_consistency",
        "information_ratio",
        "up_capture",
        "down_capture",
        "downside_beta",
        "momentum_quality",
        "omega_ratio",
        "sortino_3y",
        "ulcer_index",
        "style_purity",
        "aum_score",
        "confidence",
        "aum",
        "history_years",
    ]

    out["score"] = pd.to_numeric(out["score"], errors="coerce").round(2)
    for col in pct_cols:
        out[col] = (pd.to_numeric(out[col], errors="coerce") * 100.0).round(2)
    for col in ratio_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    out["aum"] = pd.to_numeric(out["aum"], errors="coerce").round(2)
    out["n_sip_windows"] = pd.to_numeric(out["n_sip_windows"], errors="coerce").fillna(0).astype(int)
    return out


def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 82)
    print("  MID CAP MUTUAL FUND SCORING - GPT")
    print("  Target: next 1Y monthly SIP with recovery, quality, and downside discipline")
    print("=" * 82)

    provider = MfDataProvider(date=date)
    funds = load_funds(provider)
    if funds.empty:
        raise RuntimeError(f"No {SUBSECTOR} funds found")

    index_navs = load_index_navs(provider)
    bench_nav = index_navs.get("Mid Cap", pd.Series(dtype=float))
    if bench_nav.empty:
        raise RuntimeError("Unable to load Mid Cap benchmark index")

    print(f"  Funds discovered : {len(funds)}")
    print(f"  Benchmark        : {BENCHMARK_INDEX}")
    print("  Computing current fund features...")

    current, nav_by_fund, aum_scores = build_current_rows(provider, funds, bench_nav, index_navs)
    if current.empty:
        raise RuntimeError("No funds had enough NAV history to score")

    print("  Building look-ahead-safe SIP training panel...")
    panel = build_training_panel(nav_by_fund, bench_nav, index_navs, aum_scores)
    weights = learn_feature_weights(panel)

    ranked = score_funds(current, weights)
    output = format_output(ranked)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n  Learned/blended feature weights:")
    for feature, weight in weights.sort_values(ascending=False).items():
        print(f"    {feature:22s} {weight * 100:5.1f}%")

    print(f"\n  Training observations : {len(panel)}")
    print(f"  Funds scored          : {len(output)}")
    print(f"  Results saved         : {OUTPUT_FILE}")
    print("\n  Top 10 funds:")
    display_cols = ["rank", "mfId", "name", "score", "cagr_3y", "sip_xirr_p25", "confidence"]
    print(output[display_cols].head(10).to_string(index=False))
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mid Cap MF screener (GPT)")
    parser.add_argument(
        "--date",
        default=None,
        help="Data snapshot date in YYYY-MM-DD format, e.g. 2026-05-09",
    )
    args = parser.parse_args()
    main(date=args.date)
