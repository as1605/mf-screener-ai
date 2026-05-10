#!/usr/bin/env python3
"""
Total Market Mutual Fund Scoring Algorithm - GPT
================================================

Ranks Indian diversified equity funds for the user's exact cashflow:
12 monthly SIP investments, 12 months of hold, and one exit at month 24.

The model translates the 2027-2028 India equity thesis into NAV-observable
signals: resilient rolling SIP-hold XIRR, recovery participation, hold-phase
drawdown control, factor-adjusted alpha, and capacity/history confidence.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq


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


SECTOR = "Total Market"
SUBSECTORS = [
    "Contra Fund",
    "Flexi Cap Fund",
    "Focused Fund",
    "Multi Cap Fund",
    "Value Fund",
]
BENCHMARK_INDEX = "Total Market"
FACTOR_INDICES = {
    "large": "Large Cap",
    "mid": "Mid Cap",
    "small": "Small Cap",
    "market": "Total Market",
}

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_GPT.csv"
TMP_OUTPUT_DIR = ROOT_DIR / "data" / "tmp"
DIAGNOSTICS_FILE = TMP_OUTPUT_DIR / f"{SECTOR}_GPT_diagnostics.csv"

RISK_FREE_RATE = 0.065
WEEKS_PER_YEAR = 52.0
SIP_MONTHS = 12
HOLD_MONTHS = 12
TOTAL_MONTHS = SIP_MONTHS + HOLD_MONTHS
SIP_AMOUNT = 10000.0

MIN_DAYS = 365
MIN_SCENARIO_COUNT = 3
MAX_STALE_DAYS = 45
SCENARIO_LOOKBACK_MONTHS = 60
MAX_DIAGNOSTIC_SNAPSHOTS = 3

PILLAR_WEIGHTS = {
    "hybrid": 0.34,
    "recovery": 0.18,
    "resilience": 0.22,
    "active_skill": 0.18,
    "confidence": 0.08,
}


@dataclass
class FundData:
    mf_id: str
    name: str
    subsector: str
    aum: float
    nav: pd.Series
    xirr_series: pd.Series
    bench_xirr_series: pd.Series
    data_days: int


def clean_nav_to_series(df: pd.DataFrame) -> pd.Series:
    """Convert raw chart data to a sorted, positive NAV series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["timestamp", "nav"])
    out = out[out["nav"] > 0]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    series = out.set_index("timestamp")["nav"].astype(float)
    if series.index.tz is not None:
        series.index = series.index.tz_convert(None)
    return series.sort_index()


def weekly_nav(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.resample("W-FRI").last().ffill().dropna()


def pct_return(nav: pd.Series, years: float) -> float:
    if len(nav) < 2:
        return np.nan
    cutoff = nav.index.max() - pd.Timedelta(days=int(365.25 * years))
    history = nav[nav.index >= cutoff]
    if history.empty or (nav.index.max() - history.index.min()).days < 330 * years:
        return np.nan
    start = float(history.iloc[0])
    end = float(nav.iloc[-1])
    if start <= 0 or end <= 0:
        return np.nan
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def nav_at_or_after(nav: pd.Series, target: pd.Timestamp, max_gap_days: int = 10) -> Optional[Tuple[pd.Timestamp, float]]:
    idx = nav.index.searchsorted(target, side="left")
    if idx >= len(nav):
        return None
    actual = nav.index[idx]
    if (actual - target).days > max_gap_days:
        return None
    return actual, float(nav.iloc[idx])


def nav_at_or_before(nav: pd.Series, target: pd.Timestamp, max_gap_days: int = 10) -> Optional[Tuple[pd.Timestamp, float]]:
    idx = nav.index.searchsorted(target, side="right") - 1
    if idx < 0:
        return None
    actual = nav.index[idx]
    if (target - actual).days > max_gap_days:
        return None
    return actual, float(nav.iloc[idx])


def xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    if len(cashflows) < 2:
        return np.nan
    start = cashflows[0][0]
    years = np.array([(date - start).days / 365.25 for date, _ in cashflows], dtype=float)
    amounts = np.array([amount for _, amount in cashflows], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / np.power(1.0 + rate, years)))

    try:
        return float(brentq(npv, -0.9999, 20.0, maxiter=100))
    except (ValueError, OverflowError, FloatingPointError):
        return np.nan


def simulate_hybrid_xirr(nav: pd.Series, start_date: pd.Timestamp) -> float:
    """Simulate 12 monthly SIP buys from start_date and one sale at month 24."""
    if nav.empty:
        return np.nan
    cashflows: List[Tuple[pd.Timestamp, float]] = []
    units = 0.0

    for month in range(SIP_MONTHS):
        target = start_date + pd.DateOffset(months=month)
        hit = nav_at_or_after(nav, pd.Timestamp(target))
        if hit is None:
            return np.nan
        date, price = hit
        units += SIP_AMOUNT / price
        cashflows.append((date, -SIP_AMOUNT))

    exit_target = start_date + pd.DateOffset(months=TOTAL_MONTHS)
    hit = nav_at_or_before(nav, pd.Timestamp(exit_target))
    if hit is None:
        return np.nan
    exit_date, exit_nav = hit
    cashflows.append((exit_date, units * exit_nav))
    return xirr(cashflows)


def month_start_dates(nav: pd.Series) -> Iterable[pd.Timestamp]:
    if nav.empty:
        return []
    lookback_first = nav.index.max().normalize() - pd.DateOffset(months=SCENARIO_LOOKBACK_MONTHS)
    first = max(nav.index.min().normalize() + pd.offsets.MonthBegin(1), lookback_first)
    last = nav.index.max().normalize() - pd.DateOffset(months=TOTAL_MONTHS)
    if first > last:
        return []
    return pd.date_range(first, last, freq="MS")


def rolling_hybrid_xirr(nav: pd.Series) -> pd.Series:
    rows = []
    for start in month_start_dates(nav):
        value = simulate_hybrid_xirr(nav, pd.Timestamp(start))
        if pd.notna(value):
            rows.append((pd.Timestamp(start), value))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows), dtype=float).sort_index()


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    running_peak = nav.cummax()
    drawdowns = nav / running_peak - 1.0
    return float(drawdowns.min())


def ulcer_index(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    drawdown_pct = (nav / nav.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(np.minimum(drawdown_pct, 0.0)))))


def drawdown_recovery_weeks(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    weekly = weekly_nav(nav)
    if len(weekly) < 2:
        return np.nan
    peak = weekly.cummax()
    dd = weekly / peak - 1.0
    trough_date = dd.idxmin()
    prior_peak_val = float(peak.loc[trough_date])
    after = weekly[weekly.index >= trough_date]
    recovered = after[after >= prior_peak_val * 0.995]
    if recovered.empty:
        return float(len(after))
    return float(max(0, (recovered.index[0] - trough_date).days / 7.0))


def current_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    return float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 10:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def align_weekly_returns(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    frames = {}
    for key, nav in series_map.items():
        wk = weekly_nav(nav)
        frames[key] = wk.pct_change()
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="any")


def classify_regimes(bench_nav: pd.Series) -> pd.Series:
    wk = weekly_nav(bench_nav)
    ret_13 = wk.pct_change(13)
    ret_26 = wk.pct_change(26)
    dd = wk / wk.cummax() - 1.0
    regimes = pd.Series("sideways", index=wk.index, dtype=object)
    regimes[(dd <= -0.18) | (ret_26 <= -0.14)] = "bear"
    regimes[((dd <= -0.08) | (ret_13 <= -0.06)) & (regimes != "bear")] = "correction"
    regimes[(ret_13 >= 0.08) & (dd > -0.08)] = "bull"
    regimes[(ret_13 >= 0.04) & (ret_26.shift(13) < 0) & (dd > -0.16)] = "recovery"
    return regimes


def capture_ratio(fund_ret: pd.Series, bench_ret: pd.Series, mask: pd.Series) -> float:
    aligned = pd.DataFrame({"fund": fund_ret, "bench": bench_ret, "mask": mask}).dropna()
    aligned = aligned[aligned["mask"]]
    if len(aligned) < 6 or abs(aligned["bench"].sum()) < 1e-9:
        return np.nan
    return float(aligned["fund"].sum() / aligned["bench"].sum())


def excess_return_in_mask(fund_ret: pd.Series, bench_ret: pd.Series, mask: pd.Series) -> float:
    aligned = pd.DataFrame({"fund": fund_ret, "bench": bench_ret, "mask": mask}).dropna()
    aligned = aligned[aligned["mask"]]
    if len(aligned) < 6:
        return np.nan
    return float((aligned["fund"] - aligned["bench"]).mean() * WEEKS_PER_YEAR)


def factor_metrics(fund_nav: pd.Series, factor_navs: Dict[str, pd.Series]) -> Dict[str, float]:
    returns = align_weekly_returns({"fund": fund_nav, **factor_navs})
    returns = returns.tail(156)
    if len(returns) < 52:
        return {
            "factor_alpha": np.nan,
            "factor_ir": np.nan,
            "alpha_stability": np.nan,
            "market_beta": np.nan,
            "small_mid_tilt": np.nan,
        }

    rf_week = (1.0 + RISK_FREE_RATE) ** (1.0 / WEEKS_PER_YEAR) - 1.0
    y = returns["fund"].to_numpy(dtype=float) - rf_week
    x_cols = ["large", "mid", "small", "market"]
    x = returns[x_cols].to_numpy(dtype=float) - rf_week
    x = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "factor_alpha": np.nan,
            "factor_ir": np.nan,
            "alpha_stability": np.nan,
            "market_beta": np.nan,
            "small_mid_tilt": np.nan,
        }

    residual = y - x @ coeffs
    resid_vol = float(np.std(residual, ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if len(residual) > 2 else np.nan
    alpha_ann = float(coeffs[0] * WEEKS_PER_YEAR)
    factor_ir = alpha_ann / resid_vol if resid_vol and resid_vol > 1e-9 else np.nan

    rolling_alphas: List[float] = []
    if len(returns) >= 78:
        for start in range(0, len(returns) - 52 + 1, 13):
            chunk = returns.iloc[start : start + 52]
            y_c = chunk["fund"].to_numpy(dtype=float) - rf_week
            x_c = chunk[x_cols].to_numpy(dtype=float) - rf_week
            x_c = np.column_stack([np.ones(len(x_c)), x_c])
            try:
                c, *_ = np.linalg.lstsq(x_c, y_c, rcond=None)
                rolling_alphas.append(float(c[0] * WEEKS_PER_YEAR))
            except np.linalg.LinAlgError:
                continue
    alpha_stability = (
        float(np.mean(rolling_alphas) / (np.std(rolling_alphas, ddof=1) + 1e-9))
        if len(rolling_alphas) >= 3
        else np.nan
    )

    return {
        "factor_alpha": alpha_ann,
        "factor_ir": factor_ir,
        "alpha_stability": alpha_stability,
        "market_beta": float(coeffs[4]),
        "small_mid_tilt": float(coeffs[2] + coeffs[3]),
    }


def safe_quantile(values: pd.Series, q: float) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def analyse_fund(
    fund: FundData,
    bench_nav: pd.Series,
    factor_navs: Dict[str, pd.Series],
    regimes: pd.Series,
    peer_xirr_median: pd.Series,
) -> Dict[str, float | str]:
    nav = fund.nav
    wk_returns = align_weekly_returns({"fund": nav, "bench": bench_nav})
    fund_ret = wk_returns["fund"] if "fund" in wk_returns else pd.Series(dtype=float)
    bench_ret = wk_returns["bench"] if "bench" in wk_returns else pd.Series(dtype=float)
    regime_aligned = regimes.reindex(wk_returns.index).ffill() if not wk_returns.empty else pd.Series(dtype=object)

    xirr_series = fund.xirr_series
    common_bench = fund.bench_xirr_series.reindex(xirr_series.index).dropna()
    common_peer = peer_xirr_median.reindex(xirr_series.index).dropna()
    xirr_vs_bench = pd.DataFrame({"fund": xirr_series, "bench": common_bench}).dropna()
    xirr_vs_peer = pd.DataFrame({"fund": xirr_series, "peer": common_peer}).dropna()

    recent_xirr = float(xirr_series.iloc[-1]) if len(xirr_series) else np.nan
    hybrid_p50 = safe_quantile(xirr_series, 0.50)
    hybrid_p25 = safe_quantile(xirr_series, 0.25)
    hit_bench = float((xirr_vs_bench["fund"] > xirr_vs_bench["bench"]).mean()) if len(xirr_vs_bench) else np.nan
    hit_peer = float((xirr_vs_peer["fund"] > xirr_vs_peer["peer"]).mean()) if len(xirr_vs_peer) else np.nan

    last_1y = nav[nav.index >= nav.index.max() - pd.Timedelta(days=370)]
    last_2y = nav[nav.index >= nav.index.max() - pd.Timedelta(days=740)]
    last_3y = nav[nav.index >= nav.index.max() - pd.Timedelta(days=1110)]

    up_mask = (bench_ret > 0) & regime_aligned.isin(["bull", "recovery"])
    down_mask = bench_ret < 0
    bear_mask = regime_aligned.isin(["correction", "bear"])

    ret_13_f = weekly_nav(nav).pct_change(13).iloc[-1] if len(weekly_nav(nav)) > 14 else np.nan
    ret_13_b = weekly_nav(bench_nav).pct_change(13).reindex(weekly_nav(nav).index).dropna()
    ret_13_b_last = ret_13_b.iloc[-1] if len(ret_13_b) else np.nan
    ret_26_f = weekly_nav(nav).pct_change(26).iloc[-1] if len(weekly_nav(nav)) > 27 else np.nan
    ret_26_b = weekly_nav(bench_nav).pct_change(26).reindex(weekly_nav(nav).index).dropna()
    ret_26_b_last = ret_26_b.iloc[-1] if len(ret_26_b) else np.nan
    momentum_3m_rel = float(ret_13_f - ret_13_b_last) if pd.notna(ret_13_f) and pd.notna(ret_13_b_last) else np.nan
    momentum_6m_rel = float(ret_26_f - ret_26_b_last) if pd.notna(ret_26_f) and pd.notna(ret_26_b_last) else np.nan
    overheat_penalty = max(0.0, float(current_drawdown(nav) + 0.015)) if pd.notna(current_drawdown(nav)) else np.nan

    factor = factor_metrics(nav, factor_navs)

    metrics: Dict[str, float | str] = {
        "mfId": fund.mf_id,
        "name": fund.name,
        "subsector": fund.subsector,
        "aum": fund.aum,
        "data_days": fund.data_days,
        "data_weeks": int(len(weekly_nav(nav))),
        "cagr_1y": pct_return(nav, 1.0),
        "cagr_3y": pct_return(nav, 3.0),
        "cagr_5y": pct_return(nav, 5.0),
        "hybrid_xirr_p50": hybrid_p50 * 100.0 if pd.notna(hybrid_p50) else np.nan,
        "hybrid_xirr_p25": hybrid_p25 * 100.0 if pd.notna(hybrid_p25) else np.nan,
        "hybrid_xirr_recent": recent_xirr * 100.0 if pd.notna(recent_xirr) else np.nan,
        "hybrid_hit_bench": hit_bench * 100.0 if pd.notna(hit_bench) else np.nan,
        "hybrid_hit_peer": hit_peer * 100.0 if pd.notna(hit_peer) else np.nan,
        "hybrid_window_n": float(len(xirr_series)),
        "up_recovery_capture": capture_ratio(fund_ret, bench_ret, up_mask),
        "bull_recovery_excess": excess_return_in_mask(fund_ret, bench_ret, up_mask) * 100.0,
        "momentum_3m_rel": momentum_3m_rel * 100.0 if pd.notna(momentum_3m_rel) else np.nan,
        "momentum_6m_rel": momentum_6m_rel * 100.0 if pd.notna(momentum_6m_rel) else np.nan,
        "overheat_penalty": overheat_penalty * 100.0 if pd.notna(overheat_penalty) else np.nan,
        "max_drawdown_2y": max_drawdown(last_2y) * 100.0 if len(last_2y) else np.nan,
        "ulcer_2y": ulcer_index(last_2y),
        "down_capture": capture_ratio(fund_ret, bench_ret, down_mask),
        "bear_excess": excess_return_in_mask(fund_ret, bench_ret, bear_mask) * 100.0,
        "recovery_weeks": drawdown_recovery_weeks(last_2y),
        "tail_loss_1y": safe_quantile(fund_ret.tail(52), 0.05) * 100.0,
        "volatility_1y": annualized_vol(fund_ret.tail(52)) * 100.0,
        "current_drawdown": current_drawdown(nav) * 100.0 if pd.notna(current_drawdown(nav)) else np.nan,
        "factor_alpha": factor["factor_alpha"] * 100.0 if pd.notna(factor["factor_alpha"]) else np.nan,
        "factor_ir": factor["factor_ir"],
        "alpha_stability": factor["alpha_stability"],
        "market_beta": factor["market_beta"],
        "small_mid_tilt": factor["small_mid_tilt"],
        "history_years": fund.data_days / 365.25 if fund.data_days else np.nan,
        "latest_nav_date": str(nav.index.max().date()) if len(nav) else "",
    }
    metrics["history_confidence"] = history_confidence(float(metrics["history_years"]))
    metrics["capacity_multiplier"] = capacity_multiplier(fund.aum)
    metrics["xirr_coverage"] = min(1.0, len(xirr_series) / 24.0)
    metrics["last_3y_max_drawdown"] = max_drawdown(last_3y) * 100.0 if len(last_3y) else np.nan
    return metrics


def history_confidence(years: float) -> float:
    if pd.isna(years):
        return 0.55
    if years < 1.0:
        return 0.55
    if years < 2.0:
        return 0.72
    if years < 3.0:
        return 0.88
    if years < 5.0:
        return 0.96
    return 1.0


def capacity_multiplier(aum: float) -> float:
    if pd.isna(aum) or aum <= 0:
        return 0.94
    if aum < 500:
        return 0.93
    if aum <= 25000:
        return 1.0
    if aum <= 75000:
        return 0.98 - 0.05 * ((aum - 25000) / 50000)
    return max(0.86, 0.93 - 0.04 * np.log10(aum / 75000.0 + 1.0))


def percentile_score(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.notna()
    out = pd.Series(50.0, index=df.index, dtype=float)
    if valid.sum() == 0:
        return out
    ranks = values[valid].rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks + (1.0 / valid.sum())
    out.loc[valid] = ranks.clip(0.0, 1.0) * 100.0
    return out


def compute_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()

    df["hybrid_score"] = (
        0.28 * percentile_score(df, "hybrid_xirr_p50")
        + 0.24 * percentile_score(df, "hybrid_xirr_p25")
        + 0.16 * percentile_score(df, "hybrid_xirr_recent")
        + 0.16 * percentile_score(df, "hybrid_hit_bench")
        + 0.12 * percentile_score(df, "hybrid_hit_peer")
        + 0.04 * percentile_score(df, "hybrid_window_n")
    )
    df["recovery_score"] = (
        0.30 * percentile_score(df, "up_recovery_capture")
        + 0.24 * percentile_score(df, "bull_recovery_excess")
        + 0.20 * percentile_score(df, "momentum_6m_rel")
        + 0.14 * percentile_score(df, "momentum_3m_rel")
        + 0.12 * percentile_score(df, "overheat_penalty", higher_is_better=False)
    )
    df["resilience_score"] = (
        0.22 * percentile_score(df, "max_drawdown_2y", higher_is_better=False)
        + 0.18 * percentile_score(df, "ulcer_2y", higher_is_better=False)
        + 0.18 * percentile_score(df, "down_capture", higher_is_better=False)
        + 0.18 * percentile_score(df, "bear_excess")
        + 0.12 * percentile_score(df, "recovery_weeks", higher_is_better=False)
        + 0.12 * percentile_score(df, "tail_loss_1y")
    )
    df["active_skill_score"] = (
        0.34 * percentile_score(df, "factor_alpha")
        + 0.22 * percentile_score(df, "factor_ir")
        + 0.18 * percentile_score(df, "alpha_stability")
        + 0.14 * percentile_score(df, "market_beta", higher_is_better=False)
        + 0.12 * percentile_score(df, "cagr_3y")
    )
    df["confidence_score"] = (
        0.42 * (pd.to_numeric(df["history_confidence"], errors="coerce").fillna(0.55) * 100.0)
        + 0.28 * (pd.to_numeric(df["capacity_multiplier"], errors="coerce").fillna(0.94) * 100.0)
        + 0.18 * (pd.to_numeric(df["xirr_coverage"], errors="coerce").fillna(0.0) * 100.0)
        + 0.12 * percentile_score(df, "data_days")
    )

    raw = (
        PILLAR_WEIGHTS["hybrid"] * df["hybrid_score"]
        + PILLAR_WEIGHTS["recovery"] * df["recovery_score"]
        + PILLAR_WEIGHTS["resilience"] * df["resilience_score"]
        + PILLAR_WEIGHTS["active_skill"] * df["active_skill_score"]
        + PILLAR_WEIGHTS["confidence"] * df["confidence_score"]
    )
    confidence_haircut = (
        pd.to_numeric(df["history_confidence"], errors="coerce").fillna(0.55)
        * pd.to_numeric(df["capacity_multiplier"], errors="coerce").fillna(0.94)
        * (0.70 + 0.30 * pd.to_numeric(df["xirr_coverage"], errors="coerce").fillna(0.0))
    )
    df["score"] = (raw * confidence_haircut).clip(0, 100)
    df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)


def latest_market_regime(bench_nav: pd.Series) -> str:
    regimes = classify_regimes(bench_nav)
    return str(regimes.iloc[-1]) if len(regimes) else "unknown"


def load_funds(provider: MfDataProvider, bench_nav: pd.Series) -> List[FundData]:
    df_all = provider.list_all_mf()
    universe = df_all[df_all["subsector"].isin(SUBSECTORS)].copy()
    funds: List[FundData] = []
    bench_xirr = rolling_hybrid_xirr(bench_nav)

    logger.info("Loading %d Total Market candidate funds", len(universe))
    for _, row in universe.iterrows():
        mf_id = str(row["mfId"])
        name = str(row["name"])
        try:
            nav = clean_nav_to_series(provider.get_mf_chart(mf_id))
        except Exception as exc:
            logger.warning("Skipping %s: failed to load NAV (%s)", mf_id, exc)
            continue
        if len(nav) < 2:
            continue
        data_days = int((nav.index.max() - nav.index.min()).days)
        stale_days = int((bench_nav.index.max() - nav.index.max()).days)
        if data_days < MIN_DAYS or stale_days > MAX_STALE_DAYS:
            continue
        xirr_series = rolling_hybrid_xirr(nav)
        if len(xirr_series) < MIN_SCENARIO_COUNT:
            continue
        funds.append(
            FundData(
                mf_id=mf_id,
                name=name,
                subsector=str(row["subsector"]),
                aum=float(row.get("aum", 0.0) or 0.0),
                nav=nav,
                xirr_series=xirr_series,
                bench_xirr_series=bench_xirr,
                data_days=data_days,
            )
        )
    return funds


def peer_median_xirr(funds: List[FundData]) -> pd.Series:
    if not funds:
        return pd.Series(dtype=float)
    panel = pd.DataFrame({f.mf_id: f.xirr_series for f in funds})
    return panel.median(axis=1, skipna=True).dropna()


def rank_ic(scores: pd.Series, outcomes: pd.Series) -> float:
    aligned = pd.DataFrame({"score": scores, "outcome": outcomes}).dropna()
    if len(aligned) < 5 or aligned["score"].nunique() < 2 or aligned["outcome"].nunique() < 2:
        return np.nan
    return float(aligned["score"].rank().corr(aligned["outcome"].rank()))


def walk_forward_diagnostics(
    funds: List[FundData],
    bench_nav: pd.Series,
    factor_navs: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Bounded historical test using only data available at each snapshot date."""
    if not funds:
        return pd.DataFrame()
    start = max(
        bench_nav.index.min() + pd.DateOffset(years=2),
        bench_nav.index.max() - pd.DateOffset(months=SCENARIO_LOOKBACK_MONTHS),
    )
    end = bench_nav.index.max() - pd.DateOffset(months=TOTAL_MONTHS)
    if start >= end:
        return pd.DataFrame()

    snapshots = pd.date_range(start.normalize(), end.normalize(), freq="6MS")
    snapshots = list(snapshots[-MAX_DIAGNOSTIC_SNAPSHOTS:])
    bench_xirr_all = rolling_hybrid_xirr(bench_nav)
    rows = []
    for snapshot in snapshots:
        clipped_bench = bench_nav[bench_nav.index <= snapshot]
        if len(clipped_bench) < 90:
            continue
        clipped_factors = {k: v[v.index <= snapshot] for k, v in factor_navs.items()}
        regimes = classify_regimes(clipped_bench)
        partial_funds: List[FundData] = []
        outcomes = {}
        for fund in funds:
            nav_hist = fund.nav[fund.nav.index <= snapshot]
            if len(nav_hist) < 260:
                continue
            future_candidates = fund.xirr_series[fund.xirr_series.index >= pd.Timestamp(snapshot).normalize()]
            if future_candidates.empty:
                continue
            future_xirr = float(future_candidates.iloc[0])
            if pd.isna(future_xirr):
                continue
            available_cutoff = pd.Timestamp(snapshot).normalize() - pd.DateOffset(months=TOTAL_MONTHS)
            xirr_hist = fund.xirr_series[fund.xirr_series.index <= available_cutoff]
            if len(xirr_hist) < 2:
                continue
            bench_xirr_hist = bench_xirr_all[bench_xirr_all.index <= available_cutoff]
            partial_funds.append(
                FundData(
                    mf_id=fund.mf_id,
                    name=fund.name,
                    subsector=fund.subsector,
                    aum=fund.aum,
                    nav=nav_hist,
                    xirr_series=xirr_hist,
                    bench_xirr_series=bench_xirr_hist,
                    data_days=int((nav_hist.index.max() - nav_hist.index.min()).days),
                )
            )
            outcomes[fund.mf_id] = future_xirr
        if len(partial_funds) < 12:
            continue
        peer = peer_median_xirr(partial_funds)
        feature_rows = [
            analyse_fund(fund, clipped_bench, clipped_factors, regimes, peer)
            for fund in partial_funds
        ]
        scored = compute_scores(pd.DataFrame(feature_rows))
        future = pd.Series(outcomes)
        ic = rank_ic(scored.set_index("mfId")["score"], future)
        top = scored.head(7)["mfId"].tolist()
        top_future = future.reindex(top).dropna()
        all_future = future.reindex(scored["mfId"]).dropna()
        if all_future.empty or top_future.empty:
            continue
        bench_candidates = bench_xirr_all[bench_xirr_all.index >= pd.Timestamp(snapshot).normalize()]
        bench_future = float(bench_candidates.iloc[0]) if not bench_candidates.empty else np.nan
        top_excess = float(top_future.mean() - all_future.median())
        hit_rate = float((top_future > all_future.median()).mean())
        bench_hit_rate = float((top_future > bench_future).mean()) if pd.notna(bench_future) else np.nan
        rows.append(
            {
                "snapshot": snapshot.date().isoformat(),
                "fund_count": len(scored),
                "rank_ic": ic,
                "top7_excess_vs_peer_median": top_excess * 100.0,
                "top7_peer_hit_rate": hit_rate * 100.0,
                "top7_benchmark_hit_rate": bench_hit_rate * 100.0 if pd.notna(bench_hit_rate) else np.nan,
                "future_peer_median_xirr": all_future.median() * 100.0,
                "future_top7_mean_xirr": top_future.mean() * 100.0,
            }
        )
    return pd.DataFrame(rows)


def round_cols(df: pd.DataFrame, cols: Iterable[str], digits: int = 2) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(digits)


def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 92)
    print("  TOTAL MARKET MUTUAL FUND SCORING ALGORITHM - GPT")
    print("  Method: 2027-2028 thesis mapped to 24M SIP-hold NAV evidence")
    print("  Benchmark: Nifty 500 / Total Market")
    print("=" * 92)

    provider = MfDataProvider(date=date)
    bench_nav = clean_nav_to_series(provider.get_index_chart(BENCHMARK_INDEX))
    if bench_nav.empty:
        raise RuntimeError("Benchmark NAV history is empty.")
    factor_navs = {
        name: clean_nav_to_series(provider.get_index_chart(index_name))
        for name, index_name in FACTOR_INDICES.items()
    }
    factor_navs = {k: v for k, v in factor_navs.items() if not v.empty}
    if set(factor_navs) != set(FACTOR_INDICES):
        raise RuntimeError(f"Missing factor index data: {set(FACTOR_INDICES) - set(factor_navs)}")

    print(
        f"  Benchmark history: {len(bench_nav)} daily rows "
        f"({bench_nav.index.min().date()} -> {bench_nav.index.max().date()})"
    )

    funds = load_funds(provider, bench_nav)
    if not funds:
        raise RuntimeError("No eligible Total Market funds found.")
    print(f"  Eligible Total Market funds: {len(funds)}")

    regimes = classify_regimes(bench_nav)
    market_regime = latest_market_regime(bench_nav)
    peer_xirr = peer_median_xirr(funds)
    metrics = [
        analyse_fund(fund, bench_nav, factor_navs, regimes, peer_xirr)
        for fund in funds
    ]
    scored = compute_scores(pd.DataFrame(metrics))
    scored["market_regime"] = market_regime

    diagnostics = walk_forward_diagnostics(funds, bench_nav, factor_navs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "mfId",
        "name",
        "rank",
        "score",
        "data_days",
        "subsector",
        "cagr_1y",
        "cagr_3y",
        "cagr_5y",
        "hybrid_xirr_p50",
        "hybrid_xirr_p25",
        "hybrid_xirr_recent",
        "hybrid_hit_bench",
        "hybrid_hit_peer",
        "hybrid_window_n",
        "up_recovery_capture",
        "bull_recovery_excess",
        "momentum_3m_rel",
        "momentum_6m_rel",
        "overheat_penalty",
        "max_drawdown_2y",
        "ulcer_2y",
        "down_capture",
        "bear_excess",
        "recovery_weeks",
        "tail_loss_1y",
        "volatility_1y",
        "current_drawdown",
        "factor_alpha",
        "factor_ir",
        "alpha_stability",
        "market_beta",
        "small_mid_tilt",
        "hybrid_score",
        "recovery_score",
        "resilience_score",
        "active_skill_score",
        "confidence_score",
        "history_confidence",
        "capacity_multiplier",
        "xirr_coverage",
        "aum",
        "data_weeks",
        "history_years",
        "latest_nav_date",
        "market_regime",
    ]
    export = scored[[c for c in output_cols if c in scored.columns]].copy()
    round_cols(
        export,
        [
            "score",
            "cagr_1y",
            "cagr_3y",
            "cagr_5y",
            "hybrid_xirr_p50",
            "hybrid_xirr_p25",
            "hybrid_xirr_recent",
            "hybrid_hit_bench",
            "hybrid_hit_peer",
            "up_recovery_capture",
            "bull_recovery_excess",
            "momentum_3m_rel",
            "momentum_6m_rel",
            "overheat_penalty",
            "max_drawdown_2y",
            "ulcer_2y",
            "down_capture",
            "bear_excess",
            "recovery_weeks",
            "tail_loss_1y",
            "volatility_1y",
            "current_drawdown",
            "factor_alpha",
            "factor_ir",
            "alpha_stability",
            "market_beta",
            "small_mid_tilt",
            "hybrid_score",
            "recovery_score",
            "resilience_score",
            "active_skill_score",
            "confidence_score",
            "history_confidence",
            "capacity_multiplier",
            "xirr_coverage",
            "history_years",
        ],
    )
    round_cols(export, ["aum"], digits=2)
    export.to_csv(OUTPUT_FILE, index=False)

    if not diagnostics.empty:
        round_cols(
            diagnostics,
            [
                "rank_ic",
                "top7_excess_vs_peer_median",
                "top7_peer_hit_rate",
                "top7_benchmark_hit_rate",
                "future_peer_median_xirr",
                "future_top7_mean_xirr",
            ],
        )
    diagnostics.to_csv(DIAGNOSTICS_FILE, index=False)

    print("\n  Walk-forward diagnostics:")
    if diagnostics.empty:
        print("    - Not enough full past+future windows for diagnostics.")
    else:
        print(
            f"    - snapshots: {len(diagnostics)}, "
            f"mean IC: {diagnostics['rank_ic'].mean():.3f}, "
            f"top7 excess: {diagnostics['top7_excess_vs_peer_median'].mean():.2f}%"
        )
    print(f"  Current market regime: {market_regime}")
    print(f"  Results saved: {OUTPUT_FILE}")
    print(f"  Diagnostics saved: {DIAGNOSTICS_FILE}")

    print("\n" + "-" * 92)
    print("  TOP 20 TOTAL MARKET FUNDS (GPT)")
    print("-" * 92)
    show_cols = [
        "rank",
        "name",
        "subsector",
        "score",
        "hybrid_xirr_p50",
        "hybrid_xirr_p25",
        "factor_alpha",
        "max_drawdown_2y",
        "market_regime",
    ]
    print(export.head(20)[show_cols].to_string(index=False))
    print("=" * 92 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total Market MF screener (GPT)")
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cached data folder under ./data (default: today)",
    )
    main(date=parser.parse_args().date)
