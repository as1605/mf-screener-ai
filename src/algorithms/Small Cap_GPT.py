#!/usr/bin/env python3
"""
Small Cap Mutual Fund Scoring Algorithm - GPT
=============================================

Objective
---------
Rank Indian Small Cap mutual funds for the next 1-year monthly SIP outcome.

2027 market prior
-----------------
Current small-cap research points to an earnings-led recovery: valuations have
cooled from euphoric levels, FY27/FY28 profit growth estimates are strong, and
rate cuts can help operating leverage. The risk is that the category is not
deeply cheap on absolute P/E, liquidity stress has risen in large schemes, and
high-beta recent winners can reverse quickly.

This script translates that prior into NAV-observable signals:
- recovery participation after corrections,
- benchmark-relative upside capture,
- disciplined downside capture and drawdown control,
- volatility normalization rather than volatility chasing,
- SIP-window consistency,
- AUM-aware confidence for liquidity/fund-size risk.

The final feature weights are partly learned from look-ahead-safe historical
12-month SIP XIRR windows, then blended with the above market prior.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


SECTOR = "Small Cap"
SUBSECTOR = "Small Cap Fund"
BENCHMARK_INDEX = "Small Cap"

SIP_MONTHS = 12
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 365.25
MIN_HISTORY_WEEKS = 52
MIN_PANEL_MONTHS = 24
MIN_PANEL_OBS = 60
MIN_CORR = 0.03
MAX_FEATURE_WEIGHT = 0.22

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_GPT.csv"

FEATURES = [
    "sip_hit_rate",
    "alpha_consistency",
    "momentum_quality",
    "recovery_strength",
    "swing_participation",
    "drawdown_control",
    "downside_resilience",
    "vol_normalization",
    "upside_capture",
    "cagr_stability",
    "aum_liquidity_score",
]

PRIOR_WEIGHTS = pd.Series(
    {
        "sip_hit_rate": 0.14,
        "alpha_consistency": 0.13,
        "momentum_quality": 0.11,
        "recovery_strength": 0.12,
        "swing_participation": 0.11,
        "drawdown_control": 0.11,
        "downside_resilience": 0.10,
        "vol_normalization": 0.07,
        "upside_capture": 0.07,
        "cagr_stability": 0.07,
        "aum_liquidity_score": 0.07,
    },
    dtype=float,
)
PRIOR_WEIGHTS = PRIOR_WEIGHTS / PRIOR_WEIGHTS.sum()


def clean_nav_chart(df: pd.DataFrame) -> pd.Series:
    """Return a sorted NAV series indexed by timestamp."""
    if df.empty or "timestamp" not in df or "nav" not in df:
        return pd.Series(dtype=float)

    out = df[["timestamp", "nav"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
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
    return nav.resample("D").ffill().dropna()


def to_weekly(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.resample("W-FRI").last().dropna()


def to_month_start(nav: pd.Series) -> pd.Series:
    daily = to_daily(nav)
    if daily.empty:
        return daily
    return daily.resample("MS").first().dropna()


def period_return(nav: pd.Series, periods: int) -> float:
    if len(nav) <= periods:
        return np.nan
    start = nav.iloc[-periods - 1]
    end = nav.iloc[-1]
    if start <= 0:
        return np.nan
    return float(end / start - 1.0)


def annualized_return(nav: pd.Series, periods: int, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    if len(nav) <= periods:
        return np.nan
    start = nav.iloc[-periods - 1]
    end = nav.iloc[-1]
    if start <= 0 or end <= 0:
        return np.nan
    years = periods / periods_per_year
    return float((end / start) ** (1.0 / years) - 1.0)


def calendar_cagr(nav: pd.Series, years: float) -> float:
    if nav.empty:
        return np.nan
    cutoff = nav.index[-1] - pd.Timedelta(days=int(years * DAYS_PER_YEAR))
    window = nav[nav.index >= cutoff]
    if len(window) < 2:
        return np.nan
    elapsed_years = (window.index[-1] - window.index[0]).days / DAYS_PER_YEAR
    if elapsed_years < years * 0.85:
        return np.nan
    if elapsed_years <= 0 or window.iloc[0] <= 0:
        return np.nan
    return float((window.iloc[-1] / window.iloc[0]) ** (1.0 / elapsed_years) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    running_max = nav.cummax()
    drawdowns = nav / running_max - 1.0
    return float(drawdowns.min())


def current_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    peak = nav.max()
    if peak <= 0:
        return np.nan
    return float(nav.iloc[-1] / peak - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 4:
        return np.nan
    return float(clean.std(ddof=0) * np.sqrt(WEEKS_PER_YEAR))


def downside_deviation(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    downside = clean[clean < 0]
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside**2).mean()) * np.sqrt(WEEKS_PER_YEAR))


def safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def capture_ratios(fund_ret: pd.Series, bench_ret: pd.Series, weeks: int = 104) -> Tuple[float, float]:
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna().tail(weeks)
    if aligned.empty:
        return np.nan, np.nan
    aligned.columns = ["fund", "bench"]
    up = aligned[aligned["bench"] > 0]
    down = aligned[aligned["bench"] < 0]
    up_capture = safe_ratio(up["fund"].mean(), up["bench"].mean()) if len(up) >= 4 else np.nan
    down_capture = safe_ratio(down["fund"].mean(), down["bench"].mean()) if len(down) >= 4 else np.nan
    return up_capture, down_capture


def rolling_alpha_consistency(fund_ret: pd.Series, bench_ret: pd.Series, weeks: int = 104) -> float:
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna().tail(weeks)
    if len(aligned) < 26:
        return np.nan
    aligned.columns = ["fund", "bench"]
    rolling_excess = (aligned["fund"] - aligned["bench"]).rolling(13).sum().dropna()
    if rolling_excess.empty:
        return np.nan
    return float((rolling_excess > 0).mean())


def npv(rate: float, cashflows: List[float], times_years: List[float]) -> float:
    return float(sum(cf / ((1.0 + rate) ** t) for cf, t in zip(cashflows, times_years)))


def xirr_bisect(cashflows: List[float], times_years: List[float]) -> float:
    lo = -0.95
    hi = 5.0
    f_lo = npv(lo, cashflows, times_years)
    f_hi = npv(hi, cashflows, times_years)
    if np.sign(f_lo) == np.sign(f_hi):
        hi = 20.0
        f_hi = npv(hi, cashflows, times_years)
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan

    for _ in range(90):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid, cashflows, times_years)
        if abs(f_mid) < 1e-9:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def sip_xirr(monthly_nav: pd.Series, start_pos: int, months: int = SIP_MONTHS) -> float:
    end_pos = start_pos + months
    if len(monthly_nav) <= end_pos:
        return np.nan

    window = monthly_nav.iloc[start_pos : end_pos + 1]
    if len(window) < months + 1 or (window <= 0).any():
        return np.nan

    buy_navs = window.iloc[:-1]
    sell_nav = window.iloc[-1]
    units = float((1.0 / buy_navs).sum())
    redemption = units * float(sell_nav)
    cashflows = [-1.0] * months + [redemption]
    start_date = window.index[0]
    dates = list(window.index[:-1]) + [window.index[-1]]
    times_years = [(date - start_date).days / DAYS_PER_YEAR for date in dates]
    return xirr_bisect(cashflows, times_years)


def sip_window_stats(fund_monthly: pd.Series, bench_monthly: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([fund_monthly, bench_monthly], axis=1, join="inner").dropna()
    if len(aligned) <= SIP_MONTHS:
        return {
            "sip_xirr_backtest_median": np.nan,
            "sip_xirr_backtest_p25": np.nan,
            "sip_hit_rate": np.nan,
        }

    aligned.columns = ["fund", "bench"]
    fund_xirrs = []
    bench_xirrs = []
    for start in range(0, len(aligned) - SIP_MONTHS):
        fund_xirrs.append(sip_xirr(aligned["fund"], start))
        bench_xirrs.append(sip_xirr(aligned["bench"], start))

    fund_s = pd.Series(fund_xirrs, dtype=float).dropna()
    bench_s = pd.Series(bench_xirrs, dtype=float).dropna()
    pair = pd.concat([fund_s, bench_s], axis=1, join="inner").dropna()
    if fund_s.empty:
        median = np.nan
        p25 = np.nan
    else:
        median = float(fund_s.median())
        p25 = float(fund_s.quantile(0.25))

    hit_rate = float((pair.iloc[:, 0] > pair.iloc[:, 1]).mean()) if not pair.empty else np.nan
    return {
        "sip_xirr_backtest_median": median,
        "sip_xirr_backtest_p25": p25,
        "sip_hit_rate": hit_rate,
    }


def recovery_strength(fund_weekly: pd.Series, bench_weekly: pd.Series) -> float:
    if len(fund_weekly) < 30 or len(bench_weekly) < 30:
        return np.nan

    bench_lookback = bench_weekly.tail(78)
    if len(bench_lookback) < 30:
        return np.nan
    trough_date = bench_lookback.idxmin()
    fund_after = fund_weekly[fund_weekly.index >= trough_date]
    bench_after = bench_weekly[bench_weekly.index >= trough_date]
    aligned = pd.concat([fund_after, bench_after], axis=1, join="inner").dropna()
    if len(aligned) < 8:
        return np.nan
    aligned.columns = ["fund", "bench"]
    bench_ret = aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1.0
    fund_ret = aligned["fund"].iloc[-1] / aligned["fund"].iloc[0] - 1.0
    return float(fund_ret - bench_ret)


def swing_participation(fund_ret: pd.Series, bench_weekly: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([fund_ret, bench_ret, bench_weekly], axis=1, join="inner").dropna()
    if len(aligned) < 52:
        return np.nan
    aligned.columns = ["fund_ret", "bench_ret", "bench_nav"]
    recent = aligned.tail(156)
    bench_drawdown = recent["bench_nav"] / recent["bench_nav"].cummax() - 1.0
    rebound_weeks = (bench_drawdown.shift(1) < -0.08) & (recent["bench_ret"] > 0.015)
    sample = recent[rebound_weeks]
    if len(sample) < 4:
        up_capture, _ = capture_ratios(fund_ret, bench_ret)
        return up_capture
    return safe_ratio(sample["fund_ret"].mean(), sample["bench_ret"].mean())


def aum_liquidity_scores(fund_meta: pd.DataFrame) -> pd.Series:
    aum = pd.to_numeric(fund_meta.set_index("mfId")["aum"], errors="coerce")
    valid = aum.dropna()
    if len(valid) < 5:
        return pd.Series(0.75, index=fund_meta["mfId"])

    log_aum = np.log1p(aum)
    q05, q25, q75, q95 = log_aum.dropna().quantile([0.05, 0.25, 0.75, 0.95])
    lower_span = max(q25 - q05, 1e-6)
    upper_span = max(q95 - q75, 1e-6)

    undersize_penalty = ((q25 - log_aum) / lower_span).clip(lower=0, upper=1) * 0.15
    oversize_penalty = ((log_aum - q75) / upper_span).clip(lower=0, upper=1) * 0.25
    score = (1.0 - undersize_penalty - oversize_penalty).clip(0.55, 1.0)
    return score.fillna(0.70)


def feature_snapshot(
    fund_nav: pd.Series,
    bench_nav: pd.Series,
    aum_score: float,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    if as_of is not None:
        fund_nav = fund_nav[fund_nav.index <= as_of]
        bench_nav = bench_nav[bench_nav.index <= as_of]
    fund_weekly = to_weekly(fund_nav)
    bench_weekly = to_weekly(bench_nav)
    aligned_nav = pd.concat([fund_weekly, bench_weekly], axis=1, join="inner").dropna()
    if len(aligned_nav) < MIN_HISTORY_WEEKS:
        return {}

    aligned_nav.columns = ["fund", "bench"]
    fund_weekly = aligned_nav["fund"]
    bench_weekly = aligned_nav["bench"]
    fund_ret = fund_weekly.pct_change().dropna()
    bench_ret = bench_weekly.pct_change().dropna()

    ret_13 = period_return(fund_weekly, 13)
    ret_26 = period_return(fund_weekly, 26)
    ret_52 = period_return(fund_weekly, 52)
    bench_ret_26 = period_return(bench_weekly, 26)
    vol_13 = annualized_volatility(fund_ret.tail(13))
    vol_52 = annualized_volatility(fund_ret.tail(52))
    dd_52 = max_drawdown(fund_weekly.tail(52))
    dd_156 = max_drawdown(fund_weekly.tail(156))
    current_dd = current_drawdown(fund_weekly.tail(156))
    up_capture, down_capture = capture_ratios(fund_ret, bench_ret)
    sip_stats = sip_window_stats(to_month_start(fund_nav), to_month_start(bench_nav))

    mom_quality = np.nanmean(
        [
            ret_13,
            ret_26,
            ret_52,
            (ret_26 - bench_ret_26) if pd.notna(ret_26) and pd.notna(bench_ret_26) else np.nan,
        ]
    )
    if pd.notna(vol_52) and vol_52 > 0:
        mom_quality = mom_quality / vol_52

    vol_regime = safe_ratio(vol_13, vol_52)
    downside = downside_deviation(fund_ret.tail(104))
    downside_resilience = -downside if pd.notna(downside) else np.nan
    if pd.notna(down_capture):
        downside_resilience = np.nanmean([downside_resilience, -down_capture])

    cagr_3y = annualized_return(fund_weekly, min(156, len(fund_weekly) - 1))
    cagr_5y = annualized_return(fund_weekly, min(260, len(fund_weekly) - 1))
    cagr_stability = cagr_3y - abs(cagr_3y - cagr_5y) if pd.notna(cagr_3y) and pd.notna(cagr_5y) else cagr_3y

    return {
        "sip_hit_rate": sip_stats["sip_hit_rate"],
        "alpha_consistency": rolling_alpha_consistency(fund_ret, bench_ret),
        "momentum_quality": mom_quality,
        "recovery_strength": recovery_strength(fund_weekly, bench_weekly),
        "swing_participation": swing_participation(fund_ret, bench_weekly, bench_ret),
        "drawdown_control": np.nanmean([dd_52, dd_156, current_dd]),
        "downside_resilience": downside_resilience,
        "vol_normalization": -vol_regime if pd.notna(vol_regime) else np.nan,
        "upside_capture": up_capture,
        "cagr_stability": cagr_stability,
        "aum_liquidity_score": aum_score,
        "sip_xirr_backtest_median": sip_stats["sip_xirr_backtest_median"],
        "sip_xirr_backtest_p25": sip_stats["sip_xirr_backtest_p25"],
        "up_capture": up_capture,
        "down_capture": down_capture,
        "vol_regime": vol_regime,
        "max_drawdown_1y": dd_52,
        "current_drawdown": current_dd,
    }


def build_training_panel(
    nav_by_fund: Dict[str, pd.Series],
    bench_nav: pd.Series,
    aum_scores: pd.Series,
) -> pd.DataFrame:
    rows = []
    bench_monthly = to_month_start(bench_nav)
    if len(bench_monthly) <= SIP_MONTHS + MIN_PANEL_MONTHS:
        return pd.DataFrame()

    for mf_id, fund_nav in nav_by_fund.items():
        fund_monthly = to_month_start(fund_nav)
        aligned_monthly = pd.concat([fund_monthly, bench_monthly], axis=1, join="inner").dropna()
        if len(aligned_monthly) <= SIP_MONTHS + MIN_PANEL_MONTHS:
            continue
        aligned_monthly.columns = ["fund", "bench"]

        start_idx = max(MIN_PANEL_MONTHS, 12)
        end_idx = len(aligned_monthly) - SIP_MONTHS
        for idx in range(start_idx, end_idx, 2):
            as_of = aligned_monthly.index[idx]
            features = feature_snapshot(
                fund_nav,
                bench_nav,
                float(aum_scores.get(mf_id, 0.70)),
                as_of=as_of,
            )
            if not features:
                continue
            target = sip_xirr(aligned_monthly["fund"], idx, SIP_MONTHS)
            bench_target = sip_xirr(aligned_monthly["bench"], idx, SIP_MONTHS)
            if pd.isna(target):
                continue
            row = {"mfId": mf_id, "as_of": as_of, "target_sip_xirr": target}
            row["target_alpha"] = target - bench_target if pd.notna(bench_target) else np.nan
            for feature in FEATURES:
                row[feature] = features.get(feature)
            rows.append(row)

    return pd.DataFrame(rows)


def learn_feature_weights(panel: pd.DataFrame) -> pd.Series:
    if len(panel) < MIN_PANEL_OBS:
        return PRIOR_WEIGHTS.copy()

    target_col = "target_alpha" if panel["target_alpha"].notna().sum() >= MIN_PANEL_OBS else "target_sip_xirr"
    learned = {}
    for feature in FEATURES:
        pair = panel[[feature, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < MIN_PANEL_OBS:
            continue
        corr = pair[feature].corr(pair[target_col], method="spearman")
        if pd.notna(corr) and corr > MIN_CORR:
            learned[feature] = corr

    if len(learned) < 4:
        return PRIOR_WEIGHTS.copy()

    learned_weights = pd.Series(learned, dtype=float)
    learned_weights = learned_weights / learned_weights.sum()
    learned_weights = learned_weights.reindex(FEATURES).fillna(0.0)

    blended = 0.60 * learned_weights + 0.40 * PRIOR_WEIGHTS
    blended = blended / blended.sum()
    return cap_feature_weights(blended)


def cap_feature_weights(weights: pd.Series) -> pd.Series:
    """Keep one backtested signal from overwhelming the research prior."""
    capped = weights.reindex(FEATURES).fillna(0.0).clip(upper=MAX_FEATURE_WEIGHT)
    excess = 1.0 - capped.sum()
    if excess > 1e-9:
        room = (MAX_FEATURE_WEIGHT - capped).clip(lower=0.0)
        if room.sum() > 1e-9:
            capped = capped + excess * room / room.sum()
    return capped / capped.sum()


def robust_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() < 2:
        return pd.Series(0.0, index=series.index)
    lo = clean.quantile(0.05)
    hi = clean.quantile(0.95)
    clipped = clean.clip(lo, hi)
    median = clipped.median()
    mad = (clipped - median).abs().median()
    if pd.isna(mad) or mad < 1e-9:
        std = clipped.std(ddof=0)
        if pd.isna(std) or std < 1e-9:
            return pd.Series(0.0, index=series.index)
        z = (clipped - clipped.mean()) / std
    else:
        z = (clipped - median) / (1.4826 * mad)
    return z.fillna(0.0).clip(-3.0, 3.0)


def score_funds(current: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    scored = current.copy()
    composite = pd.Series(0.0, index=scored.index)
    for feature in FEATURES:
        composite = composite + robust_zscore(scored[feature]) * float(weights.get(feature, 0.0))

    percentile = composite.rank(pct=True, method="average") * 100.0
    confidence = scored["confidence"].clip(0.50, 1.0)
    scored["raw_score"] = percentile
    scored["score"] = (percentile * (0.45 + 0.55 * confidence)).clip(0, 100)
    scored = scored.sort_values(["score", "data_days"], ascending=[False, False]).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def load_funds(provider: MfDataProvider) -> pd.DataFrame:
    all_funds = provider.list_all_mf()
    mask = (all_funds["sector"] == "Equity") & (all_funds["subsector"] == SUBSECTOR)
    funds = all_funds.loc[mask, ["mfId", "name", "aum", "sector", "subsector"]].copy()
    funds["aum"] = pd.to_numeric(funds["aum"], errors="coerce")
    return funds.sort_values("aum", ascending=False, na_position="last").reset_index(drop=True)


def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 78)
    print("  SMALL CAP MUTUAL FUND SCORING - GPT")
    print("  Target: 2027-ready 1Y monthly SIP returns")
    print("=" * 78)

    provider = MfDataProvider(date=date)
    funds = load_funds(provider)
    if funds.empty:
        raise RuntimeError(f"No {SUBSECTOR} funds found")

    print(f"  Funds discovered : {len(funds)}")
    print(f"  Benchmark        : {BENCHMARK_INDEX}")

    benchmark_nav = clean_nav_chart(provider.get_index_chart(BENCHMARK_INDEX))
    if benchmark_nav.empty:
        raise RuntimeError(f"No benchmark data found for {BENCHMARK_INDEX}")

    aum_scores = aum_liquidity_scores(funds)
    nav_by_fund: Dict[str, pd.Series] = {}
    rows = []

    for _, fund in funds.iterrows():
        mf_id = str(fund["mfId"])
        name = str(fund["name"])
        try:
            nav = clean_nav_chart(provider.get_mf_chart(mf_id))
        except Exception as exc:  # noqa: BLE001 - continue ranking other funds
            logger.warning("Skipping %s (%s): %s", name, mf_id, exc)
            continue
        if len(to_weekly(nav)) < MIN_HISTORY_WEEKS:
            logger.info("Skipping %s (%s): insufficient history", name, mf_id)
            continue

        nav_by_fund[mf_id] = nav
        features = feature_snapshot(nav, benchmark_nav, float(aum_scores.get(mf_id, 0.70)))
        if not features:
            continue

        data_days = int((nav.index[-1] - nav.index[0]).days + 1)
        history_years = data_days / DAYS_PER_YEAR
        history_confidence = min(1.0, max(0.55, history_years / 3.0))
        aum_confidence = float(aum_scores.get(mf_id, 0.70))
        confidence = min(1.0, max(0.50, 0.72 * history_confidence + 0.28 * aum_confidence))

        row = {
            "mfId": mf_id,
            "name": name,
            "aum": fund["aum"],
            "data_days": data_days,
            "history_years": history_years,
            "confidence": confidence,
            "cagr_3y": calendar_cagr(nav, 3.0),
            "cagr_5y": calendar_cagr(nav, 5.0),
        }
        row.update(features)
        rows.append(row)

    current = pd.DataFrame(rows)
    if current.empty:
        raise RuntimeError("No funds had enough NAV history to score")

    print("  Building look-ahead-safe SIP training panel...")
    panel = build_training_panel(nav_by_fund, benchmark_nav, aum_scores)
    weights = learn_feature_weights(panel)

    ranked = score_funds(current, weights)

    output_cols = [
        "mfId",
        "name",
        "rank",
        "score",
        "data_days",
        "cagr_3y",
        "cagr_5y",
        "sip_xirr_backtest_median",
        "sip_xirr_backtest_p25",
        "sip_hit_rate",
        "alpha_consistency",
        "momentum_quality",
        "recovery_strength",
        "swing_participation",
        "drawdown_control",
        "downside_resilience",
        "up_capture",
        "down_capture",
        "vol_regime",
        "max_drawdown_1y",
        "current_drawdown",
        "aum_liquidity_score",
        "confidence",
        "aum",
        "history_years",
    ]
    output = ranked[output_cols].copy()

    pct_cols = [
        "cagr_3y",
        "cagr_5y",
        "sip_xirr_backtest_median",
        "sip_xirr_backtest_p25",
        "recovery_strength",
        "drawdown_control",
        "downside_resilience",
        "max_drawdown_1y",
        "current_drawdown",
    ]
    ratio_cols = [
        "sip_hit_rate",
        "alpha_consistency",
        "momentum_quality",
        "swing_participation",
        "up_capture",
        "down_capture",
        "vol_regime",
        "aum_liquidity_score",
        "confidence",
        "history_years",
    ]

    output["score"] = output["score"].round(2)
    output["aum"] = pd.to_numeric(output["aum"], errors="coerce").round(2)
    for col in pct_cols:
        output[col] = (pd.to_numeric(output[col], errors="coerce") * 100.0).round(2)
    for col in ratio_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce").round(4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n  Learned/blended feature weights:")
    for feature, weight in weights.sort_values(ascending=False).items():
        print(f"    {feature:22s} {weight * 100:5.1f}%")

    print(f"\n  Training observations : {len(panel)}")
    print(f"  Results saved         : {OUTPUT_FILE}")
    print("\n  Top 10 funds:")
    display_cols = ["rank", "mfId", "name", "score", "cagr_3y", "cagr_5y", "confidence"]
    print(output[display_cols].head(10).to_string(index=False))
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small Cap MF screener (GPT)")
    parser.add_argument(
        "--date",
        default=None,
        help="Data snapshot date in YYYY-MM-DD format, e.g. 2026-04-20",
    )
    args = parser.parse_args()
    main(date=args.date)
