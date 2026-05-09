#!/usr/bin/env python3
"""
Multi Asset Mutual Fund Scoring Algorithm - GPT
===============================================

Rank Indian Multi Asset Allocation funds for the next 1-year monthly SIP
outcome using only observable NAV history.

The model is deliberately multi-asset specific. It does not assume that recent
1-year winners will keep winning. Instead, it scores whether a fund has shown:
- resilient 12-month SIP outcomes across rolling starts,
- useful participation in equity and precious-metal regimes,
- drawdown control when asset classes correct together,
- adaptive but not frantic exposure shifts inferred from NAV regressions,
- enough history/AUM stability to trust the estimate.

Historical feature weights are learned only from snapshots that precede the
forward 12-month SIP window, then shrunk toward conservative prior weights.
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


SECTOR = "Multi Asset"
SUBSECTOR = "Multi Asset Allocation Fund"

SIP_MONTHS = 12
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 365.25
RISK_FREE_RATE = 0.065
WEEKLY_RF = (1.0 + RISK_FREE_RATE) ** (1.0 / WEEKS_PER_YEAR) - 1.0

MIN_USABLE_WEEKS = 20
MIN_TRAINING_MONTHS = 24
MIN_TRAINING_OBS = 70
MAX_FEATURE_WEIGHT = 0.14
MIN_CORR = 0.025

OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / f"{SECTOR}_GPT.csv"

FEATURES = [
    "sip_p50",
    "sip_p25",
    "sip_hit_rate",
    "sip_consistency",
    "risk_on_alpha",
    "metal_capture",
    "mixed_alpha",
    "stress_alpha",
    "downside_protection",
    "sortino_3y",
    "calmar_3y",
    "drawdown_control",
    "ulcer_control",
    "cvar_control",
    "recovery_speed",
    "timing_alignment",
    "regime_fit",
    "allocation_balance",
    "exposure_stability",
    "durability_score",
]

PRIOR_WEIGHTS = pd.Series(
    {
        "sip_p50": 0.050,
        "sip_p25": 0.065,
        "sip_hit_rate": 0.045,
        "sip_consistency": 0.035,
        "risk_on_alpha": 0.055,
        "metal_capture": 0.055,
        "mixed_alpha": 0.050,
        "stress_alpha": 0.075,
        "downside_protection": 0.060,
        "sortino_3y": 0.055,
        "calmar_3y": 0.055,
        "drawdown_control": 0.055,
        "ulcer_control": 0.045,
        "cvar_control": 0.045,
        "recovery_speed": 0.035,
        "timing_alignment": 0.050,
        "regime_fit": 0.045,
        "allocation_balance": 0.040,
        "exposure_stability": 0.040,
        "durability_score": 0.045,
    },
    dtype=float,
)
PRIOR_WEIGHTS = PRIOR_WEIGHTS / PRIOR_WEIGHTS.sum()


def clean_nav_chart(df: pd.DataFrame) -> pd.Series:
    """Return a sorted positive NAV series indexed by UTC timestamp."""
    if df is None or df.empty or "timestamp" not in df or "nav" not in df:
        return pd.Series(dtype=float)

    out = df[["timestamp", "nav"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna(subset=["timestamp", "nav"])
    out = out[out["nav"] > 0]
    if out.empty:
        return pd.Series(dtype=float)

    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    series = out.set_index("timestamp")["nav"].astype(float)
    series.name = "nav"
    return series


def to_weekly(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.resample("W-MON").last().dropna()


def to_daily(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return nav
    return nav.resample("D").ffill().dropna()


def to_month_start(nav: pd.Series) -> pd.Series:
    daily = to_daily(nav)
    if daily.empty:
        return daily
    return daily.resample("MS").first().dropna()


def safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def period_return(nav: pd.Series, periods: int) -> float:
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
    cutoff = nav.index[-1] - pd.Timedelta(days=int(years * DAYS_PER_YEAR))
    window = nav[nav.index >= cutoff]
    if len(window) < 2:
        return np.nan
    elapsed_years = (window.index[-1] - window.index[0]).days / DAYS_PER_YEAR
    if elapsed_years < years * 0.85 or elapsed_years <= 0:
        return np.nan
    if window.iloc[0] <= 0 or window.iloc[-1] <= 0:
        return np.nan
    return float((window.iloc[-1] / window.iloc[0]) ** (1.0 / elapsed_years) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    drawdowns = nav / nav.cummax() - 1.0
    return float(drawdowns.min())


def current_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    peak = nav.max()
    if peak <= 0:
        return np.nan
    return float(nav.iloc[-1] / peak - 1.0)


def ulcer_index(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    drawdown_pct = (nav / nav.cummax() - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(np.minimum(drawdown_pct, 0.0)))))


def average_recovery_weeks(nav: pd.Series) -> float:
    if len(nav) < 8:
        return np.nan
    drawdown = nav / nav.cummax() - 1.0
    durations = []
    active = 0
    for value in drawdown:
        if value < -1e-9:
            active += 1
        elif active > 0:
            durations.append(active)
            active = 0
    if active > 0:
        durations.append(active)
    if not durations:
        return 0.0
    return float(np.mean(durations))


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
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(WEEKS_PER_YEAR))


def sortino_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 12:
        return np.nan
    excess = clean - WEEKLY_RF
    dd = downside_deviation(excess)
    if pd.isna(dd) or dd <= 1e-12:
        return np.nan
    annual_excess = float(excess.mean() * WEEKS_PER_YEAR)
    return annual_excess / dd


def cvar_5pct(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 20:
        return np.nan
    cutoff = clean.quantile(0.05)
    tail = clean[clean <= cutoff]
    if tail.empty:
        return np.nan
    return float(tail.mean())


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
    sell_nav = float(window.iloc[-1])
    units = float((1.0 / buy_navs).sum())
    redemption = units * sell_nav
    cashflows = [-1.0] * months + [redemption]
    start_date = window.index[0]
    dates = list(window.index[:-1]) + [window.index[-1]]
    times_years = [(date - start_date).days / DAYS_PER_YEAR for date in dates]
    return xirr_bisect(cashflows, times_years)


def rolling_sip_xirrs(monthly_nav: pd.Series, months: int = SIP_MONTHS) -> pd.Series:
    values = {}
    if len(monthly_nav) <= months:
        return pd.Series(dtype=float)
    for start in range(len(monthly_nav) - months):
        end_date = monthly_nav.index[start + months]
        values[end_date] = sip_xirr(monthly_nav, start, months)
    return pd.Series(values, dtype=float)


def sip_window_stats(fund_nav: pd.Series, baseline_nav: pd.Series) -> Dict[str, float]:
    aligned = pd.concat(
        [to_month_start(fund_nav), to_month_start(baseline_nav)],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) <= SIP_MONTHS:
        return {
            "sip_p50": np.nan,
            "sip_p25": np.nan,
            "sip_hit_rate": np.nan,
            "sip_consistency": np.nan,
            "sip_latest": np.nan,
            "n_sip_windows": 0,
        }

    aligned.columns = ["fund", "baseline"]
    fund_xirr = rolling_sip_xirrs(aligned["fund"])
    base_xirr = rolling_sip_xirrs(aligned["baseline"])
    pair = pd.concat([fund_xirr, base_xirr], axis=1, join="inner").dropna()
    fund_clean = fund_xirr.dropna()
    if fund_clean.empty:
        return {
            "sip_p50": np.nan,
            "sip_p25": np.nan,
            "sip_hit_rate": np.nan,
            "sip_consistency": np.nan,
            "sip_latest": np.nan,
            "n_sip_windows": 0,
        }

    dispersion = fund_clean.std(ddof=0)
    consistency = safe_ratio(fund_clean.mean(), dispersion) if dispersion and dispersion > 0 else np.nan
    return {
        "sip_p50": float(fund_clean.median()),
        "sip_p25": float(fund_clean.quantile(0.25)),
        "sip_hit_rate": float((pair.iloc[:, 0] > pair.iloc[:, 1]).mean()) if not pair.empty else np.nan,
        "sip_consistency": consistency,
        "sip_latest": float(fund_clean.iloc[-1]),
        "n_sip_windows": int(len(fund_clean)),
    }


def load_first_nav(
    provider: MfDataProvider,
    choices: List[Tuple[str, str]],
    min_points: int = 30,
) -> Tuple[str, pd.Series]:
    for source, identifier in choices:
        try:
            if source == "index":
                nav = clean_nav_chart(provider.get_index_chart(identifier))
            else:
                nav = clean_nav_chart(provider.get_mf_chart(identifier))
            weekly = to_weekly(nav)
            if len(weekly) >= min_points:
                return identifier, weekly
        except Exception as exc:  # noqa: BLE001 - try next proxy fallback
            logger.warning("Proxy %s %s unavailable: %s", source, identifier, exc)
    return "", pd.Series(dtype=float)


def load_asset_proxies(provider: MfDataProvider) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    proxy_ids: Dict[str, str] = {}
    proxy_navs: Dict[str, pd.Series] = {}

    eq_id, equity = load_first_nav(
        provider,
        [("index", "Total Market"), ("index", "_NIFTY500"), ("index", "Large Cap"), ("index", ".NSEI")],
    )
    if equity.empty:
        raise RuntimeError("No usable equity proxy found")
    proxy_ids["equity"] = eq_id
    proxy_navs["equity"] = equity

    gold_id, gold = load_first_nav(provider, [("index", "Gold"), ("index", "GBES"), ("mf", "M_SBIGL")])
    if gold.empty:
        raise RuntimeError("No usable gold proxy found")
    proxy_ids["gold"] = gold_id
    proxy_navs["gold"] = gold

    silver_id, silver = load_first_nav(provider, [("mf", "M_ICPVF")], min_points=24)
    if silver.empty:
        logger.warning("Silver proxy unavailable; metal sleeve will use gold only")
    else:
        proxy_ids["silver"] = silver_id
        proxy_navs["silver"] = silver

    return proxy_navs, proxy_ids


def proxy_return_frame(proxy_navs: Dict[str, pd.Series]) -> pd.DataFrame:
    returns = {
        key: nav.pct_change().rename(key)
        for key, nav in proxy_navs.items()
        if key in {"equity", "gold", "silver"} and not nav.empty
    }
    return pd.concat(returns.values(), axis=1, join="outer").sort_index()


def build_baseline_nav(proxy_navs: Dict[str, pd.Series]) -> pd.Series:
    returns = proxy_return_frame(proxy_navs)
    if "silver" in returns:
        weights = {"equity": 0.58, "gold": 0.17, "silver": 0.10}
        cash_weight = 0.15
    else:
        weights = {"equity": 0.60, "gold": 0.25}
        cash_weight = 0.15

    available = [col for col in weights if col in returns]
    aligned = returns[available].dropna(how="any")
    if aligned.empty:
        return pd.Series(dtype=float)

    asset_weight_sum = sum(weights[col] for col in available)
    scale = (1.0 - cash_weight) / asset_weight_sum if asset_weight_sum > 0 else 0.0
    blend = pd.Series(WEEKLY_RF * cash_weight, index=aligned.index, dtype=float)
    for col in available:
        blend = blend + aligned[col] * weights[col] * scale
    nav = (1.0 + blend).cumprod() * 100.0
    nav.name = "baseline"
    return nav


def classify_regimes(proxy_rets: pd.DataFrame) -> pd.DataFrame:
    if proxy_rets.empty or "equity" not in proxy_rets:
        return pd.DataFrame(index=proxy_rets.index)

    out = pd.DataFrame(index=proxy_rets.index)
    equity_mom = (1.0 + proxy_rets["equity"].fillna(0.0)).rolling(13).apply(np.prod, raw=True) - 1.0
    metals_cols = [col for col in ["gold", "silver"] if col in proxy_rets]
    if metals_cols:
        metal_ret = proxy_rets[metals_cols].mean(axis=1)
        metal_mom = (1.0 + metal_ret.fillna(0.0)).rolling(13).apply(np.prod, raw=True) - 1.0
    else:
        metal_ret = pd.Series(np.nan, index=proxy_rets.index)
        metal_mom = pd.Series(np.nan, index=proxy_rets.index)

    eq_vol = proxy_rets["equity"].rolling(13).std()
    stress_threshold = eq_vol.rolling(104, min_periods=20).quantile(0.75)

    out["risk_on"] = (equity_mom > 0.04) & (equity_mom >= metal_mom.fillna(-np.inf))
    out["metal_bid"] = (metal_mom > 0.04) & (metal_mom > equity_mom.fillna(-np.inf))
    out["stress"] = (equity_mom < -0.05) | ((proxy_rets["equity"] < -0.025) & (eq_vol > stress_threshold))
    out["mixed"] = ~(out["risk_on"] | out["metal_bid"] | out["stress"])
    out["metal_ret"] = metal_ret
    return out


def capture_ratio(fund_ret: pd.Series, bench_ret: pd.Series, mask: pd.Series) -> float:
    aligned = pd.concat([fund_ret, bench_ret, mask.rename("mask")], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    aligned.columns = ["fund", "bench", "mask"]
    sample = aligned[aligned["mask"].astype(bool)]
    if len(sample) < 4:
        return np.nan
    return safe_ratio(sample["fund"].mean(), sample["bench"].mean())


def annualized_alpha(fund_ret: pd.Series, baseline_ret: pd.Series, mask: pd.Series) -> float:
    aligned = pd.concat([fund_ret, baseline_ret, mask.rename("mask")], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    aligned.columns = ["fund", "baseline", "mask"]
    sample = aligned[aligned["mask"].astype(bool)]
    if len(sample) < 4:
        return np.nan
    return float((sample["fund"] - sample["baseline"]).mean() * WEEKS_PER_YEAR)


def fit_exposures(fund_ret: pd.Series, proxy_rets: pd.DataFrame) -> Dict[str, float]:
    cols = [col for col in ["equity", "gold", "silver"] if col in proxy_rets]
    aligned = pd.concat([fund_ret.rename("fund"), proxy_rets[cols]], axis=1, join="inner").dropna()
    if len(aligned) < 12 or not cols:
        return {"equity": np.nan, "gold": np.nan, "silver": np.nan, "defensive": np.nan}

    y = aligned["fund"].to_numpy(dtype=float) - WEEKLY_RF
    x = aligned[cols].to_numpy(dtype=float) - WEEKLY_RF
    if x.ndim != 2 or x.shape[0] <= x.shape[1]:
        return {"equity": np.nan, "gold": np.nan, "silver": np.nan, "defensive": np.nan}

    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"equity": np.nan, "gold": np.nan, "silver": np.nan, "defensive": np.nan}

    exposure = pd.Series(beta, index=cols, dtype=float).clip(lower=0.0, upper=1.2)
    total = float(exposure.sum())
    if total > 1.0:
        exposure = exposure / total
        total = 1.0

    out = {"equity": 0.0, "gold": 0.0, "silver": 0.0}
    for col, value in exposure.items():
        out[col] = float(value)
    out["defensive"] = float(max(0.0, 1.0 - total))
    return out


def ideal_weights_from_momentum(proxy_rets: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, float]:
    hist = proxy_rets[proxy_rets.index <= as_of].tail(13)
    if len(hist) < 6:
        return {"equity": 0.50, "gold": 0.20, "silver": 0.05, "defensive": 0.25}

    momentum = {}
    for col in ["equity", "gold", "silver"]:
        if col in hist:
            momentum[col] = float((1.0 + hist[col].dropna()).prod() - 1.0)
    positive = {key: max(0.0, value) for key, value in momentum.items()}
    pos_sum = sum(positive.values())
    if pos_sum <= 1e-12:
        return {"equity": 0.25, "gold": 0.20, "silver": 0.05 if "silver" in momentum else 0.0, "defensive": 0.50}

    ideal = {"equity": 0.0, "gold": 0.0, "silver": 0.0, "defensive": 0.15}
    for key, value in positive.items():
        ideal[key] = 0.85 * value / pos_sum
    return ideal


def exposure_alignment(exposure: Dict[str, float], ideal: Dict[str, float]) -> float:
    keys = ["equity", "gold", "silver", "defensive"]
    if any(pd.isna(exposure.get(key, np.nan)) for key in keys):
        return np.nan
    distance = sum(abs(float(exposure.get(key, 0.0)) - float(ideal.get(key, 0.0))) for key in keys)
    return float(max(0.0, 1.0 - 0.5 * distance))


def rolling_exposure_metrics(fund_ret: pd.Series, proxy_rets: pd.DataFrame) -> Dict[str, float]:
    aligned = pd.concat([fund_ret.rename("fund"), proxy_rets], axis=1, join="inner").dropna(subset=["fund"])
    if len(aligned) < 30:
        current = fit_exposures(fund_ret, proxy_rets)
        ideal = ideal_weights_from_momentum(proxy_rets, fund_ret.index.max()) if len(fund_ret) else {}
        return {
            "eq_weight": current.get("equity", np.nan),
            "gold_weight": current.get("gold", np.nan),
            "silver_weight": current.get("silver", np.nan),
            "defensive_weight": current.get("defensive", np.nan),
            "timing_alignment": np.nan,
            "regime_fit": exposure_alignment(current, ideal),
            "allocation_balance": allocation_balance(current),
            "exposure_stability": np.nan,
        }

    rows = []
    window = min(52, max(20, len(aligned) // 2))
    for end in range(window, len(aligned) + 1, 4):
        frame = aligned.iloc[end - window : end]
        exposures = fit_exposures(frame["fund"], frame[[c for c in ["equity", "gold", "silver"] if c in frame]])
        exposures["date"] = frame.index[-1]
        exposures["alignment"] = exposure_alignment(exposures, ideal_weights_from_momentum(proxy_rets, frame.index[-1]))
        rows.append(exposures)

    exp_df = pd.DataFrame(rows).set_index("date")
    current = exp_df.iloc[-1].to_dict()
    exposure_cols = [col for col in ["equity", "gold", "silver", "defensive"] if col in exp_df]
    turnover = exp_df[exposure_cols].diff().abs().sum(axis=1).dropna()
    exposure_stability = float(np.clip(1.0 - turnover.mean(), 0.0, 1.0)) if not turnover.empty else np.nan

    out = {
        "eq_weight": float(current.get("equity", np.nan)),
        "gold_weight": float(current.get("gold", np.nan)),
        "silver_weight": float(current.get("silver", np.nan)),
        "defensive_weight": float(current.get("defensive", np.nan)),
        "timing_alignment": float(exp_df["alignment"].tail(13).mean()) if "alignment" in exp_df else np.nan,
        "regime_fit": float(current.get("alignment", np.nan)),
        "allocation_balance": allocation_balance(current),
        "exposure_stability": exposure_stability,
    }
    return out


def allocation_balance(exposure: Dict[str, float]) -> float:
    values = [
        float(exposure.get("equity", 0.0) or 0.0),
        float(exposure.get("gold", 0.0) or 0.0),
        float(exposure.get("silver", 0.0) or 0.0),
        float(exposure.get("defensive", 0.0) or 0.0),
    ]
    if any(pd.isna(value) for value in values):
        return np.nan
    total = sum(values)
    if total <= 1e-12:
        return np.nan
    weights = np.array(values, dtype=float) / total
    hhi = float(np.square(weights).sum())
    return float(np.clip((1.0 - hhi) / 0.75, 0.0, 1.0))


def aum_quality_scores(funds: pd.DataFrame) -> pd.Series:
    aum = pd.to_numeric(funds.set_index("mfId")["aum"], errors="coerce")
    valid = aum.dropna()
    if len(valid) < 5:
        return pd.Series(0.72, index=funds["mfId"])

    log_aum = np.log1p(aum)
    q05, q25, q75, q95 = log_aum.dropna().quantile([0.05, 0.25, 0.75, 0.95])
    lower_span = max(q25 - q05, 1e-6)
    upper_span = max(q95 - q75, 1e-6)
    undersize_penalty = ((q25 - log_aum) / lower_span).clip(lower=0.0, upper=1.0) * 0.18
    oversize_penalty = ((log_aum - q75) / upper_span).clip(lower=0.0, upper=1.0) * 0.12
    return (1.0 - undersize_penalty - oversize_penalty).clip(0.60, 1.0).fillna(0.70)


def feature_snapshot(
    fund_nav: pd.Series,
    proxy_navs: Dict[str, pd.Series],
    baseline_nav: pd.Series,
    aum_score: float,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    if as_of is not None:
        fund_nav = fund_nav[fund_nav.index <= as_of]
        proxy_navs = {key: nav[nav.index <= as_of] for key, nav in proxy_navs.items()}
        baseline_nav = baseline_nav[baseline_nav.index <= as_of]

    fund_weekly = to_weekly(fund_nav)
    if len(fund_weekly) < 8 or baseline_nav.empty:
        return {}

    proxy_rets = proxy_return_frame(proxy_navs)
    regimes = classify_regimes(proxy_rets)
    fund_ret = fund_weekly.pct_change().dropna()
    baseline_ret = baseline_nav.pct_change().dropna()

    recent_weeks = min(156, len(fund_weekly))
    fund_recent = fund_weekly.tail(recent_weeks)
    ret_recent = fund_ret.tail(max(1, recent_weeks - 1))
    proxy_recent = proxy_rets[proxy_rets.index <= fund_weekly.index[-1]].tail(max(1, recent_weeks - 1))
    regime_recent = regimes[regimes.index <= fund_weekly.index[-1]].tail(max(1, recent_weeks - 1))
    base_recent = baseline_ret[baseline_ret.index <= fund_weekly.index[-1]].tail(max(1, recent_weeks - 1))

    sip_stats = sip_window_stats(fund_nav, baseline_nav)
    exposure_stats = rolling_exposure_metrics(fund_ret, proxy_rets)

    eq_ret = proxy_recent["equity"] if "equity" in proxy_recent else pd.Series(dtype=float)
    eq_up_capture = capture_ratio(ret_recent, eq_ret, eq_ret > 0)
    eq_down_capture = capture_ratio(ret_recent, eq_ret, eq_ret < 0)

    if "metal_ret" in regime_recent:
        metal_ret = regime_recent["metal_ret"]
    else:
        metal_cols = [col for col in ["gold", "silver"] if col in proxy_recent]
        metal_ret = proxy_recent[metal_cols].mean(axis=1) if metal_cols else pd.Series(dtype=float)
    metal_capture = capture_ratio(ret_recent, metal_ret, regime_recent.get("metal_bid", pd.Series(False, index=regime_recent.index)))

    risk_on_alpha = annualized_alpha(
        ret_recent,
        base_recent,
        regime_recent.get("risk_on", pd.Series(False, index=regime_recent.index)),
    )
    mixed_alpha = annualized_alpha(
        ret_recent,
        base_recent,
        regime_recent.get("mixed", pd.Series(False, index=regime_recent.index)),
    )
    stress_alpha = annualized_alpha(
        ret_recent,
        base_recent,
        regime_recent.get("stress", pd.Series(False, index=regime_recent.index)),
    )

    ann_3y = calendar_cagr(fund_nav, 3.0)
    dd_3y = max_drawdown(fund_recent)
    calmar = safe_ratio(ann_3y, abs(dd_3y)) if pd.notna(ann_3y) and pd.notna(dd_3y) and dd_3y < 0 else np.nan
    ulcer = ulcer_index(fund_recent)
    cvar = cvar_5pct(ret_recent)
    recovery_weeks = average_recovery_weeks(fund_recent)

    history_years = (fund_nav.index[-1] - fund_nav.index[0]).days / DAYS_PER_YEAR if len(fund_nav) >= 2 else 0.0
    history_score = float(np.clip(history_years / 3.0, 0.25, 1.0))
    metric_coverage = np.mean(
        [
            pd.notna(sip_stats["sip_p50"]),
            pd.notna(risk_on_alpha),
            pd.notna(stress_alpha),
            pd.notna(eq_down_capture),
            pd.notna(exposure_stats.get("regime_fit")),
            pd.notna(ann_3y),
        ]
    )
    durability = float(np.clip(0.45 * history_score + 0.25 * aum_score + 0.30 * metric_coverage, 0.20, 1.0))

    output = {
        "sip_p50": sip_stats["sip_p50"],
        "sip_p25": sip_stats["sip_p25"],
        "sip_hit_rate": sip_stats["sip_hit_rate"],
        "sip_consistency": sip_stats["sip_consistency"],
        "risk_on_alpha": risk_on_alpha,
        "metal_capture": metal_capture,
        "mixed_alpha": mixed_alpha,
        "stress_alpha": stress_alpha,
        "downside_protection": -eq_down_capture if pd.notna(eq_down_capture) else np.nan,
        "sortino_3y": sortino_ratio(ret_recent),
        "calmar_3y": calmar,
        "drawdown_control": dd_3y,
        "ulcer_control": -ulcer if pd.notna(ulcer) else np.nan,
        "cvar_control": cvar,
        "recovery_speed": safe_ratio(1.0, 1.0 + recovery_weeks) if pd.notna(recovery_weeks) else np.nan,
        "timing_alignment": exposure_stats["timing_alignment"],
        "regime_fit": exposure_stats["regime_fit"],
        "allocation_balance": exposure_stats["allocation_balance"],
        "exposure_stability": exposure_stats["exposure_stability"],
        "durability_score": durability,
        "sip_latest": sip_stats["sip_latest"],
        "n_sip_windows": sip_stats["n_sip_windows"],
        "eq_up_capture": eq_up_capture,
        "eq_down_capture": eq_down_capture,
        "max_drawdown_3y": dd_3y,
        "current_drawdown": current_drawdown(fund_recent),
        "ulcer_index": ulcer,
        "cvar_5pct": cvar,
        "recovery_weeks": recovery_weeks,
        "data_weeks": int(len(fund_weekly)),
    }
    output.update(exposure_stats)
    return output


def short_history_features(nav: pd.Series, aum_score: float) -> Dict[str, float]:
    """Emit an explicit low-confidence row when history is too short to model."""
    weekly = to_weekly(nav)
    out = {feature: np.nan for feature in FEATURES}
    out.update(
        {
            "sip_latest": np.nan,
            "n_sip_windows": 0,
            "eq_up_capture": np.nan,
            "eq_down_capture": np.nan,
            "max_drawdown_3y": max_drawdown(weekly),
            "current_drawdown": current_drawdown(weekly),
            "ulcer_index": ulcer_index(weekly),
            "cvar_5pct": np.nan,
            "recovery_weeks": average_recovery_weeks(weekly),
            "eq_weight": np.nan,
            "gold_weight": np.nan,
            "silver_weight": np.nan,
            "defensive_weight": np.nan,
            "timing_alignment": np.nan,
            "regime_fit": np.nan,
            "allocation_balance": np.nan,
            "exposure_stability": np.nan,
            "durability_score": float(np.clip(0.20 + 0.10 * aum_score, 0.20, 0.32)),
            "data_weeks": int(len(weekly)),
        }
    )
    return out


def build_training_panel(
    nav_by_fund: Dict[str, pd.Series],
    proxy_navs: Dict[str, pd.Series],
    baseline_nav: pd.Series,
    aum_scores: pd.Series,
) -> pd.DataFrame:
    rows = []
    baseline_monthly = to_month_start(baseline_nav)
    if len(baseline_monthly) <= MIN_TRAINING_MONTHS + SIP_MONTHS:
        return pd.DataFrame()

    for mf_id, fund_nav in nav_by_fund.items():
        fund_monthly = to_month_start(fund_nav)
        aligned = pd.concat([fund_monthly, baseline_monthly], axis=1, join="inner").dropna()
        if len(aligned) <= MIN_TRAINING_MONTHS + SIP_MONTHS:
            continue
        aligned.columns = ["fund", "baseline"]

        start_idx = max(MIN_TRAINING_MONTHS, 12)
        end_idx = len(aligned) - SIP_MONTHS
        for idx in range(start_idx, end_idx, 2):
            as_of = aligned.index[idx]
            features = feature_snapshot(
                fund_nav,
                proxy_navs,
                baseline_nav,
                float(aum_scores.get(mf_id, 0.70)),
                as_of=as_of,
            )
            if not features:
                continue
            target = sip_xirr(aligned["fund"], idx, SIP_MONTHS)
            baseline_target = sip_xirr(aligned["baseline"], idx, SIP_MONTHS)
            if pd.isna(target):
                continue

            row = {
                "mfId": mf_id,
                "as_of": as_of,
                "target_sip_xirr": target,
                "target_excess": target - baseline_target if pd.notna(baseline_target) else np.nan,
            }
            for feature in FEATURES:
                row[feature] = features.get(feature)
            rows.append(row)

    return pd.DataFrame(rows)


def cap_feature_weights(weights: pd.Series) -> pd.Series:
    capped = weights.reindex(FEATURES).fillna(0.0).clip(lower=0.0, upper=MAX_FEATURE_WEIGHT)
    for _ in range(10):
        total = capped.sum()
        if abs(total - 1.0) < 1e-9:
            break
        room = (MAX_FEATURE_WEIGHT - capped).clip(lower=0.0)
        if room.sum() <= 1e-12:
            break
        capped = capped + (1.0 - total) * room / room.sum()
        capped = capped.clip(lower=0.0, upper=MAX_FEATURE_WEIGHT)
    if capped.sum() <= 1e-12:
        return PRIOR_WEIGHTS.copy()
    return capped / capped.sum()


def learn_feature_weights(panel: pd.DataFrame) -> pd.Series:
    if len(panel) < MIN_TRAINING_OBS:
        return PRIOR_WEIGHTS.copy()

    target_col = "target_excess" if panel.get("target_excess", pd.Series(dtype=float)).notna().sum() >= MIN_TRAINING_OBS else "target_sip_xirr"
    learned = {}
    for feature in FEATURES:
        pair = panel[[feature, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < MIN_TRAINING_OBS:
            continue
        corr = pair[feature].corr(pair[target_col], method="spearman")
        if pd.notna(corr) and corr > MIN_CORR:
            learned[feature] = corr

    if len(learned) < 5:
        return PRIOR_WEIGHTS.copy()

    learned_weights = pd.Series(learned, dtype=float)
    learned_weights = learned_weights / learned_weights.sum()
    learned_weights = learned_weights.reindex(FEATURES).fillna(0.0)
    blended = 0.55 * learned_weights + 0.45 * PRIOR_WEIGHTS
    return cap_feature_weights(blended)


def percentile_scores(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index)
    lo = clean.quantile(0.05)
    hi = clean.quantile(0.95)
    clipped = clean.clip(lo, hi)
    return clipped.rank(pct=True, method="average") * 100.0


def score_funds(current: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    scored = current.copy()
    composite = pd.Series(0.0, index=scored.index, dtype=float)
    available_weight = pd.Series(0.0, index=scored.index, dtype=float)

    for feature in FEATURES:
        pctl = percentile_scores(scored[feature])
        mask = pctl.notna()
        weight = float(weights.get(feature, 0.0))
        composite.loc[mask] += pctl.loc[mask] * weight
        available_weight.loc[mask] += weight

    scored["raw_score"] = np.where(available_weight > 0, composite / available_weight, 0.0)
    low_coverage = available_weight < 0.35
    no_coverage = available_weight < 0.10
    scored.loc[low_coverage, "raw_score"] = scored.loc[low_coverage, "raw_score"].clip(upper=45.0)
    scored.loc[no_coverage, "raw_score"] = 0.0

    confidence = scored["confidence"].clip(0.35, 1.0)
    scored["score"] = (scored["raw_score"] * (0.35 + 0.65 * confidence)).clip(0.0, 100.0)
    scored = scored.sort_values(["score", "data_days"], ascending=[False, False]).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1, dtype=int)
    return scored


def load_funds(provider: MfDataProvider) -> pd.DataFrame:
    all_funds = provider.list_all_mf()
    mask = (all_funds["sector"] == "Hybrid") & (all_funds["subsector"] == SUBSECTOR)
    funds = all_funds.loc[mask, ["mfId", "name", "aum", "sector", "subsector"]].copy()
    funds["aum"] = pd.to_numeric(funds["aum"], errors="coerce")
    return funds.sort_values("aum", ascending=False, na_position="last").reset_index(drop=True)


def format_output(ranked: pd.DataFrame) -> pd.DataFrame:
    output_cols = [
        "mfId",
        "name",
        "rank",
        "score",
        "data_days",
        "cagr_3y",
        "cagr_5y",
        "sip_p50",
        "sip_p25",
        "sip_latest",
        "sip_hit_rate",
        "sip_consistency",
        "n_sip_windows",
        "risk_on_alpha",
        "metal_capture",
        "mixed_alpha",
        "stress_alpha",
        "eq_up_capture",
        "eq_down_capture",
        "sortino_3y",
        "calmar_3y",
        "max_drawdown_3y",
        "current_drawdown",
        "ulcer_index",
        "cvar_5pct",
        "recovery_weeks",
        "eq_weight",
        "gold_weight",
        "silver_weight",
        "defensive_weight",
        "timing_alignment",
        "regime_fit",
        "allocation_balance",
        "exposure_stability",
        "durability_score",
        "confidence",
        "aum",
        "data_weeks",
        "history_years",
    ]

    out = ranked[output_cols].copy()
    pct_cols = [
        "cagr_3y",
        "cagr_5y",
        "sip_p50",
        "sip_p25",
        "sip_latest",
        "risk_on_alpha",
        "mixed_alpha",
        "stress_alpha",
        "max_drawdown_3y",
        "current_drawdown",
        "cvar_5pct",
    ]
    ratio_cols = [
        "sip_hit_rate",
        "sip_consistency",
        "metal_capture",
        "eq_up_capture",
        "eq_down_capture",
        "sortino_3y",
        "calmar_3y",
        "ulcer_index",
        "recovery_weeks",
        "eq_weight",
        "gold_weight",
        "silver_weight",
        "defensive_weight",
        "timing_alignment",
        "regime_fit",
        "allocation_balance",
        "exposure_stability",
        "durability_score",
        "confidence",
        "history_years",
    ]

    out["score"] = pd.to_numeric(out["score"], errors="coerce").round(2)
    out["aum"] = pd.to_numeric(out["aum"], errors="coerce").round(2)
    for col in pct_cols:
        out[col] = (pd.to_numeric(out[col], errors="coerce") * 100.0).round(2)
    for col in ratio_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def main(date: Optional[str] = None) -> None:
    print("\n" + "=" * 78)
    print("  MULTI ASSET MUTUAL FUND SCORING - GPT")
    print("  Target: next 1Y monthly SIP with regime-aware risk control")
    print("=" * 78)

    provider = MfDataProvider(date=date)
    funds = load_funds(provider)
    if funds.empty:
        raise RuntimeError(f"No {SUBSECTOR} funds found")

    proxy_navs, proxy_ids = load_asset_proxies(provider)
    baseline_nav = build_baseline_nav(proxy_navs)
    if baseline_nav.empty:
        raise RuntimeError("Unable to build neutral multi-asset baseline")

    print(f"  Funds discovered : {len(funds)}")
    print(f"  Equity proxy     : {proxy_ids.get('equity', 'N/A')}")
    print(f"  Gold proxy       : {proxy_ids.get('gold', 'N/A')}")
    print(f"  Silver proxy     : {proxy_ids.get('silver', 'disabled')}")

    aum_scores = aum_quality_scores(funds)
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

        weekly = to_weekly(nav)
        if len(weekly) < MIN_USABLE_WEEKS:
            logger.info("Emitting low-confidence row for %s (%s): only %s weekly points", name, mf_id, len(weekly))

        aum_score = float(aum_scores.get(mf_id, 0.70))
        features = (
            feature_snapshot(nav, proxy_navs, baseline_nav, aum_score)
            if len(weekly) >= MIN_USABLE_WEEKS
            else short_history_features(nav, aum_score)
        )
        if not features:
            logger.info("Using low-confidence fallback for %s (%s): feature coverage too sparse", name, mf_id)
            features = short_history_features(nav, aum_score)

        if len(weekly) >= MIN_USABLE_WEEKS:
            nav_by_fund[mf_id] = nav

        data_days = int((nav.index[-1] - nav.index[0]).days + 1)
        history_years = data_days / DAYS_PER_YEAR
        history_confidence = float(np.clip(history_years / 3.0, 0.35, 1.0))
        metric_confidence = float(np.clip(features.get("durability_score", 0.50), 0.20, 1.0))
        confidence = float(np.clip(0.58 * history_confidence + 0.24 * aum_score + 0.18 * metric_confidence, 0.35, 1.0))

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
    panel = build_training_panel(nav_by_fund, proxy_navs, baseline_nav, aum_scores)
    weights = learn_feature_weights(panel)

    ranked = score_funds(current, weights)
    output = format_output(ranked)

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
    parser = argparse.ArgumentParser(description="Multi Asset MF screener (GPT)")
    parser.add_argument(
        "--date",
        default=None,
        help="Data snapshot date in YYYY-MM-DD format, e.g. 2026-05-09",
    )
    args = parser.parse_args()
    main(date=args.date)
