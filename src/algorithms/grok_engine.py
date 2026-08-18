#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok SIP-then-hold engine
=========================

Scores Indian mutual funds for a 24-month cashflow:
  12 monthly SIPs -> 12-month hold (no contributions) -> full exit at month 24.
The metric is the XIRR of that hybrid path.

Design notes (why this exists)
------------------------------
Older Grok screens optimized 1Y SIP XIRR (or, for Total Market, raw rolling
SIP-hold means without shrinkage). That over-ranked 2023-24 launches that only
lived through a mid/small bull and a 2025 metals melt-up, and it crushed
capacity-constrained but process-stable names via a harsh AUM floor.

This rewrite:
  * scores the actual 12+12 cashflow, not a 12-month SIP
  * splits accumulation vs hold-phase path quality (year-2 has no SIP cushion)
  * shrinks short-history funds toward the peer mean (empirical Bayes)
  * uses theory-first weights; walk-forward Spearman IC is a veto, not a fitter
  * keeps AUM as a mild capacity haircut, not a rank-killer

Luck is real: a single 24-month window can look brilliant. That is why we use
the left tail (p25), hold-phase drawdown, and shrinkage — not just the mean.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ROOT_DIR = SRC_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RISK_FREE = 0.065
SIP_AMOUNT = 10_000.0

logger = logging.getLogger("GrokSipHold")


# ---------------------------------------------------------------------------
# NAV / XIRR primitives
# ---------------------------------------------------------------------------

def clean_nav(df: pd.DataFrame) -> pd.Series:
    """Sorted positive NAV series with a tz-naive DatetimeIndex."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.copy()
    ts = pd.to_datetime(s["timestamp"], errors="coerce", utc=True)
    nav = pd.to_numeric(s["nav"], errors="coerce")
    out = pd.Series(nav.values, index=ts).dropna()
    out = out[out > 0]
    if out.empty:
        return pd.Series(dtype=float)
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def nav_asof(nav: pd.Series, when: pd.Timestamp) -> Optional[float]:
    """Last NAV on or before `when`; if none, first NAV after (small gap)."""
    if nav.empty:
        return None
    when = pd.Timestamp(when)
    prior = nav.loc[:when]
    if not prior.empty:
        return float(prior.iloc[-1])
    later = nav.loc[when:]
    if later.empty:
        return None
    if (later.index[0] - when).days > 14:
        return None
    return float(later.iloc[0])


def slice_to(nav: pd.Series, asof: pd.Timestamp) -> pd.Series:
    return nav.loc[: pd.Timestamp(asof)]


def calendar_days(nav: pd.Series) -> int:
    if len(nav) < 2:
        return 0
    return int((nav.index[-1] - nav.index[0]).days)


def trailing_cagr(nav: pd.Series, years: float) -> Optional[float]:
    if len(nav) < 2:
        return None
    end = nav.index[-1]
    start = end - pd.DateOffset(days=int(years * 365.25))
    window = nav.loc[start:]
    if len(window) < 2 or window.iloc[0] <= 0:
        return None
    span = (window.index[-1] - window.index[0]).days / 365.25
    if span < years * 0.80:
        return None
    return float((window.iloc[-1] / window.iloc[0]) ** (1.0 / span) - 1.0)


def _npv(rate: float, timed: Sequence[Tuple[float, float]]) -> float:
    if rate <= -1.0:
        return float("inf")
    return sum(cf / ((1.0 + rate) ** t) for t, cf in timed)


def xirr(cashflows: Sequence[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    if len(cashflows) < 2:
        return None
    start = cashflows[0][0]
    timed = [((dt - start).days / 365.25, amt) for dt, amt in cashflows]
    try:
        return float(brentq(lambda r: _npv(r, timed), -0.95, 12.0, maxiter=80))
    except (ValueError, RuntimeError):
        return None


def simulate_sip_hold(nav: pd.Series, start: pd.Timestamp) -> Optional[Dict[str, float]]:
    """
    12 SIPs on (or nearest before) the 1st of each month, then a silent year,
    then a full exit at month 24. Returns XIRR plus phase diagnostics.
    """
    if len(nav) < 20:
        return None
    start = pd.Timestamp(start).replace(day=1)
    exit_dt = start + pd.DateOffset(months=24)
    hold_dt = start + pd.DateOffset(months=12)
    if exit_dt > nav.index[-1] + pd.Timedelta(days=10):
        return None
    if start < nav.index[0] - pd.Timedelta(days=5):
        return None

    units = 0.0
    cfs: List[Tuple[pd.Timestamp, float]] = []
    sip_navs: List[float] = []
    for m in range(12):
        d = start + pd.DateOffset(months=m)
        px = nav_asof(nav, d)
        if px is None or px <= 0:
            return None
        units += SIP_AMOUNT / px
        cfs.append((pd.Timestamp(d), -SIP_AMOUNT))
        sip_navs.append(px)

    exit_px = nav_asof(nav, exit_dt)
    if exit_px is None or exit_px <= 0 or units <= 0:
        return None
    cfs.append((pd.Timestamp(exit_dt), units * exit_px))
    rate = xirr(cfs)
    if rate is None:
        return None

    hold_nav = nav.loc[hold_dt:exit_dt]
    if len(hold_nav) < 3:
        hold_nav = nav.loc[(hold_dt - pd.Timedelta(days=10)):exit_dt]
    if len(hold_nav) >= 3:
        h0, h1 = float(hold_nav.iloc[0]), float(hold_nav.iloc[-1])
        hold_ret = h1 / h0 - 1.0 if h0 > 0 else np.nan
        peak = hold_nav.cummax()
        hold_dd = float((hold_nav / peak - 1.0).min())
    else:
        hold_ret, hold_dd = np.nan, np.nan

    acc_nav = nav.loc[start:hold_dt]
    if len(acc_nav) >= 3:
        peak = acc_nav.cummax()
        acc_dd = float((acc_nav / peak - 1.0).min())
    else:
        acc_dd = np.nan

    return {
        "xirr": float(rate),
        "hold_ret": float(hold_ret) if hold_ret == hold_ret else np.nan,
        "hold_dd": float(hold_dd) if hold_dd == hold_dd else np.nan,
        "acc_dd": float(acc_dd) if acc_dd == acc_dd else np.nan,
        "exit_multiple": float((units * exit_px) / (SIP_AMOUNT * 12.0)),
    }


def month_range(first: pd.Timestamp, last: pd.Timestamp) -> List[pd.Timestamp]:
    cur = pd.Timestamp(first).replace(day=1)
    end = pd.Timestamp(last).replace(day=1)
    out: List[pd.Timestamp] = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return out


def rolling_siphold(nav: pd.Series) -> pd.DataFrame:
    """Monthly-start panel of completed 24-month SIP-hold paths."""
    if len(nav) < 40:
        return pd.DataFrame()
    first = (nav.index[0] + pd.DateOffset(months=1)).replace(day=1)
    last = (nav.index[-1] - pd.DateOffset(months=24)).replace(day=1)
    if last < first:
        return pd.DataFrame()
    rows = []
    for start in month_range(first, last):
        sim = simulate_sip_hold(nav, start)
        if sim:
            sim["start"] = start
            rows.append(sim)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Path / skill features
# ---------------------------------------------------------------------------

def max_drawdown(nav: pd.Series) -> Optional[float]:
    if len(nav) < 8:
        return None
    return float((nav / nav.cummax() - 1.0).min())


def ulcer_index(nav: pd.Series) -> Optional[float]:
    if len(nav) < 8:
        return None
    dd = nav / nav.cummax() - 1.0
    return float(np.sqrt(np.mean(np.square(dd.values))))


def recovery_half_life_weeks(nav: pd.Series) -> Optional[float]:
    """Weeks from deepest trough back to 50% of that peak-to-trough gap."""
    if len(nav) < 16:
        return None
    dd = nav / nav.cummax() - 1.0
    if float(dd.min()) > -0.04:
        return 8.0
    trough = int(dd.values.argmin())
    peak_val = float(nav.iloc[: trough + 1].max())
    trough_val = float(nav.iloc[trough])
    target = trough_val + 0.5 * (peak_val - trough_val)
    after = nav.iloc[trough:]
    hit = after[after >= target]
    if hit.empty:
        return float(min(80.0, (nav.index[-1] - nav.index[trough]).days / 7.0))
    return float((hit.index[0] - nav.index[trough]).days / 7.0)


def weekly_returns(nav: pd.Series) -> pd.Series:
    w = nav.resample("W-FRI").last().dropna()
    return w.pct_change().dropna()


def monthly_returns(nav: pd.Series) -> pd.Series:
    m = nav.resample("ME").last().dropna()
    return m.pct_change().dropna()


def align_weekly(*series: pd.Series) -> pd.DataFrame:
    cols = [weekly_returns(s).rename(f"r{i}") for i, s in enumerate(series) if s is not None and len(s) > 5]
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1, join="inner").dropna()
    return df


def information_ratio(fund: pd.Series, bench: pd.Series) -> Optional[float]:
    df = pd.concat({"f": monthly_returns(fund), "b": monthly_returns(bench)}, axis=1, join="inner").dropna()
    if len(df) < 18:
        return None
    ex = df["f"] - df["b"]
    sd = float(ex.std(ddof=1))
    if sd < 1e-12:
        return None
    return float(ex.mean() / sd * math.sqrt(12.0))


def capture_ratio(fund: pd.Series, bench: pd.Series, downside: bool) -> Optional[float]:
    df = pd.concat({"f": monthly_returns(fund), "b": monthly_returns(bench)}, axis=1, join="inner").dropna()
    if len(df) < 18:
        return None
    mask = df["b"] < 0 if downside else df["b"] > 0
    sub = df.loc[mask]
    if len(sub) < 4:
        return None
    b = float((1.0 + sub["b"]).prod() - 1.0)
    f = float((1.0 + sub["f"]).prod() - 1.0)
    if abs(b) < 1e-12:
        return None
    return float(f / b)


def sortino_ratio(nav: pd.Series, rf: float = RISK_FREE) -> Optional[float]:
    r = monthly_returns(nav)
    if len(r) < 18:
        return None
    rf_m = (1.0 + rf) ** (1.0 / 12.0) - 1.0
    ex = r - rf_m
    downside = ex[ex < 0]
    if len(downside) < 3:
        return float(ex.mean() * 12.0 / 0.02)
    dd = float(np.sqrt(np.mean(np.square(downside.values))))
    if dd < 1e-12:
        return None
    return float(ex.mean() / dd * math.sqrt(12.0))


def residual_alpha(
    fund: pd.Series,
    large: pd.Series,
    mid: pd.Series,
    small: pd.Series,
) -> Optional[float]:
    """Annualized OLS intercept of weekly fund returns on large/mid/small."""
    df = align_weekly(fund, large, mid, small)
    if len(df) < 40 or not {"r0", "r1", "r2", "r3"}.issubset(df.columns):
        return None
    y = df["r0"].values
    X = np.column_stack([np.ones(len(df)), df["r1"].values, df["r2"].values, df["r3"].values])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float((1.0 + coef[0]) ** 52 - 1.0)


def rebound_elasticity(fund: pd.Series, bench: pd.Series, dd_thresh: float = 0.08, weeks: int = 8) -> Optional[float]:
    df = align_weekly(fund, bench)
    if len(df) < weeks + 20 or not {"r0", "r1"}.issubset(df.columns):
        return None
    rb = df["r1"].values
    rf = df["r0"].values
    cum = np.cumprod(1.0 + rb)
    peak = np.maximum.accumulate(cum)
    dd = cum / (peak + 1e-12) - 1.0
    scores = []
    i = 0
    while i < len(dd) - weeks:
        if dd[i] <= -dd_thresh:
            ex = float(np.sum(rf[i + 1 : i + 1 + weeks]) - np.sum(rb[i + 1 : i + 1 + weeks]))
            scores.append(ex / (-dd[i] + 1e-9))
            i += weeks
        else:
            i += 1
    if not scores:
        return 0.0
    return float(np.median(scores))


def swing_elasticity(fund: pd.Series, bench: pd.Series) -> Optional[float]:
    df = align_weekly(fund, bench)
    if len(df) < 30 or not {"r0", "r1"}.issubset(df.columns):
        return None
    rf, rb = df["r0"].values, df["r1"].values
    up = np.mean(rf[rb > 0] - rb[rb > 0]) if np.any(rb > 0) else 0.0
    down = np.mean(rf[rb < 0] - rb[rb < 0]) if np.any(rb < 0) else 0.0
    return float(up - down)


def style_mid_share(fund: pd.Series, large: pd.Series, mid: pd.Series, small: pd.Series, window: int = 52) -> Tuple[Optional[float], Optional[float]]:
    df = align_weekly(fund, large, mid, small)
    if len(df) < window or not {"r0", "r1", "r2", "r3"}.issubset(df.columns):
        return None, None
    y = df["r0"].values[-window:]
    X = df[["r1", "r2", "r3"]].values[-window:]
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    tot = float(np.sum(np.abs(coef)) + 1e-12)
    mid_share = float(abs(coef[1]) / tot)
    betas = []
    step = 13
    for s in range(0, max(1, len(df) - window), step):
        yw = df["r0"].values[s : s + window]
        Xw = df[["r1", "r2", "r3"]].values[s : s + window]
        if len(yw) < window:
            break
        try:
            cw, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            betas.append(float(cw[1]))
        except np.linalg.LinAlgError:
            continue
    drift = float(np.std(betas)) if len(betas) >= 2 else None
    return mid_share, drift


def gold_participation(fund: pd.Series, gold: pd.Series) -> Optional[float]:
    """Fraction of gold-up weeks the fund is also up, times relative magnitude clip."""
    df = align_weekly(fund, gold)
    if len(df) < 26 or not {"r0", "r1"}.issubset(df.columns):
        return None
    rf, rg = df["r0"].values, df["r1"].values
    mask = rg > 0
    if not np.any(mask):
        return None
    # Participation, not excess: funds are not gold vehicles.
    ratio = np.clip(rf[mask] / (rg[mask] + 1e-9), -0.2, 1.5)
    return float(np.mean(ratio))


def metals_overfit(fund: pd.Series, silver: pd.Series) -> Optional[float]:
    """Recent silver beta minus long silver beta. High = loaded metals late."""
    df = align_weekly(fund, silver)
    if len(df) < 40 or not {"r0", "r1"}.issubset(df.columns):
        return None
    rs = df["r1"].values
    rf = df["r0"].values

    def beta(a, b) -> float:
        if np.std(b) < 1e-10:
            return 0.0
        return float(np.cov(a, b)[0, 1] / (np.var(b) + 1e-12))

    long_b = beta(rf, rs)
    short_b = beta(rf[-26:], rs[-26:]) if len(rf) >= 26 else long_b
    return float(short_b - long_b)


def equity_beta_stability(fund: pd.Series, equity: pd.Series, window: int = 26) -> Optional[float]:
    df = align_weekly(fund, equity)
    if len(df) < window + 8 or not {"r0", "r1"}.issubset(df.columns):
        return None
    betas = []
    rf, re = df["r0"].values, df["r1"].values
    for i in range(window, len(rf)):
        x = re[i - window : i]
        y = rf[i - window : i]
        if np.std(x) < 1e-10:
            continue
        betas.append(float(np.cov(y, x)[0, 1] / (np.var(x) + 1e-12)))
    if len(betas) < 4:
        return None
    return float(np.std(betas))


def batting_vs_bench(fund_panel: pd.DataFrame, bench_panel: pd.DataFrame) -> Optional[float]:
    """Hit-rate vs benchmark, empirical-Bayes shrunk toward 0.5.

    Monthly 24-month windows overlap by 23/24, so a 35-window 1.00 batting
    average is not 35 independent wins. Sixteen pseudo-observations at 0.5
    stop a single-cycle fund from looking like a sure thing.
    """
    if fund_panel.empty or bench_panel.empty:
        return None
    m = fund_panel[["start", "xirr"]].merge(
        bench_panel[["start", "xirr"]].rename(columns={"xirr": "bx"}),
        on="start",
        how="inner",
    )
    if len(m) < 3:
        return None
    raw = float(np.mean(m["xirr"].values > m["bx"].values))
    n = float(len(m))
    k = 16.0
    return (n * raw + k * 0.5) / (n + k)


# ---------------------------------------------------------------------------
# Cross-sectional scoring
# ---------------------------------------------------------------------------

def robust_z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    scale = float(mad * 1.4826) if mad and mad > 1e-12 else float(x.std(ddof=0) or 1.0)
    z = (x - med) / scale
    return z.clip(-clip, clip)


def shrink_lambda(n_windows: float, k: float) -> float:
    """Shrink using effective independent 24-month blocks, not overlapping months.

    Starts are monthly and each path is 24 months, so n_windows/12 is years
    of start dates. `k` is still authored on the old month-count scale
    (16–24); k/8 (~2–3 years) is the year-scale prior. Dividing both n and
    k by 12 would be a no-op and would keep treating overlap as extra
    evidence.
    """
    n_eff = max(0.0, float(n_windows) / 12.0)
    k_eff = max(1.5, float(k) / 8.0)
    return float(n_eff / (n_eff + k_eff)) if (n_eff + k_eff) > 0 else 0.0


def aum_multiplier(aum: float, knots: Sequence[Tuple[float, float]]) -> float:
    if aum is None or (isinstance(aum, float) and math.isnan(aum)) or aum <= 0:
        return knots[0][1]
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.interp(float(aum), xs, ys))


def panel_stats(panel: pd.DataFrame) -> Dict[str, float]:
    if panel.empty or "xirr" not in panel:
        return {
            "n_windows": 0.0,
            "xirr_mean": np.nan,
            "xirr_p25": np.nan,
            "xirr_std": np.nan,
            "hold_ret": np.nan,
            "hold_dd": np.nan,
            "acc_dd": np.nan,
        }
    x = panel["xirr"].dropna().astype(float)
    return {
        "n_windows": float(len(x)),
        "xirr_mean": float(x.mean()) if len(x) else np.nan,
        "xirr_p25": float(x.quantile(0.25)) if len(x) else np.nan,
        "xirr_std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "hold_ret": float(panel["hold_ret"].mean()) if "hold_ret" in panel else np.nan,
        "hold_dd": float(panel["hold_dd"].mean()) if "hold_dd" in panel else np.nan,
        "acc_dd": float(panel["acc_dd"].mean()) if "acc_dd" in panel else np.nan,
    }


def calmar(nav: pd.Series) -> Optional[float]:
    c3 = trailing_cagr(nav, 3.0)
    md = max_drawdown(nav)
    if c3 is None or md is None or abs(md) < 1e-6:
        return None
    return float(c3 / abs(md))


def omega_ratio(nav: pd.Series) -> Optional[float]:
    """Sum of gains / abs(sum of losses) on monthly returns."""
    r = monthly_returns(nav)
    if len(r) < 18:
        return None
    pos = float(r[r > 0].sum())
    neg = float(r[r < 0].sum())
    return float(pos / (abs(neg) + 1e-12))


def weekly_beta(fund: pd.Series, bench: pd.Series) -> Optional[float]:
    if bench is None or len(bench) < 30 or len(fund) < 30:
        return None
    df = align_weekly(fund, bench)
    if len(df) < 26 or not {"r0", "r1"}.issubset(df.columns):
        return None
    if float(df["r1"].std()) < 1e-12:
        return None
    return float(np.cov(df["r0"], df["r1"])[0, 1] / (np.var(df["r1"]) + 1e-12))


# ---------------------------------------------------------------------------
# Sector configuration
# ---------------------------------------------------------------------------

@dataclass
class SectorSpec:
    name: str
    outfile: str
    benchmark: str
    subsector_pred: Callable[[str], bool]
    equity_sector_only: bool = False
    extra_subsectors: Optional[Sequence[str]] = None
    shrink_k: float = 10.0
    aum_knots: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: ((0.0, 1.0), (25_000.0, 0.97), (80_000.0, 0.93))
    )
    # signed weights; negative => lower is better
    weights: Dict[str, float] = field(default_factory=dict)
    extras: Tuple[str, ...] = ()


def _is_mid(sub: str) -> bool:
    return "Mid" in (sub or "") and "Cap" in (sub or "")


def _is_small(sub: str) -> bool:
    return "Small" in (sub or "") and "Cap" in (sub or "")


def _is_multi(sub: str) -> bool:
    return "Multi Asset" in (sub or "")


TOTAL_SUBS = {"Contra Fund", "Flexi Cap Fund", "Focused Fund", "Multi Cap Fund", "Value Fund"}


def _is_total(sub: str) -> bool:
    return (sub or "") in TOTAL_SUBS


# Theory-first weights, vetoed by walk-forward top-k XIRR (not a fitter).
# Iter 1: Mid purity/IR; Small drop IR; Multi return-adequacy; TM skill.
# Iter 2: Small add return-adequacy (UTI/Quantum were too quiet);
#         Multi drop Calmar (Edelweiss 8% CAGR → Calmar 9.8) + history term.
# Iter 3: Multi cut n_windows (Quant rode it) and raise p25 + alloc-stability.
# Iter 4: Mid shrink 24 (WOC still #2); Small add batting; TM more shrink + omega.
# Iter 5: overlapping windows are not independent — shrink on n/12 years,
#         Bayes-shrink batting to 0.5, fold return_adequacy clone, drop TM
#         mid-share "style_purity", peer-shrink panel stats before z-score.
SPECS: Dict[str, SectorSpec] = {
    "Mid Cap": SectorSpec(
        name="Mid Cap",
        outfile="Mid Cap_Grok.csv",
        benchmark="Mid Cap",
        subsector_pred=_is_mid,
        shrink_k=24.0,  # WOC-style 3-4y funds must not auto-win
        aum_knots=((0.0, 1.00), (20_000.0, 0.98), (50_000.0, 0.95), (120_000.0, 0.92)),
        extras=("style_purity", "style_drift", "swing_elasticity", "upside_capture", "capture_spread"),
        weights={
            "style_purity": 0.10,     # strong IC, but Invesco is a blend and still skillful
            "info_ratio": 0.13,
            "upside_capture": 0.09,
            "swing_elasticity": 0.08,
            "capture_spread": 0.08,
            "residual_alpha": 0.08,
            "xirr_mean": 0.08,
            "hold_ret": 0.08,
            "xirr_p25": 0.08,
            "hold_dd": 0.04,          # safety-heavy packs had negative edge
            "ulcer": -0.02,
            "style_drift": -0.03,
            "sortino": 0.04,
            "downside_capture": -0.03,
        },
    ),
    "Small Cap": SectorSpec(
        name="Small Cap",
        outfile="Small Cap_Grok.csv",
        benchmark="Small Cap",
        subsector_pred=_is_small,
        shrink_k=16.0,  # 2023-24 launches get one lucky window
        aum_knots=((0.0, 1.02), (2_000.0, 1.00), (8_000.0, 0.97), (20_000.0, 0.93), (50_000.0, 0.88)),
        extras=("hold_vol", "rel_dd", "xirr_std", "omega", "batting"),
        weights={
            # Iter 2: quiet 7–15% XIRR funds outranked 28–33% process names.
            # return_adequacy was xirr_mean - Rf (identical z-score); folded in.
            "xirr_p25": 0.22,
            "xirr_mean": 0.18,
            "xirr_std": -0.10,
            "ulcer": -0.08,
            "rel_dd": -0.06,
            "hold_vol": -0.08,
            "hold_dd": 0.08,
            "sortino": 0.08,
            "calmar": 0.04,
            "omega": 0.04,
            "downside_capture": -0.04,
            "batting": 0.04,
        },
    ),
    "Multi Asset": SectorSpec(
        name="Multi Asset",
        outfile="Multi Asset_Grok.csv",
        benchmark="Total Market",
        subsector_pred=_is_multi,
        shrink_k=22.0,  # 18m metals riders (WOC) get shrunk harder
        aum_knots=((0.0, 0.90), (200.0, 0.95), (800.0, 1.00), (20_000.0, 1.00)),
        extras=("gold_part", "metals_overfit", "alloc_stability", "eq_down_prot"),
        weights={
            # Iter 3: Quant's 139 windows + 7% p25 should not beat Nippon/Tata.
            # return_adequacy was a clone of xirr_mean; folded into xirr_mean.
            "xirr_mean": 0.22,
            "xirr_p25": 0.14,
            "gold_part": 0.10,
            "metals_overfit": -0.10,
            "n_windows": 0.08,
            "eq_down_prot": 0.06,
            "sortino": 0.08,
            "alloc_stability": -0.06,
            "omega": 0.06,
            "hold_dd": 0.04,
            "ulcer": -0.04,
        },
    ),
    "Total Market": SectorSpec(
        name="Total Market",
        outfile="Total Market_Grok.csv",
        benchmark="Total Market",
        subsector_pred=_is_total,
        equity_sector_only=True,
        shrink_k=16.0,
        aum_knots=((0.0, 1.00), (20_000.0, 0.98), (50_000.0, 0.95), (90_000.0, 0.93)),
        extras=("batting", "capture_spread", "omega"),
        weights={
            "residual_alpha": 0.14,
            "info_ratio": 0.10,
            "batting": 0.12,
            "xirr_p25": 0.12,
            "capture_spread": 0.08,
            "hold_dd": 0.06,
            "downside_capture": -0.08,
            "xirr_mean": 0.06,
            "ulcer": -0.06,
            "sortino": 0.06,
            "omega": 0.04,
            "calmar": 0.04,
        },
    ),
}


# ---------------------------------------------------------------------------
# Per-fund feature pack
# ---------------------------------------------------------------------------

def features_for_fund(
    spec: SectorSpec,
    nav: pd.Series,
    benches: Dict[str, pd.Series],
    aum: float,
) -> Dict[str, object]:
    bench = benches.get("bench", pd.Series(dtype=float))
    panel = rolling_siphold(nav)
    bench_panel = rolling_siphold(bench) if len(bench) else pd.DataFrame()
    st = panel_stats(panel)

    rec: Dict[str, object] = {
        "data_days": calendar_days(nav),
        "cagr_1y": trailing_cagr(nav, 1.0),
        "cagr_3y": trailing_cagr(nav, 3.0),
        "cagr_5y": trailing_cagr(nav, 5.0),
        "n_windows": st["n_windows"],
        "xirr_mean": st["xirr_mean"],
        "xirr_p25": st["xirr_p25"],
        "xirr_std": st["xirr_std"],
        "hold_ret": st["hold_ret"],
        "hold_dd": st["hold_dd"],
        "acc_dd": st["acc_dd"],
        "info_ratio": information_ratio(nav, bench) if len(bench) else None,
        "downside_capture": capture_ratio(nav, bench, True) if len(bench) else None,
        "upside_capture": capture_ratio(nav, bench, False) if len(bench) else None,
        "rebound": rebound_elasticity(nav, bench) if len(bench) else None,
        "ulcer": ulcer_index(nav),
        "calmar": calmar(nav),
        "sortino": sortino_ratio(nav),
        "max_dd": max_drawdown(nav),
        "recovery_weeks": recovery_half_life_weeks(nav),
        "aum_factor": aum_multiplier(aum, spec.aum_knots),
        "shrink_lambda": shrink_lambda(st["n_windows"], spec.shrink_k),
    }

    large, mid, small = benches.get("large"), benches.get("mid"), benches.get("small")
    if large is not None and mid is not None and small is not None:
        rec["residual_alpha"] = residual_alpha(nav, large, mid, small)
        purity, drift = style_mid_share(nav, large, mid, small)
        rec["style_purity"] = purity
        rec["style_drift"] = drift
    else:
        rec["residual_alpha"] = None
        rec["style_purity"] = None
        rec["style_drift"] = None

    rec["swing_elasticity"] = swing_elasticity(nav, bench) if len(bench) else None
    rec["batting"] = batting_vs_bench(panel, bench_panel)

    if not panel.empty and "hold_ret" in panel:
        rec["hold_vol"] = float(panel["hold_ret"].std(ddof=1)) if len(panel) > 2 else np.nan
    else:
        rec["hold_vol"] = np.nan

    gold, silver, equity = benches.get("gold"), benches.get("silver"), benches.get("equity", bench)
    rec["gold_part"] = gold_participation(nav, gold) if gold is not None and len(gold) else None
    rec["metals_overfit"] = metals_overfit(nav, silver) if silver is not None and len(silver) else None
    rec["alloc_stability"] = equity_beta_stability(nav, equity) if equity is not None and len(equity) else None
    dc = rec["downside_capture"]
    rec["eq_down_prot"] = (1.0 - float(dc)) if dc is not None else None
    up, dn = rec.get("upside_capture"), rec.get("downside_capture")
    rec["capture_spread"] = (
        (float(up) - float(dn)) if up is not None and dn is not None else None
    )
    md, bdd = rec.get("max_dd"), max_drawdown(bench) if len(bench) else None
    rec["rel_dd"] = (
        float(md) / float(bdd) if md is not None and bdd not in (None, 0) else None
    )
    rec["omega"] = omega_ratio(nav)
    xm = rec.get("xirr_mean")
    rec["return_adequacy"] = (
        (float(xm) - RISK_FREE) if xm is not None and xm == xm else None
    )
    rec["equity_beta"] = weekly_beta(nav, equity) if equity is not None and len(equity) else None
    rec["aum"] = aum
    return rec


# Features estimated from overlapping 24-month windows. A 15-window p25 is
# not a left tail — it is 15 heavily overlapping views of one regime.
_PANEL_FEATS = (
    "xirr_mean",
    "xirr_p25",
    "hold_ret",
    "hold_dd",
    "hold_vol",
    "xirr_std",
    "batting",
    "return_adequacy",
)


def shrink_panel_to_peers(df: pd.DataFrame, k_years: float = 2.0) -> pd.DataFrame:
    """Pull overlap-inflated panel stats toward the cross-sectional median."""
    n_eff = pd.to_numeric(df.get("n_windows"), errors="coerce").fillna(0.0) / 12.0
    lam = (n_eff / (n_eff + k_years)).clip(0.0, 1.0)
    out = df.copy()
    for col in _PANEL_FEATS:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        med = x.median()
        if pd.isna(med):
            continue
        out[col] = med + lam.values * (x - med)
    return out


def score_frame(df: pd.DataFrame, spec: SectorSpec) -> pd.DataFrame:
    work = shrink_panel_to_peers(df)
    feats = list(spec.weights.keys())
    z = pd.DataFrame(index=work.index)
    for f in feats:
        z[f] = robust_z(work[f]) if f in work.columns else 0.0
        z[f] = z[f].fillna(0.0)

    w = np.array([spec.weights[f] for f in feats], dtype=float)
    raw = z.values @ w
    lam = pd.to_numeric(df.get("shrink_lambda"), errors="coerce").fillna(0.35).clip(0.15, 1.0)
    # Shrink raw skill toward 0 (cross-sectional mean of robust z).
    shrunk = raw * lam.values
    # Wider CDF so a cluster of good funds does not pin to the same ceiling.
    score = 8.0 + 86.0 * norm.cdf(shrunk / 0.70)
    aum_f = pd.to_numeric(df.get("aum_factor"), errors="coerce").fillna(1.0)
    nwin = pd.to_numeric(df["n_windows"], errors="coerce").fillna(0.0)
    hist = np.where(nwin >= 1, 1.0, 0.82)
    # Tiny history quality term: breaks 0.00 ties, does not reorder strong gaps.
    score = score + 0.20 * np.log1p(nwin.values)
    final = np.clip(score * aum_f.values * hist, 8.0, 96.5)
    out = df.copy()
    out["raw_skill"] = np.round(raw, 4)
    out["score"] = np.round(final, 2)
    out["rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Universe + I/O
# ---------------------------------------------------------------------------

def select_universe(all_mf: pd.DataFrame, spec: SectorSpec) -> pd.DataFrame:
    df = all_mf.copy()
    if spec.equity_sector_only and "sector" in df.columns:
        df = df[df["sector"].astype(str) == "Equity"]
    mask = df["subsector"].astype(str).map(spec.subsector_pred)
    return df.loc[mask].copy()


def load_benches(provider, spec: SectorSpec) -> Dict[str, pd.Series]:
    from mf_data_provider import MfDataProvider  # noqa: F401 — type hint only

    def _idx(name: str) -> pd.Series:
        try:
            return clean_nav(provider.get_index_chart(name))
        except Exception as exc:
            logger.warning("index %s failed: %s", name, exc)
            return pd.Series(dtype=float)

    benches = {
        "bench": _idx(spec.benchmark),
        "large": _idx("Large Cap"),
        "mid": _idx("Mid Cap"),
        "small": _idx("Small Cap"),
        "equity": _idx("Total Market"),
        "gold": _idx("Gold"),
    }
    if spec.name == "Multi Asset":
        try:
            benches["silver"] = clean_nav(provider.get_mf_chart("M_ICPVF"))
        except Exception as exc:
            logger.warning("silver proxy failed: %s", exc)
            benches["silver"] = pd.Series(dtype=float)
    return benches


def analyse_sector(provider, spec: SectorSpec) -> pd.DataFrame:
    all_mf = provider.list_all_mf()
    uni = select_universe(all_mf, spec)
    logger.info("Scoring %d %s funds", len(uni), spec.name)
    benches = load_benches(provider, spec)
    if benches["bench"].empty:
        logger.warning("Empty primary benchmark for %s", spec.name)

    rows: List[Dict[str, object]] = []
    for i, row in uni.iterrows():
        mf_id = str(row["mfId"])
        try:
            chart = provider.get_mf_chart(mf_id)
        except Exception as exc:
            logger.warning("%s chart failed: %s", mf_id, exc)
            continue
        nav = clean_nav(chart)
        if len(nav) < 40:
            continue
        feats = features_for_fund(spec, nav, benches, float(row.get("aum") or 0.0))
        feats["mfId"] = mf_id
        feats["name"] = row.get("name", mf_id)
        feats["subsector"] = row.get("subsector", "")
        rows.append(feats)
        if len(rows) % 25 == 0:
            logger.info("  %d/%d analysed", len(rows), len(uni))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return score_frame(df, spec)


def format_output(df: pd.DataFrame, spec: SectorSpec) -> pd.DataFrame:
    pct_cols = [
        "cagr_1y",
        "cagr_3y",
        "cagr_5y",
        "xirr_mean",
        "xirr_p25",
        "xirr_std",
        "hold_ret",
        "hold_dd",
        "acc_dd",
        "residual_alpha",
        "max_dd",
        "hold_vol",
        "return_adequacy",
    ]
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").apply(
                lambda v: round(v * 100.0, 2) if pd.notna(v) else np.nan
            )
    for c in ("info_ratio", "downside_capture", "upside_capture", "rebound", "ulcer",
              "calmar", "sortino", "style_purity", "style_drift", "swing_elasticity",
              "batting", "gold_part", "metals_overfit", "alloc_stability",
              "eq_down_prot", "capture_spread", "rel_dd", "omega",
              "return_adequacy", "equity_beta",
              "aum_factor", "shrink_lambda", "raw_skill"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    cols = [
        "mfId",
        "name",
        "rank",
        "score",
        "data_days",
        "cagr_3y",
        "cagr_5y",
        "n_windows",
        "xirr_mean",
        "xirr_p25",
        "hold_ret",
        "hold_dd",
        "info_ratio",
        "residual_alpha",
        "downside_capture",
        "rebound",
        "ulcer",
        "sortino",
        "calmar",
        "shrink_lambda",
        "aum_factor",
    ]
    extra_cols = [c for c in spec.extras if c in out.columns]
    if spec.name == "Total Market" and "subsector" in out.columns:
        cols.insert(4, "subsector")
    keep = [c for c in cols + extra_cols if c in out.columns]
    return out[keep]


def write_results(df: pd.DataFrame, spec: SectorSpec) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / spec.outfile
    out = format_output(df, spec)
    out.to_csv(path, index=False)
    logger.info("Wrote %d funds -> %s", len(out), path)
    logger.info(
        "Score range %.1f–%.1f (std %.1f, unique %d)",
        out["score"].min(),
        out["score"].max(),
        out["score"].std(),
        out["score"].nunique(),
    )
    logger.info("Top 8:\n%s", out.head(8)[["rank", "name", "score"]].to_string(index=False))
    return path


# ---------------------------------------------------------------------------
# Walk-forward IC (hypothesis veto, not a weight fitter)
# ---------------------------------------------------------------------------

def walk_forward_ic(
    provider,
    spec: SectorSpec,
    step_months: int = 3,
    min_history_days: int = 400,
) -> pd.DataFrame:
    """
    At each as-of T (with 24 months of future NAV), compute features on data<=T
    and the *realized* SIP-hold XIRR starting at T. Report Spearman IC.

    Used to veto features that are anti-predictive. Weights stay theory-first
    so we do not overfit five years of one bull/bear sequence.
    """
    all_mf = provider.list_all_mf()
    uni = select_universe(all_mf, spec)
    benches_full = load_benches(provider, spec)

    navs: Dict[str, pd.Series] = {}
    meta: Dict[str, Dict[str, object]] = {}
    for _, row in uni.iterrows():
        mf_id = str(row["mfId"])
        try:
            nav = clean_nav(provider.get_mf_chart(mf_id))
        except Exception:
            continue
        if len(nav) < 60:
            continue
        navs[mf_id] = nav
        meta[mf_id] = {"name": row.get("name", mf_id), "aum": float(row.get("aum") or 0.0)}

    if not navs:
        return pd.DataFrame()

    global_end = max(s.index[-1] for s in navs.values())
    global_start = min(s.index[0] for s in navs.values())
    first_asof = (global_start + pd.DateOffset(days=min_history_days)).replace(day=1)
    last_asof = (global_end - pd.DateOffset(months=24)).replace(day=1)
    asofs = month_range(first_asof, last_asof)[::step_months]
    logger.info("Walk-forward %s: %d as-of dates, %d funds", spec.name, len(asofs), len(navs))

    feat_names = list(spec.weights.keys())
    ic_rows = []
    for asof in asofs:
        recs = []
        labels = []
        for mf_id, nav in navs.items():
            hist = slice_to(nav, asof)
            if calendar_days(hist) < min_history_days:
                continue
            fut = simulate_sip_hold(nav, asof)
            if fut is None:
                continue
            bhist = {k: slice_to(v, asof) if len(v) else v for k, v in benches_full.items()}
            try:
                feats = features_for_fund(spec, hist, bhist, float(meta[mf_id]["aum"]))
            except Exception:
                continue
            feats["mfId"] = mf_id
            recs.append(feats)
            labels.append(fut["xirr"])
        if len(recs) < 8:
            continue
        fdf = pd.DataFrame(recs)
        y = np.array(labels, dtype=float)
        row = {"asof": asof.strftime("%Y-%m-%d"), "n": len(fdf)}
        for f in feat_names:
            x = pd.to_numeric(fdf[f], errors="coerce") if f in fdf.columns else pd.Series(dtype=float)
            mask = x.notna() & np.isfinite(y)
            if mask.sum() < 8:
                row[f] = np.nan
                continue
            ic, _ = spearmanr(x[mask], y[mask])
            # Flip IC for lower-is-better features so + always means "worked".
            sign = 1.0 if spec.weights[f] >= 0 else -1.0
            row[f] = float(ic * sign) if ic == ic else np.nan
        ic_rows.append(row)
        logger.info("  asof %s n=%d mean_ic=%.3f", row["asof"], row["n"],
                    np.nanmean([row[f] for f in feat_names]))

    ic_df = pd.DataFrame(ic_rows)
    if not ic_df.empty:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPERIMENTS_DIR / f"{spec.name}_Grok_ic.csv"
        ic_df.to_csv(path, index=False)
        means = ic_df[feat_names].mean()
        logger.info("Mean signed IC (%s):\n%s", spec.name, means.sort_values(ascending=False).to_string())
        logger.info("Wrote IC panel -> %s", path)
    return ic_df


def run(sector: str, date: Optional[str] = None, backtest: bool = False) -> Optional[Path]:
    from mf_data_provider import MfDataProvider

    if sector not in SPECS:
        raise ValueError(f"Unknown sector {sector!r}; choose from {list(SPECS)}")
    spec = SPECS[sector]
    provider = MfDataProvider(date=date)
    if backtest:
        walk_forward_ic(provider, spec)
        return None
    df = analyse_sector(provider, spec)
    if df.empty:
        logger.error("No funds scored for %s", sector)
        return None
    return write_results(df, spec)
