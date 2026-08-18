#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward hypothesis lab for the 12+12 SIP-hold objective.

Builds a look-ahead-safe panel (features at T, realized 24m XIRR from T),
then tests theory-first weight vectors by *top-k realized XIRR*, not by
fitting coefficients to the same panel.

Usage:
    python src/experiments/hypotheses.py [--sector "Small Cap"] [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
ALG = SRC / "algorithms"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ALG) not in sys.path:
    sys.path.insert(0, str(ALG))

from grok_engine import (  # noqa: E402
    EXPERIMENTS_DIR,
    RISK_FREE,
    SPECS,
    calendar_days,
    capture_ratio,
    clean_nav,
    equity_beta_stability,
    features_for_fund,
    gold_participation,
    information_ratio,
    load_benches,
    max_drawdown,
    metals_overfit,
    month_range,
    residual_alpha,
    robust_z,
    select_universe,
    simulate_sip_hold,
    slice_to,
    sortino_ratio,
    style_mid_share,
    trailing_cagr,
    ulcer_index,
)
from mf_data_provider import MfDataProvider  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hyp")

# Extra candidate features (beyond current SPECS weights).
EXTRA_FEATS = (
    "xirr_std",
    "cagr_1y",
    "upside_capture",
    "capture_spread",
    "rel_dd",
    "omega",
    "hit_rate",
    "return_adequacy",
    "n_windows",
    "equity_beta",
    "silver_beta",
    "gold_down_part",
)


def extra_features(nav: pd.Series, benches: Dict[str, pd.Series], rec: Dict) -> Dict:
    bench = benches.get("bench", pd.Series(dtype=float))
    rec = dict(rec)
    rec["cagr_1y"] = trailing_cagr(nav, 1.0)
    up = rec.get("upside_capture")
    dn = rec.get("downside_capture")
    rec["capture_spread"] = (
        (float(up) - float(dn)) if up is not None and dn is not None else None
    )
    md = rec.get("max_dd")
    bdd = max_drawdown(bench) if len(bench) else None
    rec["rel_dd"] = (
        float(md) / float(bdd) if md is not None and bdd not in (None, 0) else None
    )
    r = nav.resample("ME").last().dropna().pct_change().dropna()
    if len(r) >= 18:
        pos, neg = r[r > 0], r[r < 0]
        rec["omega"] = float(pos.sum() / (abs(neg.sum()) + 1e-12))
    else:
        rec["omega"] = None
    rec["hit_rate"] = rec.get("batting")
    xm = rec.get("xirr_mean")
    rec["return_adequacy"] = (float(xm) - RISK_FREE) if xm is not None and xm == xm else None

    # Equity beta (full-sample weekly).
    eq = benches.get("equity", bench)
    rec["equity_beta"] = _beta(nav, eq)
    sil = benches.get("silver")
    rec["silver_beta"] = _beta(nav, sil) if sil is not None and len(sil) else None
    gold = benches.get("gold")
    rec["gold_down_part"] = _gold_down(nav, gold) if gold is not None and len(gold) else None
    return rec


def _beta(fund: pd.Series, bench: pd.Series) -> Optional[float]:
    if bench is None or len(bench) < 30 or len(fund) < 30:
        return None
    a = fund.resample("W-FRI").last().dropna().pct_change()
    b = bench.resample("W-FRI").last().dropna().pct_change()
    df = pd.concat({"f": a, "b": b}, axis=1, join="inner").dropna()
    if len(df) < 26 or float(df["b"].std()) < 1e-12:
        return None
    return float(np.cov(df["f"], df["b"])[0, 1] / (np.var(df["b"]) + 1e-12))


def _gold_down(fund: pd.Series, gold: pd.Series) -> Optional[float]:
    a = fund.resample("W-FRI").last().dropna().pct_change()
    b = gold.resample("W-FRI").last().dropna().pct_change()
    df = pd.concat({"f": a, "b": b}, axis=1, join="inner").dropna()
    mask = df["b"] < 0
    if mask.sum() < 8:
        return None
    return float(df.loc[mask, "f"].mean() / (df.loc[mask, "b"].mean() + 1e-12))


# ---------------------------------------------------------------------------
# Hypotheses: signed weights (neg = lower is better). Not fitted.
# ---------------------------------------------------------------------------

HYPOTHESES: Dict[str, Dict[str, Dict[str, float]]] = {
    "Small Cap": {
        "H0_current": {
            "xirr_mean": 0.08, "xirr_p25": 0.20, "hold_ret": 0.06, "hold_dd": 0.10,
            "info_ratio": 0.04, "residual_alpha": 0.04, "downside_capture": -0.04,
            "rebound": 0.03, "ulcer": -0.08, "calmar": 0.04, "sortino": 0.06, "hold_vol": -0.08,
        },
        "H1_left_tail": {  # p25 + consistency beat the mean (mean-reversion after bulls)
            "xirr_p25": 0.28, "xirr_std": -0.12, "hold_dd": 0.12, "ulcer": -0.14,
            "hold_vol": -0.10, "sortino": 0.08, "rel_dd": -0.08, "hit_rate": 0.08,
        },
        "H2_quality_recovery": {  # after 2024-26 correction, quality + orderly rebound
            "xirr_p25": 0.18, "rel_dd": -0.14, "ulcer": -0.12, "rebound": 0.08,
            "downside_capture": -0.10, "capture_spread": 0.10, "sortino": 0.08,
            "hold_dd": 0.10, "cagr_1y": -0.10,  # fade last-12m heroes
        },
        "H3_anti_ir": {  # IR/alpha are luck after a bull; drop them
            "xirr_p25": 0.24, "hold_dd": 0.12, "ulcer": -0.14, "hold_vol": -0.12,
            "sortino": 0.10, "calmar": 0.08, "hit_rate": 0.10, "xirr_std": -0.10,
        },
        "H4_balanced_sc": {
            "xirr_p25": 0.22, "xirr_mean": 0.04, "hold_dd": 0.10, "ulcer": -0.12,
            "hold_vol": -0.10, "sortino": 0.08, "rel_dd": -0.08, "capture_spread": 0.08,
            "hit_rate": 0.06, "cagr_1y": -0.06, "xirr_std": -0.06,
        },
        # Combined after first bake-off: consistency + left tail, no IR chase.
        "H5_combined": {
            "xirr_p25": 0.22, "xirr_std": -0.14, "ulcer": -0.12, "rel_dd": -0.12,
            "hold_vol": -0.10, "hold_dd": 0.08, "sortino": 0.08, "calmar": 0.06,
            "xirr_mean": 0.04, "hit_rate": 0.04,
        },
    },
    "Mid Cap": {
        "H0_current": {
            "xirr_mean": 0.10, "xirr_p25": 0.16, "hold_ret": 0.10, "hold_dd": 0.10,
            "info_ratio": 0.10, "residual_alpha": 0.07, "downside_capture": -0.08,
            "rebound": 0.07, "ulcer": -0.05, "calmar": 0.04, "sortino": 0.04,
            "style_purity": 0.05, "style_drift": -0.04, "swing_elasticity": 0.03,
        },
        "H1_flex_style": {  # 2026-28 large/mid rotation: don't worship purity
            "xirr_p25": 0.18, "hold_dd": 0.12, "info_ratio": 0.08, "residual_alpha": 0.08,
            "downside_capture": -0.10, "ulcer": -0.08, "capture_spread": 0.08,
            "sortino": 0.06, "style_purity": 0.02, "style_drift": -0.02, "hit_rate": 0.08,
        },
        "H2_hold_quality": {
            "xirr_p25": 0.16, "hold_dd": 0.16, "hold_ret": 0.08, "ulcer": -0.10,
            "rel_dd": -0.10, "downside_capture": -0.10, "sortino": 0.08,
            "residual_alpha": 0.08, "cagr_1y": -0.06, "xirr_std": -0.08,
        },
        "H3_skill_not_cagr": {
            "residual_alpha": 0.14, "info_ratio": 0.10, "capture_spread": 0.10,
            "xirr_p25": 0.14, "hold_dd": 0.10, "ulcer": -0.08, "hit_rate": 0.10,
            "style_purity": 0.04, "cagr_1y": -0.08,
        },
        "H4_balanced_mc": {
            "xirr_p25": 0.16, "xirr_mean": 0.06, "hold_dd": 0.12, "residual_alpha": 0.08,
            "info_ratio": 0.06, "downside_capture": -0.08, "ulcer": -0.08,
            "capture_spread": 0.08, "sortino": 0.06, "style_purity": 0.03,
            "hit_rate": 0.07, "cagr_1y": -0.04, "xirr_std": -0.06,
        },
        # Combined: purity + IR + upside; safety was anti-predictive.
        "H5_combined": {
            "style_purity": 0.14, "info_ratio": 0.14, "upside_capture": 0.10,
            "swing_elasticity": 0.08, "capture_spread": 0.08, "residual_alpha": 0.08,
            "xirr_mean": 0.08, "hold_ret": 0.08, "xirr_p25": 0.08,
            "hold_dd": 0.04, "ulcer": -0.02,
        },
    },
    "Total Market": {
        "H0_current": {
            "xirr_mean": 0.11, "xirr_p25": 0.16, "hold_ret": 0.09, "hold_dd": 0.10,
            "info_ratio": 0.12, "residual_alpha": 0.08, "downside_capture": -0.08,
            "rebound": 0.06, "ulcer": -0.05, "calmar": 0.04, "sortino": 0.04,
            "batting": 0.07, "style_purity": 0.04,
        },
        "H1_value_defense": {  # large-cap/value catch-up + hold-phase
            "xirr_p25": 0.16, "hold_dd": 0.12, "downside_capture": -0.12,
            "residual_alpha": 0.10, "info_ratio": 0.08, "ulcer": -0.08,
            "batting": 0.10, "capture_spread": 0.08, "rel_dd": -0.08, "cagr_1y": -0.06,
        },
        "H2_process_skill": {
            "residual_alpha": 0.14, "info_ratio": 0.12, "batting": 0.12,
            "xirr_p25": 0.12, "hold_dd": 0.10, "capture_spread": 0.10,
            "ulcer": -0.08, "xirr_std": -0.08, "omega": 0.06,
        },
        "H3_anti_momentum": {
            "xirr_p25": 0.18, "hold_dd": 0.12, "ulcer": -0.10, "batting": 0.10,
            "downside_capture": -0.10, "cagr_1y": -0.12, "xirr_std": -0.10,
            "residual_alpha": 0.08, "sortino": 0.06, "rel_dd": -0.06,
        },
        "H4_balanced_tm": {
            "xirr_p25": 0.16, "xirr_mean": 0.06, "hold_dd": 0.10, "info_ratio": 0.08,
            "residual_alpha": 0.10, "downside_capture": -0.08, "ulcer": -0.07,
            "batting": 0.08, "capture_spread": 0.08, "cagr_1y": -0.05,
            "xirr_std": -0.06, "sortino": 0.04,
        },
        "H5_combined": {
            "residual_alpha": 0.12, "info_ratio": 0.12, "batting": 0.10,
            "xirr_p25": 0.12, "hold_dd": 0.08, "capture_spread": 0.10,
            "downside_capture": -0.08, "ulcer": -0.06, "xirr_mean": 0.06,
            "sortino": 0.06,
        },
    },
    "Multi Asset": {
        "H0_current": {
            "xirr_mean": 0.14, "xirr_p25": 0.14, "hold_ret": 0.08, "hold_dd": 0.12,
            "downside_capture": -0.12, "ulcer": -0.07, "sortino": 0.07, "calmar": 0.05,
            "gold_part": 0.06, "metals_overfit": -0.08, "alloc_stability": -0.05,
            "eq_down_prot": 0.11,
        },
        "H1_no_safety_trap": {  # Edelweiss-style 8% CAGR will lose a 24m race
            "return_adequacy": 0.18, "xirr_p25": 0.12, "hold_dd": 0.08,
            "gold_part": 0.08, "metals_overfit": -0.10, "eq_down_prot": 0.10,
            "ulcer": -0.06, "sortino": 0.08, "alloc_stability": -0.06, "omega": 0.06,
        },
        "H2_gold_not_silver": {  # structural gold bid; silver already melted up
            "gold_part": 0.16, "metals_overfit": -0.16, "silver_beta": -0.08,
            "eq_down_prot": 0.12, "return_adequacy": 0.12, "xirr_p25": 0.10,
            "hold_dd": 0.08, "alloc_stability": -0.08, "gold_down_part": -0.06,
        },
        "H3_process_sleeves": {  # long-history balanced multi-asset
            "xirr_p25": 0.14, "return_adequacy": 0.12, "eq_down_prot": 0.12,
            "gold_part": 0.08, "metals_overfit": -0.10, "alloc_stability": -0.10,
            "ulcer": -0.08, "sortino": 0.08, "n_windows": 0.08, "hold_dd": 0.08,
        },
        "H4_balanced_ma": {
            "return_adequacy": 0.14, "xirr_p25": 0.12, "xirr_mean": 0.06,
            "eq_down_prot": 0.10, "gold_part": 0.08, "metals_overfit": -0.10,
            "hold_dd": 0.08, "ulcer": -0.06, "sortino": 0.06, "alloc_stability": -0.06,
            "n_windows": 0.06, "cagr_1y": -0.04,
        },
        # Combined: kill the safety trap, keep gold, penalise late silver.
        "H5_combined": {
            "return_adequacy": 0.18, "xirr_p25": 0.12, "gold_part": 0.10,
            "metals_overfit": -0.12, "eq_down_prot": 0.10, "calmar": 0.08,
            "sortino": 0.08, "alloc_stability": -0.06, "n_windows": 0.06,
            "hold_dd": 0.04, "omega": 0.06,
        },
    },
}

# Naive competitor-like baselines (what Claude/GPT/Gemini often chase).
BASELINES = {
    "chase_1y": {"cagr_1y": 1.0},
    "chase_ir": {"info_ratio": 1.0},
    "chase_mean_xirr": {"xirr_mean": 1.0},
    "chase_safety": {"ulcer": -1.0, "hold_dd": 1.0, "downside_capture": -1.0},
    "chase_3y": {"cagr_3y": 1.0},
}


def panel_path(sector: str) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPERIMENTS_DIR / f"{sector}_hyp_panel.pkl"


def build_panel(provider, spec, step_months: int = 3, min_hist: int = 400, first_asof: Optional[str] = None) -> pd.DataFrame:
    uni = select_universe(provider.list_all_mf(), spec)
    benches_full = load_benches(provider, spec)
    navs, meta = {}, {}
    for _, row in uni.iterrows():
        mf_id = str(row["mfId"])
        try:
            nav = clean_nav(provider.get_mf_chart(mf_id))
        except Exception:
            continue
        if len(nav) < 60:
            continue
        navs[mf_id] = nav
        meta[mf_id] = {
            "name": row.get("name", mf_id),
            "aum": float(row.get("aum") or 0.0),
            "subsector": row.get("subsector", ""),
        }
    if not navs:
        return pd.DataFrame()

    end = max(s.index[-1] for s in navs.values())
    start = min(s.index[0] for s in navs.values())
    first = (start + pd.DateOffset(days=min_hist)).replace(day=1)
    last = (end - pd.DateOffset(months=24)).replace(day=1)
    if first_asof:
        first = max(first, pd.Timestamp(first_asof))
    asofs = month_range(first, last)[::step_months]
    log.info("%s panel: %d as-of, %d funds", spec.name, len(asofs), len(navs))

    rows = []
    for asof in asofs:
        n_ok = 0
        for mf_id, nav in navs.items():
            hist = slice_to(nav, asof)
            if calendar_days(hist) < min_hist:
                continue
            fut = simulate_sip_hold(nav, asof)
            if fut is None:
                continue
            bhist = {k: slice_to(v, asof) if len(v) else v for k, v in benches_full.items()}
            try:
                feats = features_for_fund(spec, hist, bhist, float(meta[mf_id]["aum"]))
                feats = extra_features(hist, bhist, feats)
            except Exception:
                continue
            feats["mfId"] = mf_id
            feats["name"] = meta[mf_id]["name"]
            feats["subsector"] = meta[mf_id]["subsector"]
            feats["asof"] = asof.strftime("%Y-%m-%d")
            feats["y_xirr"] = fut["xirr"]
            feats["y_hold_dd"] = fut["hold_dd"]
            rows.append(feats)
            n_ok += 1
        log.info("  asof %s n=%d", asof.date(), n_ok)
    return pd.DataFrame(rows)


def score_slice(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    z = pd.DataFrame(index=df.index)
    for f, w in weights.items():
        if f not in df.columns:
            z[f] = 0.0
            continue
        z[f] = robust_z(df[f]).fillna(0.0)
    raw = sum(z[f] * w for f, w in weights.items() if f in z.columns)
    lam = pd.to_numeric(df.get("shrink_lambda"), errors="coerce").fillna(0.35).clip(0.15, 1.0)
    return raw * lam


def eval_topk(panel: pd.DataFrame, weights: Dict[str, float], ks=(5, 8)) -> Dict:
    rows = []
    for asof, g in panel.groupby("asof"):
        if len(g) < 8:
            continue
        s = score_slice(g, weights)
        gg = g.assign(_s=s).sort_values("_s", ascending=False)
        rec = {"asof": asof, "n": len(g), "univ": float(g["y_xirr"].mean())}
        for k in ks:
            rec[f"top{k}"] = float(gg.head(k)["y_xirr"].mean())
            rec[f"top{k}_edge"] = rec[f"top{k}"] - rec["univ"]
        rec["bot5"] = float(gg.tail(5)["y_xirr"].mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return {"n_dates": 0}
    # Split last 8 dates as "recent" (closer to current regime).
    recent = out.tail(8)
    return {
        "n_dates": int(len(out)),
        "top5": float(out["top5"].mean()),
        "top8": float(out["top8"].mean()),
        "univ": float(out["univ"].mean()),
        "edge5": float(out["top5_edge"].mean()),
        "edge8": float(out["top8_edge"].mean()),
        "recent_edge5": float(recent["top5_edge"].mean()),
        "spread": float(out["top5"].mean() - out["bot5"].mean()),
        "winrate5": float((out["top5_edge"] > 0).mean()),
        "recent_winrate5": float((recent["top5_edge"] > 0).mean()),
    }


def feature_ic(panel: pd.DataFrame, feats: List[str], signed: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for asof, g in panel.groupby("asof"):
        if len(g) < 8:
            continue
        y = g["y_xirr"].astype(float)
        rec = {"asof": asof, "n": len(g)}
        for f in feats:
            if f not in g.columns:
                rec[f] = np.nan
                continue
            x = pd.to_numeric(g[f], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 8:
                rec[f] = np.nan
                continue
            ic, _ = spearmanr(x[m], y[m])
            sign = 1.0 if signed.get(f, 1.0) >= 0 else -1.0
            rec[f] = float(ic * sign) if ic == ic else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def run_sector(
    provider,
    sector: str,
    rebuild: bool,
    step_months: int = 3,
    first_asof: Optional[str] = None,
) -> pd.DataFrame:
    spec = SPECS[sector]
    path = panel_path(sector)
    if path.exists() and not rebuild:
        panel = pd.read_pickle(path)
        log.info("Loaded panel %s %s", sector, panel.shape)
    else:
        panel = build_panel(provider, spec, step_months=step_months, first_asof=first_asof)
        if panel.empty:
            log.error("Empty panel for %s", sector)
            return pd.DataFrame()
        panel.to_pickle(path)
        log.info("Wrote %s", path)

    # IC
    all_w = {}
    for h in HYPOTHESES[sector].values():
        all_w.update(h)
    all_w.update({"cagr_3y": 1.0, "cagr_1y": 1.0, "xirr_std": -1.0, "rel_dd": -1.0})
    ic = feature_ic(panel, sorted(set(list(all_w) + list(EXTRA_FEATS))), all_w)
    ic.to_csv(EXPERIMENTS_DIR / f"{sector}_hyp_ic.csv", index=False)
    means = ic.drop(columns=["asof", "n"]).mean().sort_values(ascending=False)
    recent = ic.tail(8).drop(columns=["asof", "n"]).mean().sort_values(ascending=False)
    log.info("Mean signed IC %s:\n%s", sector, means.head(15).to_string())
    log.info("Recent IC %s:\n%s", sector, recent.head(15).to_string())

    # Bake-off
    results = []
    pack = dict(HYPOTHESES[sector])
    pack.update({f"B_{k}": v for k, v in BASELINES.items()})
    for name, w in pack.items():
        ev = eval_topk(panel, w)
        ev["hypothesis"] = name
        ev["sector"] = sector
        results.append(ev)
        log.info(
            "  %-22s edge5=%.3f recent=%.3f win=%.2f top5=%.3f",
            name, ev.get("edge5", np.nan), ev.get("recent_edge5", np.nan),
            ev.get("winrate5", np.nan), ev.get("top5", np.nan),
        )
    rdf = pd.DataFrame(results).sort_values("recent_edge5", ascending=False)
    rdf.to_csv(EXPERIMENTS_DIR / f"{sector}_hyp_bakeoff.csv", index=False)
    return rdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default=None)
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=3, help="Months between walk-forward as-of dates")
    ap.add_argument("--first-asof", default=None, metavar="YYYY-MM-DD")
    args = ap.parse_args()
    provider = MfDataProvider(date=args.date)
    sectors = [args.sector] if args.sector else list(SPECS)
    frames = []
    for s in sectors:
        frames.append(run_sector(provider, s, args.rebuild, step_months=args.step, first_asof=args.first_asof))
    if frames:
        allr = pd.concat(frames, ignore_index=True)
        allr.to_csv(EXPERIMENTS_DIR / "hyp_bakeoff_all.csv", index=False)
        print(allr.to_string(index=False))


if __name__ == "__main__":
    main()
