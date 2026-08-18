#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit of Grok SIP-hold scores. Read-only: no result writes."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

SRC = Path(__file__).resolve().parents[1]
ALG = SRC / "algorithms"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ALG) not in sys.path:
    sys.path.insert(0, str(ALG))

from grok_engine import (  # noqa: E402
    RISK_FREE,
    SIP_AMOUNT,
    SPECS,
    calendar_days,
    clean_nav,
    features_for_fund,
    information_ratio,
    load_benches,
    residual_alpha,
    rolling_siphold,
    score_frame,
    select_universe,
    simulate_sip_hold,
    trailing_cagr,
    xirr,
)
from mf_data_provider import MfDataProvider  # noqa: E402

TOP = {
    "Mid Cap": ["M_EDME", "M_WOCD", "M_INMI", "M_MIAI", "M_HDCMS"],
    "Small Cap": ["M_BOIL", "M_EDEC", "M_INEP", "M_IDFM", "M_NIMS"],
    "Multi Asset": ["M_NIPU", "M_TATI", "M_WOMT", "M_DSPUI", "M_QUNS", "M_SBTO"],
    "Total Market": ["M_MAHD", "M_AXVV", "M_DSPVE", "M_PARO", "M_ICIXD"],
}


def indie_xirr(cfs):
    start = cfs[0][0]
    timed = [((dt - start).days / 365.25, amt) for dt, amt in cfs]

    def npv(r):
        if r <= -1.0:
            return float("inf")
        return sum(cf / ((1.0 + r) ** t) for t, cf in timed)

    return float(brentq(npv, -0.95, 12.0, maxiter=80))


def indie_sip_hold(nav: pd.Series, start: pd.Timestamp) -> float:
    start = pd.Timestamp(start).replace(day=1)
    units = 0.0
    cfs = []
    for m in range(12):
        d = start + pd.DateOffset(months=m)
        prior = nav.loc[:d]
        px = float(prior.iloc[-1])
        units += SIP_AMOUNT / px
        cfs.append((pd.Timestamp(d), -SIP_AMOUNT))
    exit_dt = start + pd.DateOffset(months=24)
    prior = nav.loc[:exit_dt]
    exit_px = float(prior.iloc[-1])
    cfs.append((pd.Timestamp(exit_dt), units * exit_px))
    return indie_xirr(cfs)


def cond_style(fund, large, mid, small):
    w = fund.resample("W-FRI").last().dropna().pct_change()
    a = large.resample("W-FRI").last().dropna().pct_change()
    b = mid.resample("W-FRI").last().dropna().pct_change()
    c = small.resample("W-FRI").last().dropna().pct_change()
    df = pd.concat({"y": w, "L": a, "M": b, "S": c}, axis=1, join="inner").dropna()
    if len(df) < 40:
        return None
    X = np.column_stack([np.ones(len(df)), df["L"], df["M"], df["S"]])
    cond = float(np.linalg.cond(X))
    corr = df[["L", "M", "S"]].corr()
    coef, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    alpha = float((1.0 + coef[0]) ** 52 - 1.0)
    # ridge-ish: drop one factor, use mid-only + intercept as sanity
    X2 = np.column_stack([np.ones(len(df)), df["M"]])
    c2, *_ = np.linalg.lstsq(X2, df["y"].values, rcond=None)
    a2 = float((1.0 + c2[0]) ** 52 - 1.0)
    return {
        "n_weeks": len(df),
        "cond": cond,
        "corr_LM": float(corr.loc["L", "M"]),
        "corr_MS": float(corr.loc["M", "S"]),
        "corr_LS": float(corr.loc["L", "S"]),
        "alpha3": alpha,
        "alpha_mid_only": a2,
        "betas": [float(x) for x in coef[1:]],
    }


def audit_published(spec):
    pub = Path(SRC.parent / "results" / spec.outfile)
    if not pub.exists():
        print("  no published CSV")
        return None
    p = pd.read_csv(pub)
    print(f"  published n={len(p)} score {p['score'].min():.1f}-{p['score'].max():.1f}")
    # clones on published units (percent-scaled)
    if "return_adequacy" in p.columns and "xirr_mean" in p.columns:
        a = pd.to_numeric(p["return_adequacy"], errors="coerce")
        b = pd.to_numeric(p["xirr_mean"], errors="coerce") - RISK_FREE * 100.0
        mask = a.notna() & b.notna()
        mx = float((a[mask] - b[mask]).abs().max()) if mask.any() else float("nan")
        ic, _ = spearmanr(a[mask], pd.to_numeric(p.loc[mask, "xirr_mean"]))
        print(f"  PUB return_adequacy vs xirr_mean-RF max|diff|={mx:.4f} spearman={ic:.4f}")
    if "eq_down_prot" in p.columns and "downside_capture" in p.columns:
        a = pd.to_numeric(p["eq_down_prot"], errors="coerce")
        b = 1.0 - pd.to_numeric(p["downside_capture"], errors="coerce")
        mask = a.notna() & b.notna()
        mx = float((a[mask] - b[mask]).abs().max()) if mask.any() else float("nan")
        print(f"  PUB eq_down_prot vs 1-downside_capture max|diff|={mx:.4f}")
    print("  shrink vs calendar years:")
    for _, r in p.head(8).iterrows():
        n = float(r.get("n_windows") or 0)
        days = float(r.get("data_days") or 0)
        yrs = days / 365.25
        n_eff = max(0.0, yrs - 2.0)
        lam = n / (n + spec.shrink_k) if n + spec.shrink_k else 0
        lam_y = n_eff / (n_eff + 3.0) if n_eff + 3 else 0
        print(
            f"    {r['mfId']:8} nwin={n:5.0f} years={yrs:5.2f} n_eff={n_eff:4.2f} "
            f"lam={lam:.3f} lam_years_k3={lam_y:.3f} score={r['score']:.2f} {r['name']}"
        )
    return p


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    date = args[0] if args else "2026-08-15"
    provider = MfDataProvider(date=date)
    all_mf = provider.list_all_mf()
    print(f"AUDIT date={date} funds={len(all_mf)} full={full}")

    for sector, ids in TOP.items():
        spec = SPECS[sector]
        benches = load_benches(provider, spec)
        uni = select_universe(all_mf, spec)
        print("\n" + "=" * 88)
        print(f"{sector}: universe={len(uni)} shrink_k={spec.shrink_k}")
        print("weights:", spec.weights)
        pub = audit_published(spec)

        scored = None
        if full:
            rows = []
            for _, row in uni.iterrows():
                mid = str(row["mfId"])
                try:
                    nav = clean_nav(provider.get_mf_chart(mid))
                except Exception:
                    continue
                if len(nav) < 40:
                    continue
                feats = features_for_fund(spec, nav, benches, float(row.get("aum") or 0.0))
                feats["mfId"] = mid
                feats["name"] = row.get("name", mid)
                rows.append(feats)
            fdf = pd.DataFrame(rows)
            scored = score_frame(fdf, spec)
            if pub is not None:
                merged = pub[["mfId", "rank", "score"]].merge(
                    scored[["mfId", "rank", "score"]].rename(columns={"rank": "rrank", "score": "rscore"}),
                    on="mfId",
                )
                dscore = (merged["score"] - merged["rscore"]).abs()
                drank = (merged["rank"] - merged["rrank"]).abs()
                print(
                    f"  reproduce published: max|dscore|={dscore.max():.4f} "
                    f"max|drank|={drank.max()} n={len(merged)}"
                )

        for mf_id in ids:
            meta = all_mf[all_mf["mfId"] == mf_id]
            if meta.empty:
                print(f"  MISSING {mf_id}")
                continue
            row = meta.iloc[0]
            nav = clean_nav(provider.get_mf_chart(mf_id))
            feats = features_for_fund(spec, nav, benches, float(row.get("aum") or 0))
            panel = rolling_siphold(nav)
            print(f"\n  -- {mf_id} {row.get('name')} aum={row.get('aum')} sub={row.get('subsector')}")
            print(
                f"     days={calendar_days(nav)} navs={len(nav)} "
                f"{nav.index[0].date()} -> {nav.index[-1].date()} windows={len(panel)}"
            )
            c3 = trailing_cagr(nav, 3.0)
            c5 = trailing_cagr(nav, 5.0)
            print(
                f"     cagr3={None if c3 is None else round(c3*100,2)} "
                f"cagr5={None if c5 is None else round(c5*100,2)} "
                f"engine_c3={None if feats['cagr_3y'] is None else round(feats['cagr_3y']*100,2)}"
            )
            if not panel.empty:
                starts = panel["start"]
                print(
                    f"     window starts {starts.min().date()} -> {starts.max().date()} "
                    f"span_months={((starts.max()-starts.min()).days/30.44):.1f}"
                )
                print(
                    f"     xirr mean/p25/std "
                    f"{panel['xirr'].mean()*100:.2f}/"
                    f"{panel['xirr'].quantile(0.25)*100:.2f}/"
                    f"{panel['xirr'].std()*100:.2f}"
                )
                # independent XIRR on first, last, and a mid window
                for lab, st in [
                    ("first", starts.iloc[0]),
                    ("mid", starts.iloc[len(starts) // 2]),
                    ("last", starts.iloc[-1]),
                ]:
                    eng = simulate_sip_hold(nav, st)
                    ind = indie_sip_hold(nav, st)
                    print(
                        f"     {lab} start={pd.Timestamp(st).date()} "
                        f"engine={eng['xirr']*100:.4f} indie={ind*100:.4f} "
                        f"diff_bp={(eng['xirr']-ind)*10000:.2f}"
                    )
            ir = information_ratio(nav, benches["bench"])
            print(f"     IR engine={feats['info_ratio']} recompute={ir}")
            st = cond_style(nav, benches["large"], benches["mid"], benches["small"])
            if st:
                print(
                    f"     residual_alpha engine={feats['residual_alpha']} "
                    f"recompute3={st['alpha3']:.4f} mid_only={st['alpha_mid_only']:.4f} "
                    f"cond={st['cond']:.1f} corrLM={st['corr_LM']:.3f} "
                    f"betas={['%.2f'%x for x in st['betas']]} n={st['n_weeks']}"
                )
            print(
                f"     batting={feats.get('batting')} shrink={feats['shrink_lambda']:.3f} "
                f"down_cap={feats.get('downside_capture')} "
                f"style_purity={feats.get('style_purity')}"
            )
            # NAV sanity
            dnav = nav.pct_change().dropna()
            jumps = dnav[dnav.abs() > 0.12]
            print(f"     daily |ret|>12% count={len(jumps)} min={dnav.min():.3f} max={dnav.max():.3f}")


if __name__ == "__main__":
    main()
