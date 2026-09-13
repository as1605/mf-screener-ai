import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from mf_data_provider import MfDataProvider

def clean_nav_to_series(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.set_index("timestamp")["nav"]

def calculate_xirr(cashflows):
    if len(cashflows) < 2: return None
    start_date = cashflows[0][0]
    years = [(cf[0] - start_date).days / 365.25 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    def npv(r):
        if r <= -1.0: return float('inf')
        return sum(a / ((1 + r) ** y) for a, y in zip(amounts, years))
    low, high = -0.9999, 100.0
    for _ in range(100):
        mid = (low + high) / 2.0
        try: npv_mid = npv(mid)
        except: return None
        if abs(npv_mid) < 1e-4: return mid
        if npv(low) * npv_mid < 0: high = mid
        else: low = mid
    return (low + high) / 2.0

def sim_24m(nav, start_date):
    end_date = start_date + relativedelta(months=24)
    if nav.index[-1] < end_date: return None
    cashflows, total_units = [], 0.0
    for i in range(12):
        td = start_date + relativedelta(months=i)
        av = nav[nav.index >= td]
        if av.empty or (av.index[0] - td).days > 15: return None
        total_units += 1000.0 / av.iloc[0]
        cashflows.append((av.index[0], -1000.0))
    av_end = nav[nav.index <= end_date]
    if av_end.empty or (end_date - av_end.index[-1]).days > 15: return None
    cashflows.append((av_end.index[-1], total_units * av_end.iloc[-1]))
    return calculate_xirr(cashflows)

def ulcer_index(nav):
    roll_max = nav.cummax()
    drawdown = (nav / roll_max) - 1.0
    return np.sqrt(np.mean(drawdown ** 2)) * 100

def calmar_ratio(nav):
    days = (nav.index[-1] - nav.index[0]).days
    if days == 0: return np.nan
    years = days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    roll_max = nav.cummax()
    drawdown = (nav / roll_max) - 1.0
    mdd = abs(drawdown.min())
    return cagr / mdd if mdd > 0 else np.nan

provider = MfDataProvider()
bench_df = provider.get_index_chart("_NIFTY500")
bench_nav = clean_nav_to_series(bench_df)

test_funds = ["M_AXVV", "M_ICIXD", "M_WOCE", "M_INXO", "M_CAFF", "M_PARO", "M_MAHD"]

results = []
for mf_id in test_funds:
    chart = provider.get_mf_chart(mf_id, duration="5y")
    nav = clean_nav_to_series(chart)
    
    st = nav.index[0].replace(day=1) + relativedelta(months=1)
    xirrs, b_xirrs = [], []
    while st + relativedelta(months=24) <= nav.index[-1]:
        x = sim_24m(nav, st)
        bx = sim_24m(bench_nav, st)
        if x is not None and bx is not None:
            xirrs.append(x)
            b_xirrs.append(bx)
        st += relativedelta(months=1)
        
    mean_xirr = np.mean(xirrs) if xirrs else 0
    win_rate = sum(1 for x, bx in zip(xirrs, b_xirrs) if x > bx) / len(xirrs) if xirrs else 0
    
    daily_ret = nav.pct_change().dropna()
    bench_ret = bench_nav.pct_change().dropna()
    df = pd.DataFrame({"fund": daily_ret, "bench": bench_ret}).dropna()
    down_mask = df["bench"] < 0
    if down_mask.sum() > 0:
        fund_down = (1 + df.loc[down_mask, "fund"]).prod() ** (252 / down_mask.sum()) - 1
        bench_down = (1 + df.loc[down_mask, "bench"]).prod() ** (252 / down_mask.sum()) - 1
        d_cap = fund_down / bench_down if bench_down < 0 else np.nan
    else:
        d_cap = np.nan
        
    ulcer = ulcer_index(nav)
    calmar = calmar_ratio(nav)
    
    results.append({
        "mf_id": mf_id,
        "mean_xirr": mean_xirr,
        "win_rate": win_rate,
        "d_cap": d_cap,
        "ulcer": ulcer,
        "calmar": calmar
    })

df = pd.DataFrame(results)

SCORE_WEIGHTS = {
    "mean_xirr": (True, 0.25),
    "win_rate": (True, 0.20),
    "calmar": (True, 0.20),
    "ulcer": (False, 0.15),
    "d_cap": (False, 0.20),
}
df["score"] = 0.0
for col, (higher, w) in SCORE_WEIGHTS.items():
    pctl = df[col].rank(pct=True)
    if not higher: pctl = 1 - pctl
    df["score"] += pctl * 100 * w

df = df.sort_values("score", ascending=False)
print(df.to_string(index=False))
