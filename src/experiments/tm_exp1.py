import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

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

provider = MfDataProvider()
bench_df = provider.get_index_chart("_NIFTY500")
bench_nav = clean_nav_to_series(bench_df)

print("NIFTY 500 Daily Series length:", len(bench_nav))

# Grab a few top funds from the current ranking to test
test_funds = ["M_AXVV", "M_ICIXD", "M_WOCE", "M_INXO"]

for mf_id in test_funds:
    chart = provider.get_mf_chart(mf_id, duration="5y")
    nav = clean_nav_to_series(chart)
    print(f"Fund {mf_id} - Length: {len(nav)}")
    
    # Calculate daily returns
    daily_ret = nav.pct_change().dropna()
    bench_ret = bench_nav.pct_change().dropna()
    
    # Align
    df = pd.DataFrame({"fund": daily_ret, "bench": bench_ret}).dropna()
    
    # Info ratio
    tracking_diff = df["fund"] - df["bench"]
    te = tracking_diff.std() * np.sqrt(252)
    alpha = tracking_diff.mean() * 252
    ir = alpha / te if te > 0 else 0
    
    # Sortino (daily)
    downside = df["fund"][df["fund"] < 0]
    down_dev = downside.std() * np.sqrt(252)
    sortino = (df["fund"].mean() * 252 - 0.065) / down_dev if down_dev > 0 else 0
    
    # Rolling 24m XIRR
    st = nav.index[0].replace(day=1) + relativedelta(months=1)
    xirrs = []
    while st + relativedelta(months=24) <= nav.index[-1]:
        x = sim_24m(nav, st)
        if x is not None: xirrs.append(x)
        st += relativedelta(months=1)
        
    print(f"  IR: {ir:.2f}, Sortino: {sortino:.2f}, Mean 24m XIRR: {np.mean(xirrs) if xirrs else 0:.2%}, Min XIRR: {np.min(xirrs) if xirrs else 0:.2%}")

