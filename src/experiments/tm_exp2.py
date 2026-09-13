import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from mf_data_provider import MfDataProvider

def clean_nav_to_series(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out = out.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.set_index("timestamp")["nav"]

provider = MfDataProvider()
bench_df = provider.get_index_chart("_NIFTY500")
bench_nav = clean_nav_to_series(bench_df)

test_funds = ["M_AXVV", "M_ICIXD", "M_WOCE", "M_INXO", "M_CAFF", "M_PARO", "M_MAHD"]

def calc_max_drawdown(nav):
    roll_max = nav.cummax()
    drawdown = (nav / roll_max) - 1.0
    return drawdown.min()

print("Fund | Daily Alpha | Tracking Error | Downside Capture | Max Drawdown")

for mf_id in test_funds:
    chart = provider.get_mf_chart(mf_id, duration="5y")
    nav = clean_nav_to_series(chart)
    
    daily_ret = nav.pct_change().dropna()
    bench_ret = bench_nav.pct_change().dropna()
    
    df = pd.DataFrame({"fund": daily_ret, "bench": bench_ret}).dropna()
    if df.empty: continue
    
    # Active
    diff = df["fund"] - df["bench"]
    te = diff.std() * np.sqrt(252)
    alpha = diff.mean() * 252
    
    # Downside capture
    down_mask = df["bench"] < 0
    if down_mask.sum() > 0:
        fund_down = (1 + df.loc[down_mask, "fund"]).prod() ** (252 / down_mask.sum()) - 1
        bench_down = (1 + df.loc[down_mask, "bench"]).prod() ** (252 / down_mask.sum()) - 1
        d_cap = fund_down / bench_down if bench_down < 0 else np.nan
    else:
        d_cap = np.nan
        
    mdd = calc_max_drawdown(nav)
    
    print(f"{mf_id} | {alpha:.2%} | {te:.2%} | {d_cap:.2f} | {mdd:.2%}")

