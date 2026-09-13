import sys
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

print("Fund | Beta | R-squared | Tracking Error")

for mf_id in test_funds:
    chart = provider.get_mf_chart(mf_id, duration="5y")
    nav = clean_nav_to_series(chart)
    
    daily_ret = nav.pct_change().dropna()
    bench_ret = bench_nav.pct_change().dropna()
    
    df = pd.DataFrame({"fund": daily_ret, "bench": bench_ret}).dropna()
    
    cov = df.cov().iloc[0, 1]
    var = df["bench"].var()
    beta = cov / var if var > 0 else 0
    
    corr = df.corr().iloc[0, 1]
    r2 = corr ** 2
    
    te = (df["fund"] - df["bench"]).std() * np.sqrt(252)
    
    print(f"{mf_id} | {beta:.2f} | {r2:.2f} | {te:.2%}")
