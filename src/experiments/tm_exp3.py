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

test_funds = ["M_AXVV", "M_ICIXD", "M_WOCE", "M_INXO", "M_CAFF", "M_PARO", "M_MAHD"]

print("Fund | CAGR | Max Drawdown | Calmar Ratio | Ulcer Index")

for mf_id in test_funds:
    chart = provider.get_mf_chart(mf_id, duration="5y")
    nav = clean_nav_to_series(chart)
    
    if nav.empty or len(nav) < 252: continue
    
    # CAGR
    days = (nav.index[-1] - nav.index[0]).days
    if days == 0: continue
    years = days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    
    # Max Drawdown & Ulcer Index
    roll_max = nav.cummax()
    drawdown = (nav / roll_max) - 1.0
    mdd = abs(drawdown.min())
    
    # Ulcer Index = sqrt( mean( drawdown^2 ) )
    ulcer = np.sqrt(np.mean(drawdown ** 2)) * 100
    
    calmar = cagr / mdd if mdd > 0 else np.nan
    
    print(f"{mf_id} | {cagr:.2%} | {mdd:.2%} | {calmar:.2f} | {ulcer:.2f}")
