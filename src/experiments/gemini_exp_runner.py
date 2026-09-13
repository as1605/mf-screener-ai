import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path('/Users/adityasingh/Desktop/Codes/Personal/mf-screener-ai/src')))
from mf_data_provider import MfDataProvider

provider = MfDataProvider()

# Fetch benchmark
bench_df = provider.get_index_chart("Small Cap")
bench_df["timestamp"] = pd.to_datetime(bench_df["timestamp"], utc=True)
bench_nav = bench_df.set_index("timestamp")["nav"]
bench_nav = bench_nav.resample("D").ffill().dropna()

# Fetch all small cap funds
df_all = provider.list_all_mf()
small_cap_df = df_all[df_all["subsector"] == "Small Cap Fund"]

results = []
for _, row in small_cap_df.iterrows():
    mf_id = row["mfId"]
    aum = row.get("aum", 0) or 0
    try:
        chart = provider.get_mf_chart(mf_id, duration='5y')
        chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
        nav = chart.set_index("timestamp")["nav"]
        nav = nav.resample("D").ffill().dropna()
        
        # Align with benchmark
        aligned = pd.concat([nav, bench_nav], axis=1, join="inner").dropna()
        if len(aligned) < 252 * 3:
            continue
            
        fund_r = aligned.iloc[:, 0].pct_change().dropna()
        bench_r = aligned.iloc[:, 1].pct_change().dropna()
        
        # Calculate daily beta
        cov = np.cov(fund_r, bench_r)[0, 1]
        var = np.var(bench_r)
        beta = cov / var if var > 0 else 0
        
        # Asymmetric Beta
        up_days = bench_r > 0
        down_days = bench_r < 0
        
        up_cov = np.cov(fund_r[up_days], bench_r[up_days])[0, 1] if sum(up_days)>1 else 0
        up_var = np.var(bench_r[up_days]) if sum(up_days)>1 else 0
        beta_up = up_cov / up_var if up_var > 0 else 0
        
        down_cov = np.cov(fund_r[down_days], bench_r[down_days])[0, 1] if sum(down_days)>1 else 0
        down_var = np.var(bench_r[down_days]) if sum(down_days)>1 else 0
        beta_down = down_cov / down_var if down_var > 0 else 0
        
        # 12m SIP + 12m hold simulation
        # Using 24 month rolling windows
        sim_returns = []
        months = nav.resample('ME').last().index
        for i in range(len(months) - 24):
            start = months[i]
            sip_end = months[i+11] # 12 months
            hold_end = months[i+23] # 24 months
            
            # SIP points
            sip_dates = months[i:i+12]
            sip_navs = nav.reindex(sip_dates, method='bfill')
            
            units = (1000 / sip_navs).sum()
            final_val = units * nav.reindex([hold_end], method='bfill').iloc[0]
            sim_return = (final_val - 12000) / 12000
            sim_returns.append(sim_return)
            
        if len(sim_returns) > 0:
            median_24m_ret = np.median(sim_returns)
            min_24m_ret = np.min(sim_returns)
        else:
            median_24m_ret = np.nan
            min_24m_ret = np.nan
            
        results.append({
            "mfId": mf_id,
            "name": row["name"],
            "aum": aum,
            "beta": beta,
            "beta_up": beta_up,
            "beta_down": beta_down,
            "median_24m_ret": median_24m_ret,
            "min_24m_ret": min_24m_ret
        })
        
    except Exception as e:
        pass
        
res_df = pd.DataFrame(results)
res_df.to_csv("/Users/adityasingh/Desktop/Codes/Personal/mf-screener-ai/results/experiments/gemini_exp1.csv", index=False)
print("Experiment 1 complete. Saved to gemini_exp1.csv")
