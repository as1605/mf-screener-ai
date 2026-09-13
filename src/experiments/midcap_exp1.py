import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import brentq
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.mf_data_provider import MfDataProvider

def _xirr(cashflows):
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    days = np.array([(cf[0] - t0).days for cf in cashflows], dtype=float)
    amts = np.array([cf[1] for cf in cashflows], dtype=float)
    if np.all(amts >= 0) or np.all(amts <= 0):
        return None
    def npv(rate):
        return float(np.sum(amts / (1.0 + rate) ** (days / 365.0)))
    try:
        return float(brentq(npv, -0.99, 10.0, xtol=1e-6, maxiter=200))
    except:
        return None

def calc_24m_sip_hold_xirr(nav_series, start_date, monthly_amount=10000):
    """
    12 months SIP + 12 months Hold. Total 24 months.
    Returns the XIRR of this specific scenario.
    """
    end_date = start_date + pd.DateOffset(months=24)
    nav_subset = nav_series.loc[start_date:end_date]
    if nav_subset.empty:
        return None
        
    buys = pd.date_range(start=start_date, periods=12, freq='MS')
    units = 0.0
    cashflows = []
    
    for buy_date in buys:
        available = nav_subset.loc[buy_date:]
        if available.empty:
            continue
        actual_date = available.index[0]
        nav_val = available.iloc[0]
        units += monthly_amount / nav_val
        cashflows.append((actual_date, -monthly_amount))
        
    if len(cashflows) < 12:
        return None
        
    # Evaluate at month 24
    available_end = nav_subset.loc[:end_date]
    if available_end.empty:
        return None
    
    final_date = available_end.index[-1]
    final_nav = available_end.iloc[-1]
    final_value = units * final_nav
    cashflows.append((final_date, final_value))
    
    return _xirr(cashflows)

def main():
    provider = MfDataProvider()
    bench_df = provider.get_index_chart('.NIMI150')
    bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp']).dt.normalize()
    bench_nav = bench_df.set_index('timestamp')['nav'].sort_index()
    bench_nav = bench_nav[~bench_nav.index.duplicated(keep='last')]
    
    df_all = provider.list_all_mf()
    mid_caps = df_all[df_all['subsector'] == 'Mid Cap Fund']
    
    results = []
    
    # Just test first 10 funds for speed
    for _, row in mid_caps.head(10).iterrows():
        mfId = row['mfId']
        chart = provider.get_mf_chart(mfId, duration='5y')
        if chart.empty:
            continue
            
        chart['timestamp'] = pd.to_datetime(chart['timestamp']).dt.normalize()
        nav = chart.set_index('timestamp')['nav'].sort_index()
        nav = nav[~nav.index.duplicated(keep='last')]
        
        # Calculate Rolling 24m XIRRs
        start_eval = nav.index[0]
        end_eval = nav.index[-1] - pd.DateOffset(months=24)
        if start_eval >= end_eval:
            continue
            
        rolling_dates = pd.date_range(start=start_eval, end=end_eval, freq='MS')
        
        fund_xirrs = []
        bench_xirrs = []
        
        for d in rolling_dates:
            fx = calc_24m_sip_hold_xirr(nav, d)
            bx = calc_24m_sip_hold_xirr(bench_nav, d)
            
            if fx is not None and bx is not None:
                fund_xirrs.append(fx)
                bench_xirrs.append(bx)
                
        if not fund_xirrs:
            continue
            
        win_rate = np.mean([1 if f > b else 0 for f, b in zip(fund_xirrs, bench_xirrs)])
        avg_xirr = np.mean(fund_xirrs)
        min_xirr = np.min(fund_xirrs)
        
        # Calculate downside capture using daily data
        # Align daily returns
        aligned = pd.concat([nav.pct_change(), bench_nav.pct_change()], axis=1).dropna()
        aligned.columns = ['fund', 'bench']
        
        down_market = aligned[aligned['bench'] < 0]
        down_cap = down_market['fund'].mean() / down_market['bench'].mean() if len(down_market) > 0 else 1.0
        
        results.append({
            'name': row['name'],
            'win_rate': win_rate,
            'avg_xirr': avg_xirr,
            'min_xirr': min_xirr,
            'down_cap': down_cap
        })
        
    df_res = pd.DataFrame(results)
    print(df_res)

if __name__ == '__main__':
    main()
