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
    available_end = nav_subset.loc[:end_date]
    if available_end.empty:
        return None
    final_date = available_end.index[-1]
    final_nav = available_end.iloc[-1]
    final_value = units * final_nav
    cashflows.append((final_date, final_value))
    return _xirr(cashflows)

def get_max_drawdown(nav_series):
    roll_max = nav_series.cummax()
    drawdown = nav_series / roll_max - 1.0
    return drawdown.min()

def get_sortino(nav_series, risk_free_rate=0.065):
    daily_ret = nav_series.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    excess_ret = daily_ret - (risk_free_rate / 252)
    down_std = excess_ret[excess_ret < 0].std()
    if down_std == 0:
        return 0.0
    return (excess_ret.mean() / down_std) * np.sqrt(252)

def main():
    provider = MfDataProvider()
    bench_df = provider.get_index_chart('.NIMI150')
    bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp']).dt.normalize()
    bench_nav = bench_df.set_index('timestamp')['nav'].sort_index()
    bench_nav = bench_nav[~bench_nav.index.duplicated(keep='last')]
    
    df_all = provider.list_all_mf()
    mid_caps = df_all[df_all['subsector'] == 'Mid Cap Fund']
    
    results = []
    
    for _, row in mid_caps.iterrows():
        mfId = row['mfId']
        chart = provider.get_mf_chart(mfId, duration='5y')
        if chart.empty:
            continue
        chart['timestamp'] = pd.to_datetime(chart['timestamp']).dt.normalize()
        nav = chart.set_index('timestamp')['nav'].sort_index()
        nav = nav[~nav.index.duplicated(keep='last')]
        
        start_eval = nav.index[0]
        end_eval = nav.index[-1] - pd.DateOffset(months=24)
        fund_xirrs = []
        bench_xirrs = []
        
        if start_eval < end_eval:
            rolling_dates = pd.date_range(start=start_eval, end=end_eval, freq='MS')
            for d in rolling_dates:
                fx = calc_24m_sip_hold_xirr(nav, d)
                bx = calc_24m_sip_hold_xirr(bench_nav, d)
                if fx is not None and bx is not None:
                    fund_xirrs.append(fx)
                    bench_xirrs.append(bx)
                    
        win_rate = np.mean([1 if f > b else 0 for f, b in zip(fund_xirrs, bench_xirrs)]) if fund_xirrs else 0.0
        avg_xirr = np.mean(fund_xirrs) if fund_xirrs else 0.0
        min_xirr = np.min(fund_xirrs) if fund_xirrs else 0.0
        
        mdd = get_max_drawdown(nav)
        sortino = get_sortino(nav)
        
        results.append({
            'mfId': mfId,
            'name': row['name'],
            'win_rate': win_rate,
            'avg_xirr': avg_xirr,
            'min_xirr': min_xirr,
            'max_dd': mdd,
            'sortino': sortino
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(os.path.dirname(__file__), 'midcap_exp2_results.csv'), index=False)
    print(df_res.sort_values('sortino', ascending=False).head(15))

if __name__ == '__main__':
    main()
