import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from scipy.stats import spearmanr
from scipy.optimize import brentq

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore")

# Define Constants
RISK_FREE_RATE = 0.065
SECTOR = "Mid Cap"
MODEL = "Gemini"

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

def calc_sip_xirr(nav_series, start_date, end_date, monthly_amount=10000):
    nav_subset = nav_series.loc[start_date:end_date]
    if nav_subset.empty:
        return None
    
    buys = pd.date_range(start=start_date, end=end_date, freq='MS')
    units = 0.0
    cashflows = []
    
    for buy_date in buys:
        # Find next available NAV
        available = nav_subset.loc[buy_date:]
        if available.empty:
            continue
        actual_date = available.index[0]
        nav_val = available.iloc[0]
        units += monthly_amount / nav_val
        cashflows.append((actual_date, -monthly_amount))
        
    if len(cashflows) < 6:
        return None
        
    final_date = nav_subset.index[-1]
    final_nav = nav_subset.iloc[-1]
    final_value = units * final_nav
    cashflows.append((final_date, final_value))
    
    return _xirr(cashflows)

def get_swing_elasticity(fund_nav, bench_nav):
    if len(bench_nav) < 250:
        return 0.0, 0.0
    
    # Calculate weekly returns to smooth out daily noise and improve alignment
    fund_weekly = fund_nav.resample('W').last().dropna()
    bench_weekly = bench_nav.resample('W').last().dropna()
    
    fund_ret = fund_weekly.pct_change().dropna()
    bench_ret = bench_weekly.reindex(fund_weekly.index).ffill().pct_change().dropna()
    
    aligned = pd.concat([fund_ret, bench_ret], axis=1).dropna()
    aligned.columns = ['fund', 'bench']
    
    if len(aligned) < 50: # Weekly data, 50 weeks is ~1 year
        return 0.0, 0.0
        
    down_market = aligned[aligned['bench'] < 0]
    down_capture = down_market['fund'].mean() / down_market['bench'].mean() if len(down_market) > 0 and down_market['bench'].mean() < 0 else 1.0
    
    up_market = aligned[aligned['bench'] > 0]
    up_capture = up_market['fund'].mean() / up_market['bench'].mean() if len(up_market) > 0 and up_market['bench'].mean() > 0 else 1.0
    
    swing_elasticity = up_capture - down_capture
    # print(f"DEBUG: len(aligned) = {len(aligned)}")
    return float(swing_elasticity), float(down_capture)

def calc_momentum(fund_nav, bench_nav, days):
    cutoff = fund_nav.index[-1] - timedelta(days=days)
    f_subset = fund_nav.loc[cutoff:]
    
    b_subset = bench_nav.reindex(f_subset.index).ffill()
    
    if len(f_subset) < 20 or len(b_subset) < 20:
        return 0.0
        
    f_ret = f_subset.pct_change().dropna()
    b_ret = b_subset.pct_change().dropna()
    
    aligned = pd.concat([f_ret, b_ret], axis=1).dropna()
    if aligned.empty:
        return 0.0
        
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(252)

def calculate_fund_metrics(fund_nav, bench_nav, eval_date):
    fund_nav = fund_nav.loc[:eval_date]
    bench_nav = bench_nav.loc[:eval_date]
    
    if len(fund_nav) < 250:
        return None
        
    # Consistency (Rolling 1Y SIP Win Rate over last 2 years)
    start_eval = eval_date - timedelta(days=2*365)
    rolling_dates = pd.date_range(start=start_eval, end=eval_date-timedelta(days=365), freq='ME')
    
    wins = 0
    total = 0
    for d in rolling_dates:
        fx = calc_sip_xirr(fund_nav, d, d + timedelta(days=365))
        bx = calc_sip_xirr(bench_nav, d, d + timedelta(days=365))
        if fx is not None and bx is not None:
            total += 1
            if fx > bx:
                wins += 1
                
    win_rate = (wins / total) if total > 0 else 0.5
    
    # Swing & Quality
    swing_elasticity, down_capture = get_swing_elasticity(fund_nav.loc[eval_date-timedelta(days=3*365):], 
                                                          bench_nav.loc[eval_date-timedelta(days=3*365):])
                                                          
    # Momentum (3M, 6M)
    ir_3m = calc_momentum(fund_nav, bench_nav, 90)
    ir_6m = calc_momentum(fund_nav, bench_nav, 180)
    
    # Sortino Ratio
    f_weekly = fund_nav.resample('W').last().dropna()
    f_ret = f_weekly.pct_change().dropna()
    if not f_ret.empty:
        excess = f_ret - (RISK_FREE_RATE/52)
        down_std = excess[excess < 0].std()
        sortino = (excess.mean() / down_std) * np.sqrt(52) if down_std > 0 else 0
    else:
        sortino = 0.0
        
    return {
        'win_rate': win_rate,
        'swing_elasticity': swing_elasticity,
        'down_capture': down_capture,
        'ir_3m': ir_3m,
        'ir_6m': ir_6m,
        'sortino': sortino
    }

def main(date=None):
    print("=" * 80)
    print("MID CAP GEMINI ALGO - Dynamic Self-Tuning")
    print("=" * 80)
    
    provider = MfDataProvider(date=date)
    bench_df = provider.get_index_chart('.NIMI150')
    if bench_df.empty:
        print("Failed to get benchmark data.")
        return
        
    bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp']).dt.normalize()
    bench_nav = bench_df.set_index('timestamp')['nav'].sort_index()
    bench_nav = bench_nav[~bench_nav.index.duplicated(keep='last')]
    
    df_all = provider.list_all_mf()
    mid_caps = df_all[df_all['subsector'] == 'Mid Cap Fund'].copy()
    
    funds_data = {}
    for _, row in mid_caps.iterrows():
        mfId = row['mfId']
        chart = provider.get_mf_chart(mfId)
        if not chart.empty:
            chart['timestamp'] = pd.to_datetime(chart['timestamp']).dt.normalize()
            nav = chart.set_index('timestamp')['nav'].sort_index()
            nav = nav[~nav.index.duplicated(keep='last')]
            funds_data[mfId] = {
                'name': row['name'],
                'aum': row['aum'],
                'nav': nav
            }
            
    print(f"Loaded {len(funds_data)} mid cap funds.")
    
    current_date = bench_nav.index[-1]
    
    # Dynamic Tuning Engine
    # We will evaluate at t-1Y and t-2Y to predict t-0 and t-1Y respectively
    eval_points = [current_date - timedelta(days=365), current_date - timedelta(days=2*365)]
    
    historical_metrics = {pt: {} for pt in eval_points}
    forward_returns = {pt: {} for pt in eval_points}
    
    print("Running dynamic tuning engine...")
    for pt in eval_points:
        for mfId, data in funds_data.items():
            metrics = calculate_fund_metrics(data['nav'], bench_nav, pt)
            if metrics is not None:
                historical_metrics[pt][mfId] = metrics
                
                # Calculate forward 1Y SIP return
                fwd_xirr = calc_sip_xirr(data['nav'], pt, pt + timedelta(days=365))
                if fwd_xirr is not None:
                    forward_returns[pt][mfId] = fwd_xirr
                    
    # Calculate ICs
    features = ['win_rate', 'swing_elasticity', 'down_capture', 'ir_3m', 'ir_6m', 'sortino']
    feature_ics = {f: [] for f in features}
    
    for pt in eval_points:
        h_mets = historical_metrics[pt]
        f_rets = forward_returns[pt]
        
        common_funds = list(set(h_mets.keys()).intersection(set(f_rets.keys())))
        if len(common_funds) < 10:
            continue
            
        y = [f_rets[f] for f in common_funds]
        for feat in features:
            x = [h_mets[f][feat] for f in common_funds]
            if np.std(x) > 1e-6 and np.std(y) > 1e-6:
                ic, _ = spearmanr(x, y)
                if not np.isnan(ic):
                    feature_ics[feat].append(ic)
                    
    # Average ICs and compute weights
    weights = {}
    print("\nDynamic Weights (based on historical IC):")
    for feat in features:
        avg_ic = np.mean(feature_ics[feat]) if len(feature_ics[feat]) > 0 else 0
        # down_capture should be negatively correlated, so invert it
        if feat == 'down_capture':
            avg_ic = -avg_ic
        
        # We only assign weight if IC > 0 (meaning it was predictive)
        weight = max(0.01, avg_ic) # floor at 0.01 to keep all factors slightly active
        weights[feat] = weight
        print(f"  {feat}: IC = {avg_ic:.3f} -> Weight = {weight:.3f}")
        
    total_weight = sum(weights.values())
    for feat in weights:
        weights[feat] /= total_weight
        
    # Now calculate current metrics
    print("\nCalculating current metrics and scores...")
    results = []
    
    for mfId, data in funds_data.items():
        metrics = calculate_fund_metrics(data['nav'], bench_nav, current_date)
        if metrics is None:
            continue
            
        # AUM Penalty (Size trap penalty)
        aum = data['aum'] if pd.notna(data['aum']) else 0
        aum_penalty = 1.0
        if aum > 40000:
            aum_penalty = 0.85
        elif aum > 25000:
            aum_penalty = 0.90
        elif aum > 15000:
            aum_penalty = 0.95
            
        # History confidence penalty
        days = (data['nav'].index[-1] - data['nav'].index[0]).days
        conf = 1.0
        if days < 3 * 365:
            conf = 0.8 # Penalize short history
            
        res = {
            'mfId': mfId,
            'name': data['name'],
            'data_days': days,
            'aum': aum
        }
        res.update(metrics)
        
        # Calculate CAGRs
        start_nav = data['nav'].iloc[0]
        end_nav = data['nav'].iloc[-1]
        
        for y in [3, 5]:
            ago = current_date - timedelta(days=y*365)
            past_nav = data['nav'].loc[:ago]
            if not past_nav.empty:
                val = past_nav.iloc[-1]
                cagr = ((end_nav / val) ** (1/y) - 1) * 100
                res[f'cagr_{y}y'] = cagr
            else:
                res[f'cagr_{y}y'] = 0.0
                
        results.append(res)
        
    df_res = pd.DataFrame(results)
    
    # Rank Normalize features
    for feat in features:
        if feat == 'down_capture':
            df_res[f'{feat}_score'] = df_res[feat].rank(ascending=False, pct=True) * 100
        else:
            df_res[f'{feat}_score'] = df_res[feat].rank(ascending=True, pct=True) * 100
            
    # Compute composite score
    df_res['raw_score'] = 0.0
    for feat in features:
        df_res['raw_score'] += df_res[f'{feat}_score'] * weights[feat]
        
    # Apply penalties
    # AUM Penalty logic inline
    aum_penalties = []
    for aum in df_res['aum']:
        p = 1.0
        if aum > 40000: p = 0.85
        elif aum > 25000: p = 0.90
        elif aum > 15000: p = 0.95
        aum_penalties.append(p)
        
    df_res['aum_penalty'] = aum_penalties
    
    conf_penalties = []
    for d in df_res['data_days']:
        p = 1.0
        if d < 3 * 365: p = 0.8
        conf_penalties.append(p)
        
    df_res['conf_penalty'] = conf_penalties
    
    df_res['score'] = df_res['raw_score'] * df_res['aum_penalty'] * df_res['conf_penalty']
    df_res['rank'] = df_res['score'].rank(ascending=False).astype(int)
    
    df_res = df_res.sort_values('rank')
    
    cols = ['mfId', 'name', 'rank', 'score', 'data_days', 'cagr_3y', 'cagr_5y', 'win_rate', 'swing_elasticity']
    final_df = df_res[cols].copy()
    
    # Format
    final_df['score'] = final_df['score'].round(2)
    final_df['cagr_3y'] = final_df['cagr_3y'].round(2).astype(str) + '%'
    final_df['cagr_5y'] = final_df['cagr_5y'].round(2).astype(str) + '%'
    final_df['win_rate'] = (final_df['win_rate'] * 100).round(2).astype(str) + '%'
    final_df['swing_elasticity'] = final_df['swing_elasticity'].round(3)
    
    # Output
    out_dir = os.path.join(os.path.dirname(__file__), '../../results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{SECTOR}_{MODEL}.csv')
    
    final_df.to_csv(out_path, index=False)
    print(f"\\nResults saved to {out_path}")
    print(final_df.head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Cached data folder date")
    args = parser.parse_args()
    main(args.date)
