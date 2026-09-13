import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from scipy.optimize import brentq
from scipy.stats import linregress

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore")

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

def get_sortino_and_asymmetry(fund_nav, bench_nav, risk_free_rate=0.065):
    f_ret = fund_nav.pct_change().dropna()
    b_ret = bench_nav.pct_change().dropna()
    aligned = pd.concat([f_ret, b_ret], axis=1).dropna()
    if aligned.empty:
        return 0.0, 1.0
    aligned.columns = ['fund', 'bench']
    
    excess_ret = aligned['fund'] - (risk_free_rate / 252)
    down_std = excess_ret[excess_ret < 0].std()
    sortino = 0.0
    if down_std > 0:
        sortino = (excess_ret.mean() / down_std) * np.sqrt(252)
        
    up_market = aligned[aligned['bench'] > 0]
    down_market = aligned[aligned['bench'] < 0]
    
    up_beta = up_market['fund'].mean() / up_market['bench'].mean() if len(up_market) > 0 and up_market['bench'].mean() != 0 else 1.0
    down_beta = down_market['fund'].mean() / down_market['bench'].mean() if len(down_market) > 0 and down_market['bench'].mean() != 0 else 1.0
    
    asym = up_beta / down_beta if down_beta > 0 else 1.0
    return sortino, asym

def calc_ir(f_ret, b_ret):
    excess = f_ret - b_ret
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)

def calc_vol(f_ret):
    return f_ret.std() * np.sqrt(252)

def calc_down_cap(f_ret, b_ret):
    aligned = pd.concat([f_ret, b_ret], axis=1).dropna()
    aligned.columns = ['fund', 'bench']
    down = aligned[aligned['bench'] < 0]
    if len(down) == 0 or down['bench'].mean() == 0:
        return 1.0
    return down['fund'].mean() / down['bench'].mean()

def calc_appraisal_and_r2(f_ret, b_ret):
    import pandas as pd
    import numpy as np
    aligned = pd.concat([f_ret, b_ret], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0, 0.0
    Y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1]
    
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    
    alpha_daily = intercept
    residuals = Y - (intercept + slope * X)
    idio_risk_daily = residuals.std()
    
    alpha_ann = alpha_daily * 252
    idio_risk_ann = idio_risk_daily * np.sqrt(252)
    
    appraisal = alpha_ann / idio_risk_ann if idio_risk_ann > 0 else 0.0
    r_squared = r_value**2
    return appraisal, r_squared

def calculate_fund_metrics(fund_nav, bench_nav, current_date):
    fund_nav = fund_nav.loc[:current_date]
    bench_nav = bench_nav.loc[:current_date]
    
    if len(fund_nav) < 250:
        return None
        
    start_eval = fund_nav.index[0]
    end_eval = fund_nav.index[-1] - pd.DateOffset(months=24)
    
    fund_xirrs = []
    bench_xirrs = []
    
    if start_eval < end_eval:
        rolling_dates = pd.date_range(start=start_eval, end=end_eval, freq='MS')
        for d in rolling_dates:
            fx = calc_24m_sip_hold_xirr(fund_nav, d)
            bx = calc_24m_sip_hold_xirr(bench_nav, d)
            if fx is not None and bx is not None:
                fund_xirrs.append(fx)
                bench_xirrs.append(bx)
                
    win_rate = np.mean([1 if f > b else 0 for f, b in zip(fund_xirrs, bench_xirrs)]) if fund_xirrs else 0.0
    min_xirr = np.min(fund_xirrs) if fund_xirrs else -1.0
    avg_xirr = np.mean(fund_xirrs) if fund_xirrs else 0.0
    
    mdd = get_max_drawdown(fund_nav)
    sortino, asym = get_sortino_and_asymmetry(fund_nav, bench_nav)
    
    f_ret = fund_nav.pct_change().dropna()
    b_ret = bench_nav.pct_change().dropna()
    
    date_6m = current_date - pd.DateOffset(months=6)
    date_1y = current_date - pd.DateOffset(years=1)
    date_3y = current_date - pd.DateOffset(years=3)
    
    f_ret_6m = f_ret.loc[date_6m:]
    b_ret_6m = b_ret.loc[date_6m:]
    f_ret_1y = f_ret.loc[date_1y:]
    b_ret_1y = b_ret.loc[date_1y:]
    f_ret_3y = f_ret.loc[date_3y:]
    b_ret_3y = b_ret.loc[date_3y:]
    
    ir_6m = calc_ir(f_ret_6m, b_ret_6m) if not f_ret_6m.empty else 0
    ir_3y = calc_ir(f_ret_3y, b_ret_3y) if not f_ret_3y.empty else 0
    vol_6m = calc_vol(f_ret_6m) if not f_ret_6m.empty else 0
    vol_3y = calc_vol(f_ret_3y) if not f_ret_3y.empty else 0
    dc_6m = calc_down_cap(f_ret_6m, b_ret_6m) if not f_ret_6m.empty else 1.0
    dc_3y = calc_down_cap(f_ret_3y, b_ret_3y) if not f_ret_3y.empty else 1.0
    
    appraisal, r2 = calc_appraisal_and_r2(f_ret_1y, b_ret_1y)
    
    return {
        'win_rate': win_rate,
        'min_xirr': min_xirr,
        'max_dd': mdd,
        'sortino': sortino,
        'asym': asym,
        'avg_xirr': avg_xirr,
        'ir_6m': ir_6m,
        'ir_3y': ir_3y,
        'vol_6m': vol_6m,
        'vol_3y': vol_3y,
        'dc_6m': dc_6m,
        'dc_3y': dc_3y,
        'appraisal': appraisal,
        'r2': r2
    }

def main(date=None):
    print("=" * 80)
    print("MID CAP GEMINI ALGO - Institutional Decision Tree (5y Daily)")
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
        chart = provider.get_mf_chart(mfId, duration='5y')
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
    
    results = []
    for mfId, data in funds_data.items():
        metrics = calculate_fund_metrics(data['nav'], bench_nav, current_date)
        if metrics is None:
            continue
            
        days = (data['nav'].index[-1] - data['nav'].index[0]).days
        aum = data['aum'] if pd.notna(data['aum']) else 0
        
        # Calculate Base Score from basic metrics
        # (This acts as the pre-decision-tree baseline, representing foundational fund health)
        res = {
            'mfId': mfId,
            'name': data['name'],
            'data_days': days,
            'aum': aum
        }
        res.update(metrics)
        
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
    
    # ---------------------------------------------------------
    # Decision Tree Scoring System
    # ---------------------------------------------------------
    
    # Baseline Score (0-100) based on Max DD, Sortino, Win Rate, Min XIRR
    df_res['max_dd_score'] = df_res['max_dd'].rank(ascending=False, pct=True) * 100
    df_res['sortino_score'] = df_res['sortino'].rank(ascending=True, pct=True) * 100
    df_res['win_rate_score'] = df_res['win_rate'].rank(ascending=True, pct=True) * 100
    df_res['min_xirr_score'] = df_res['min_xirr'].rank(ascending=True, pct=True) * 100
    
    df_res['base_score'] = (
        df_res['max_dd_score'] * 0.30 +
        df_res['sortino_score'] * 0.30 +
        df_res['win_rate_score'] * 0.20 +
        df_res['min_xirr_score'] * 0.20
    )
    
    final_scores = []
    for idx, row in df_res.iterrows():
        score = row['base_score']
        
        # --- STAGE 1: Hard Filters & Fundamental Penalties ---
        # Closet Indexer Penalty
        if row['r2'] > 0.95:
            score *= 0.50
        
        # AUM Bloat Penalty
        if row['aum'] > 10000:
            score *= 0.50
            
        # Debt-Hugger Penalty
        if row['avg_xirr'] < 0.12:
            score *= 0.50
            
        # Short History Penalty
        if row['data_days'] < 3 * 365:
            score *= 0.50
        elif row['data_days'] < 4 * 365:
            score *= 0.80
            
        # --- STAGE 2: Manager Edge Decay Penalties ---
        if row['ir_6m'] < row['ir_3y'] * 0.8:
            score *= 0.80
        if row['vol_6m'] > row['vol_3y'] * 1.2:
            score *= 0.80
        if row['dc_6m'] > row['dc_3y'] * 1.2:
            score *= 0.80
            
        # --- STAGE 3: Alpha & Asymmetry Boosts ---
        # We boost if they are in the top quartile (75th percentile).
        # We'll calculate quartiles globally first.
        final_scores.append(score)
        
    df_res['score'] = final_scores
    
    # Calculate quartiles for Stage 3
    appraisal_p75 = df_res['appraisal'].quantile(0.75)
    asym_p75 = df_res['asym'].quantile(0.75)
    
    # Apply boosts
    df_res['score'] = np.where(df_res['appraisal'] >= appraisal_p75, df_res['score'] * 1.25, df_res['score'])
    df_res['score'] = np.where(df_res['asym'] >= asym_p75, df_res['score'] * 1.25, df_res['score'])
    
    df_res['rank'] = df_res['score'].rank(ascending=False).astype(int)
    df_res = df_res.sort_values('rank')
    
    cols = ['mfId', 'name', 'rank', 'score', 'aum', 'cagr_3y', 'win_rate', 'max_dd', 'sortino', 'asym', 'appraisal', 'r2']
    final_df = df_res[cols].copy()
    
    # Format for readability
    final_df['score'] = final_df['score'].round(2)
    final_df['cagr_3y'] = final_df['cagr_3y'].round(2).astype(str) + '%'
    final_df['win_rate'] = (final_df['win_rate'] * 100).round(2).astype(str) + '%'
    final_df['max_dd'] = (final_df['max_dd'] * 100).round(2).astype(str) + '%'
    final_df['sortino'] = final_df['sortino'].round(3)
    final_df['asym'] = final_df['asym'].round(3)
    final_df['appraisal'] = final_df['appraisal'].round(3)
    final_df['r2'] = final_df['r2'].round(3)
    
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
