
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mf_data_provider import MfDataProvider

def calculate_metrics(fund_df, benchmark_df, risk_free_rate=0.065):
    """
    Calculate various performance metrics for a fund against a benchmark.
    """
    # Merge fund and benchmark data on timestamp
    merged = pd.merge(fund_df, benchmark_df, on='timestamp', how='inner', suffixes=('_fund', '_bench'))
    
    if len(merged) < 250:  # Require at least ~1 year of overlapping data
        return None

    # Standardize analysis window to last 5 years (fair comparison)
    # This prevents funds that started post-crash (e.g. post-2020) from having an unfair advantage
    # over funds that carry the baggage of historical crashes in their full-history metrics.
    max_date = merged['timestamp'].max()
    start_date = max_date - timedelta(days=5*365)
    merged = merged[merged['timestamp'] > start_date]

    # Calculate daily returns
    merged['return_fund'] = merged['nav_fund'].pct_change()
    merged['return_bench'] = merged['nav_bench'].pct_change()
    merged = merged.dropna()

    if len(merged) < 250:
        return None

    # 1. Rolling Returns (3-year rolling CAGR)
    # We'll take the mean of rolling 3Y CAGRs to measure consistency
    window_days = 3 * 365
    if len(merged) > window_days:
        # Calculate rolling 3Y return
        # (Price_t / Price_t-n)^(1/3) - 1
        # We can approximate using rolling window on daily returns? No, better to use NAVs.
        # But we need to align dates. Let's stick to a simpler rolling return metric:
        # Average 1-year rolling return
        rolling_window = 252 # Trading days
        merged['rolling_1y_fund'] = merged['nav_fund'].pct_change(periods=rolling_window)
        merged['rolling_1y_bench'] = merged['nav_bench'].pct_change(periods=rolling_window)
        
        # Win rate: % of time fund beat benchmark on 1y rolling basis
        rolling_data = merged.dropna(subset=['rolling_1y_fund', 'rolling_1y_bench'])
        if len(rolling_data) > 0:
            win_rate = np.mean(rolling_data['rolling_1y_fund'] > rolling_data['rolling_1y_bench'])
            avg_rolling_return = rolling_data['rolling_1y_fund'].mean()
        else:
            win_rate = 0.5
            avg_rolling_return = merged['return_fund'].mean() * 252
    else:
        win_rate = 0.5 # Neutral if not enough data
        avg_rolling_return = merged['return_fund'].mean() * 252

    # 2. Max Drawdown
    cumulative_returns = (1 + merged['return_fund']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # 3. Sharpe Ratio
    excess_returns = merged['return_fund'] - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # 4. Sortino Ratio
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
    else:
        sortino_ratio = sharpe_ratio # Fallback

    # 5. Beta and Alpha
    covariance = np.cov(merged['return_fund'], merged['return_bench'])[0][1]
    benchmark_variance = merged['return_bench'].var()
    beta = covariance / benchmark_variance
    
    # Alpha (Jensen's Alpha) = Rp - [Rf + Beta * (Rm - Rf)]
    # Annualized
    rp = merged['return_fund'].mean() * 252
    rm = merged['return_bench'].mean() * 252
    alpha = rp - (risk_free_rate + beta * (rm - risk_free_rate))

    # 6. Downside Capture Ratio
    # Down Market: Benchmark return < 0
    down_market = merged[merged['return_bench'] < 0]
    if len(down_market) > 0:
        down_capture = (down_market['return_fund'].mean() / down_market['return_bench'].mean())
    else:
        down_capture = 1.0

    # 3Y and 5Y CAGR (for reporting)
    start_nav = fund_df['nav'].iloc[0]
    end_nav = fund_df['nav'].iloc[-1]
    days = (fund_df['timestamp'].iloc[-1] - fund_df['timestamp'].iloc[0]).days
    
    cagr_3y = 0
    if days > 365 * 3:
        three_y_ago = fund_df['timestamp'].iloc[-1] - timedelta(days=365*3)
        idx = fund_df['timestamp'].searchsorted(three_y_ago)
        if idx < len(fund_df):
            nav_3y = fund_df['nav'].iloc[idx]
            cagr_3y = ((end_nav / nav_3y) ** (1/3) - 1) * 100
    elif days > 0:
        cagr_3y = ((end_nav / start_nav) ** (365/days) - 1) * 100

    cagr_5y = 0
    if days > 365 * 5:
        # Get exact 5y ago date
        five_y_ago = fund_df['timestamp'].iloc[-1] - timedelta(days=365*5)
        # Find closest date
        idx = fund_df['timestamp'].searchsorted(five_y_ago)
        if idx < len(fund_df):
             nav_5y = fund_df['nav'].iloc[idx]
             cagr_5y = ((end_nav / nav_5y) ** (1/5) - 1) * 100
    elif days > 0:
        cagr_5y = ((end_nav / start_nav) ** (365/days) - 1) * 100

    return {
        'win_rate': win_rate,
        'avg_rolling_return': avg_rolling_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta,
        'down_capture': down_capture,
        'cagr_3y': cagr_3y,
        'cagr_5y': cagr_5y,
        'data_days': days
    }

def normalize_series(series, lower_is_better=False):
    """Normalize a pandas series to 0-100 scale."""
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isnull().all():
        return series
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(50, index=series.index)
        
    if lower_is_better:
        return 100 * (max_val - series) / (max_val - min_val)
    else:
        return 100 * (series - min_val) / (max_val - min_val)

def main():
    SECTOR = "Mid Cap"
    MODEL = "Gemini"
    
    print(f"Starting analysis for {SECTOR} using {MODEL} strategy...")
    
    provider = MfDataProvider()
    
    # Get all funds in sector
    sectors = provider.list_mf_by_sector()
    # Find the correct key for Mid Cap. It might be under Equity -> Mid Cap Fund
    # Let's search for it
    target_funds = []
    
    # "Mid Cap" in the task likely refers to "Equity" -> "Mid Cap Fund"
    if 'Equity' in sectors and 'Mid Cap Fund' in sectors['Equity']:
        target_funds = sectors['Equity']['Mid Cap Fund']
    else:
        print("Could not find 'Mid Cap Fund' in Equity sector.")
        return

    print(f"Found {len(target_funds)} funds in {SECTOR}...")

    # Get Benchmark Data
    indices = provider.list_indices()
    # Task says Mid Cap index is .NIMI150
    benchmark_id = indices.get('Mid Cap', '.NIMI150')
    print(f"Fetching benchmark data for {benchmark_id}...")
    benchmark_df = provider.get_index_chart(benchmark_id)
    benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
    benchmark_df['nav'] = pd.to_numeric(benchmark_df['nav'], errors='coerce')
    
    results = []
    
    for mf_id in target_funds:
        try:
            fund_info = provider.list_all_mf()
            fund_name = fund_info[fund_info['mfId'] == mf_id]['name'].values[0]
            
            print(f"Processing {fund_name} ({mf_id})...")
            fund_df = provider.get_mf_chart(mf_id)
            fund_df['timestamp'] = pd.to_datetime(fund_df['timestamp'])
            fund_df['nav'] = pd.to_numeric(fund_df['nav'], errors='coerce')
            
            if len(fund_df) < 250:
                print(f"Skipping {fund_name}: Insufficient data (< 1 year)")
                continue
                
            metrics = calculate_metrics(fund_df, benchmark_df)
            
            if metrics:
                metrics['mfId'] = mf_id
                metrics['name'] = fund_name
                results.append(metrics)
                
        except Exception as e:
            print(f"Error processing {mf_id}: {str(e)}")
            continue

    if not results:
        print("No results generated.")
        return

    df_results = pd.DataFrame(results)
    
    # Calculate Score
    # Weights
    # Win Rate (Consistency): 20%
    # Sharpe: 20%
    # Sortino: 20%
    # Alpha: 20%
    # Downside Capture (Lower is better): 10%
    # Max Drawdown (Higher (closer to 0) is better): 10%
    
    # Normalize metrics
    metrics_cols = ['win_rate', 'sharpe_ratio', 'sortino_ratio', 'alpha', 'down_capture', 'max_drawdown']
    
    # Fill NaNs with median
    for col in metrics_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].fillna(df_results[col].median())
            
    df_results['score_consistency'] = normalize_series(df_results['win_rate'])
    df_results['score_sharpe'] = normalize_series(df_results['sharpe_ratio'])
    df_results['score_sortino'] = normalize_series(df_results['sortino_ratio'])
    df_results['score_alpha'] = normalize_series(df_results['alpha'])
    df_results['score_down_capture'] = normalize_series(df_results['down_capture'], lower_is_better=True)
    df_results['score_drawdown'] = normalize_series(df_results['max_drawdown']) # max_drawdown is negative, so closer to 0 (max) is better. Normalizing standard way works (max is best).
    
    df_results['score'] = (
        0.20 * df_results['score_consistency'] +
        0.20 * df_results['score_sharpe'] +
        0.20 * df_results['score_sortino'] +
        0.20 * df_results['score_alpha'] +
        0.10 * df_results['score_down_capture'] +
        0.10 * df_results['score_drawdown']
    )
    
    # Drop rows with NaN score
    df_results = df_results.dropna(subset=['score'])
    
    # Rank
    df_results['rank'] = df_results['score'].rank(ascending=False).astype(int)
    
    # Sort by rank
    df_results = df_results.sort_values('rank')
    
    # Select columns for output
    output_columns = [
        'mfId', 'name', 'rank', 'score', 'data_days', 'cagr_3y', 'cagr_5y', 
        'alpha', 'beta', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'down_capture', 'win_rate'
    ]
    
    final_df = df_results[output_columns]
    
    # Save to CSV
    output_path = f"results/{SECTOR}_{MODEL}.csv"
    os.makedirs('results', exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
