#!/usr/bin/env python3
"""
Generate and Print Statistics Report for Mutual Fund Data

A self-contained script that generates comprehensive statistics about
mutual funds and indices, then prints them to console.

Usage:
    python generate_statistics.py
Author: Claude Sonnet 4.5 Extended
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from mf_data_provider import MfDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_and_print_statistics():
    """Generate and print statistics report"""
    
    print("\n" + "="*70)
    print("GENERATING STATISTICS REPORT")
    print("="*70)
    
    # Initialize provider
    logger.info("Initializing MfDataProvider...")
    provider = MfDataProvider()
    
    print(f"\nData Directory: {provider.data_dir}")
    print("\nAnalyzing:")
    print("  - All mutual fund subsectors")
    print("  - Data availability (>1Y and >5Y)")
    print("  - Total AUM per subsector")
    print("  - Index performance (5Y rolling returns)")
    print("\n" + "="*70 + "\n")
    
    # Get all mutual funds
    logger.info("Fetching mutual fund list...")
    df_all = provider.list_all_mf()
    
    # ========== HEADER ==========
    print("\n" + "="*70)
    print("MUTUAL FUND DATA - STATISTICS & INTEGRATION METADATA")
    print("="*70)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Directory: {provider.data_dir}")
    
    # ========== SUBSECTOR STATISTICS ==========
    print("\n" + "="*70)
    print("MUTUAL FUND SUBSECTOR STATISTICS")
    print("="*70 + "\n")
    
    # Precompute common date thresholds (naive datetimes)
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    five_years_ago = now - timedelta(days=365 * 5)

    logger.info("Analyzing subsectors...")
    subsector_stats = []
    
    for sector in sorted(df_all['sector'].dropna().unique()):
        sector_df = df_all[df_all['sector'] == sector]
        
        for subsector in sorted(sector_df['subsector'].dropna().unique()):
            subsector_df = sector_df[sector_df['subsector'] == subsector]
            
            # Calculate total AUM
            total_aum = subsector_df['aum'].sum()
            
            # Check data availability for each fund
            funds_with_1y = 0
            funds_with_5y = 0

            for mf_id in subsector_df['mfId']:
                try:
                    chart = provider.get_mf_chart(mf_id)
                    
                    if len(chart) > 0:
                        # Normalize timestamps to naive datetime (UTC) for safe comparison
                        timestamps = pd.to_datetime(chart['timestamp'], utc=True).dt.tz_convert(None)
                        earliest_date = timestamps.min()
                        
                        if earliest_date <= one_year_ago:
                            funds_with_1y += 1
                        
                        if earliest_date <= five_years_ago:
                            funds_with_5y += 1
                
                except Exception as e:
                    logger.debug(f"Skipping {mf_id}: {str(e)}")
                    continue
            
            subsector_stats.append({
                'Sector': sector,
                'Subsector': subsector,
                'Total Funds': len(subsector_df),
                'Total AUM (Cr)': f"{total_aum:,.2f}",
                'Funds >1Y': funds_with_1y,
                'Funds >5Y': funds_with_5y
            })
    
    # Print subsector table
    if subsector_stats:
        df_subsector = pd.DataFrame(subsector_stats)
        print(df_subsector.to_string(index=False))
    else:
        print("No subsector data available")
    
    # ========== INDEX STATISTICS ==========
    print("\n" + "="*70)
    print("INDEX STATISTICS")
    print("="*70 + "\n")
    
    logger.info("Analyzing indices...")
    index_stats = []
    indices = provider.list_indices()
    
    for name, index_id in sorted(indices.items()):
        try:
            chart = provider.get_index_chart(index_id)
            
            if len(chart) > 0:
                # Normalize timestamps to naive datetime (UTC) for safe comparison
                timestamps = pd.to_datetime(chart['timestamp'], utc=True).dt.tz_convert(None)

                # Calculate 5-year rolling return
                five_year_return = None
                
                chart_5y = chart[timestamps >= five_years_ago]
                
                if len(chart_5y) > 0:
                    start_value = chart_5y.iloc[0]['nav']
                    end_value = chart_5y.iloc[-1]['nav']
                    
                    ts_5y = timestamps[chart_5y.index]
                    years = (ts_5y.iloc[-1] - ts_5y.iloc[0]).days / 365.25
                    
                    if years > 0 and start_value > 0:
                        five_year_return = ((end_value / start_value) ** (1 / years) - 1) * 100
                
                # Get data range
                earliest = timestamps.min()
                latest = timestamps.max()
                data_years = (latest - earliest).days / 365.25
                
                index_stats.append({
                    'Index': name,
                    'ID': index_id,
                    'Data Points': len(chart),
                    'Years': f"{data_years:.1f}",
                    '5Y Return (%)': f"{five_year_return:.2f}" if five_year_return else "N/A",
                    'Latest': f"{chart.iloc[-1]['nav']:.2f}",
                    'Date': timestamps.iloc[-1].strftime('%Y-%m-%d')
                })
            else:
                index_stats.append({
                    'Index': name,
                    'ID': index_id,
                    'Data Points': 0,
                    'Years': "N/A",
                    '5Y Return (%)': "N/A",
                    'Latest': "N/A",
                    'Date': "N/A"
                })
        
        except Exception as e:
            logger.warning(f"Failed to get stats for {name}: {str(e)}")
            index_stats.append({
                'Index': name,
                'ID': index_id,
                'Data Points': "Error",
                'Years': "Error",
                '5Y Return (%)': "Error",
                'Latest': "Error",
                'Date': "Error"
            })
    
    # Print index table
    if index_stats:
        df_index = pd.DataFrame(index_stats)
        print(df_index.to_string(index=False))
    else:
        print("No index data available")
    
    # ========== SUMMARY STATISTICS ==========
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    total_funds_with_1y = sum(stat['Funds >1Y'] for stat in subsector_stats)
    total_funds_with_5y = sum(stat['Funds >5Y'] for stat in subsector_stats)
    
    print(f"Total Mutual Funds:          {len(df_all):,}")
    print(f"Total Sectors:               {df_all['sector'].nunique()}")
    print(f"Total Subsectors:            {df_all['subsector'].nunique()}")
    print(f"Total AUM:                   ₹{df_all['aum'].sum():,.2f} Crores")
    print(f"Total Indices:               {len(indices)}")
    print(f"Funds with >1Y Data:         {total_funds_with_1y:,}")
    print(f"Funds with >5Y Data:         {total_funds_with_5y:,}")
    
    # ========== DATA QUALITY ==========
    print("\n" + "="*70)
    print("DATA QUALITY METRICS")
    print("="*70 + "\n")
    
    pct_1y = (total_funds_with_1y / len(df_all) * 100) if len(df_all) > 0 else 0
    pct_5y = (total_funds_with_5y / len(df_all) * 100) if len(df_all) > 0 else 0
    
    print(f"Data Completeness:           {pct_1y:.1f}% of funds have >1Y data")
    print(f"Long-term Data:              {pct_5y:.1f}% of funds have >5Y data")
    
    # ========== TOP SUBSECTORS ==========
    print("\n" + "="*70)
    print("TOP 10 SUBSECTORS BY AUM")
    print("="*70 + "\n")
    
    df_subsector['AUM_numeric'] = df_subsector['Total AUM (Cr)'].str.replace(',', '').astype(float)
    top_subsectors = df_subsector.nlargest(10, 'AUM_numeric')
    
    for _, row in top_subsectors.iterrows():
        print(f"{row['Subsector']:40s} ₹{row['Total AUM (Cr)']:>15s} Cr ({row['Total Funds']:>3d} funds)")
    
    # ========== TOP FUNDS ==========
    print("\n" + "="*70)
    print("TOP 10 FUNDS BY AUM")
    print("="*70 + "\n")
    
    top_funds = df_all.nlargest(10, 'aum')
    for _, fund in top_funds.iterrows():
        print(f"{fund['name']:50s} ₹{fund['aum']:>12,.2f} Cr")
    
    print("\n" + "="*70)
    print("REPORT COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        generate_and_print_statistics()
    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation interrupted by user")
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        raise