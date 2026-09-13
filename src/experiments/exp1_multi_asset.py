#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mf_data_provider import MfDataProvider

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

SECTOR = "Multi Asset"
SUBSECTOR = "Multi Asset Allocation Fund"
EQUITY_INDEX = "Total Market"           # .NIFTY500
GOLD_MF_ID = "M_SBIGL"
RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252

def daily_returns(nav_series: pd.Series) -> pd.Series:
    return nav_series.pct_change().dropna()

def max_drawdown(returns: pd.Series) -> float:
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    return drawdown.min()

def calculate_24m_scenario(fund_nav: pd.Series) -> pd.Series:
    """Simulate 12m SIP + 12m Hold (24 months total)"""
    fund_monthly = fund_nav.resample('MS').first()
    months_total = 24
    if len(fund_monthly) < months_total + 1:
        return pd.Series(dtype=float)
        
    returns = {}
    for i in range(len(fund_monthly) - months_total):
        f_invest = fund_monthly.iloc[i : i+12] # 12 months SIP
        f_final = fund_monthly.iloc[i+months_total] # end of 24 months
        
        f_units = 1000 / f_invest
        f_val = f_units.sum() * f_final
        f_ret = (f_val - (1000 * 12)) / (1000 * 12)
        returns[fund_monthly.index[i+months_total]] = f_ret
    return pd.Series(returns)

def analyse_fund(mf_id: str, fund_nav: pd.Series, equity_nav: pd.Series, name: str, aum: float) -> dict:
    n = len(fund_nav)
    result = {"mfId": mf_id, "name": name, "aum": round(aum, 2), "data_days": n}
    
    fund_rets = daily_returns(fund_nav)
    equity_rets = daily_returns(equity_nav)
    
    # 24m Scenario
    scenario_rets = calculate_24m_scenario(fund_nav)
    if len(scenario_rets) > 0:
        result["scenario_mean"] = scenario_rets.mean()
        result["scenario_min"] = scenario_rets.min()
    else:
        result["scenario_mean"] = None
        result["scenario_min"] = None
        
    # Max Drawdown
    result["max_drawdown"] = max_drawdown(fund_rets)
    
    # Equity Beta (to see if they are just hugging equity)
    aligned = pd.concat([fund_rets, equity_rets], axis=1, join="inner").dropna()
    if len(aligned) > 50:
        f_r = aligned.iloc[:, 0]
        b_r = aligned.iloc[:, 1]
        cov = np.cov(b_r, f_r)[0, 1]
        var = np.var(b_r, ddof=1)
        result["equity_beta"] = cov / var if var > 0 else None
    else:
        result["equity_beta"] = None

    return result

def main(date: Optional[str] = None):
    provider = MfDataProvider(date=date)
    indices = provider.list_indices()
    equity_idx_id = indices.get(EQUITY_INDEX)
    equity_df = provider.get_index_chart(equity_idx_id)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
    equity_nav = equity_df.sort_values("timestamp").set_index("timestamp")["nav"]
    
    df_all = provider.list_all_mf()
    maa_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    
    results = []
    for _, row in maa_df.iterrows():
        try:
            chart = provider.get_mf_chart(row["mfId"], duration='5y')
            if len(chart) < 252:
                continue
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            fund_nav = chart.sort_values("timestamp").set_index("timestamp")["nav"]
            res = analyse_fund(row["mfId"], fund_nav, equity_nav, row["name"], row.get("aum", 0))
            results.append(res)
        except Exception as e:
            continue
            
    df = pd.DataFrame(results)
    df.to_csv(ROOT_DIR / "results/experiments/exp1_results.csv", index=False)
    print("Done exp1.")

if __name__ == "__main__":
    main()
