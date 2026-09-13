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

SECTOR = "Multi Asset"
SUBSECTOR = "Multi Asset Allocation Fund"
EQUITY_INDEX = "Total Market"
GOLD_MF_ID = "M_SBIGL"

def daily_returns(nav_series: pd.Series) -> pd.Series:
    return nav_series.pct_change().dropna()

def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0: return 0.0
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    return drawdown.min()

def calculate_24m_scenario_sortino(fund_nav: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    fund_monthly = fund_nav.resample('MS').first()
    months_total = 24
    if len(fund_monthly) < months_total + 1:
        return None, None
    returns = []
    for i in range(len(fund_monthly) - months_total):
        f_invest = fund_monthly.iloc[i : i+12]
        f_final = fund_monthly.iloc[i+months_total]
        f_units = 1000 / f_invest
        f_val = f_units.sum() * f_final
        f_ret = (f_val - (1000 * 12)) / (1000 * 12)
        returns.append(f_ret)
    if not returns: return None, None
    returns = np.array(returns)
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        sortino = mean_ret / (returns.std() + 1e-6)
    else:
        downside_std = np.sqrt(np.mean(downside**2))
        sortino = mean_ret / downside_std
    return mean_ret, sortino

def compute_up_beta(fund_ret: pd.Series, bench_ret: pd.Series) -> Optional[float]:
    aligned = pd.concat([fund_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 20: return None
    f_r, b_r = aligned.iloc[:, 0], aligned.iloc[:, 1]
    up_days = b_r > 0
    if up_days.sum() > 5:
        cov = np.cov(b_r[up_days], f_r[up_days])[0, 1]
        var = np.var(b_r[up_days], ddof=1)
        if var > 0: return cov / var
    return None

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100

def main(date: Optional[str] = None):
    provider = MfDataProvider(date=date)
    indices = provider.list_indices()
    equity_idx_id = indices.get(EQUITY_INDEX)
    equity_df = provider.get_index_chart(equity_idx_id)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
    equity_nav = equity_df.sort_values("timestamp").set_index("timestamp")["nav"]
    
    gold_df = provider.get_mf_chart(GOLD_MF_ID, duration='5y')
    gold_df["timestamp"] = pd.to_datetime(gold_df["timestamp"], utc=True)
    gold_nav = gold_df.sort_values("timestamp").set_index("timestamp")["nav"]
    
    df_all = provider.list_all_mf()
    maa_df = df_all[df_all["subsector"] == SUBSECTOR].copy()
    
    results = []
    for _, row in maa_df.iterrows():
        try:
            chart = provider.get_mf_chart(row["mfId"], duration='5y')
            if len(chart) < 252: continue
            chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True)
            fund_nav = chart.sort_values("timestamp").set_index("timestamp")["nav"]
            
            fund_rets = daily_returns(fund_nav)
            mean_24m, sortino_24m = calculate_24m_scenario_sortino(fund_nav)
            mdd = max_drawdown(fund_rets)
            gold_up = compute_up_beta(fund_rets, daily_returns(gold_nav))
            vol = fund_rets.std() * np.sqrt(252)
            
            results.append({
                "mfId": row["mfId"],
                "name": row["name"],
                "aum": row.get("aum", 0) or 0,
                "data_days": len(fund_nav),
                "mean_24m": mean_24m,
                "sortino_24m": sortino_24m,
                "max_drawdown": mdd,
                "gold_up_beta": gold_up,
                "volatility": vol
            })
        except Exception as e:
            continue
            
    df = pd.DataFrame(results)
    
    # Compute Score
    score_components = {
        "sortino_24m":  (True,  0.35),
        "max_drawdown": (True,  0.30), # True because mdd is negative, higher (closer to 0) is better
        "mean_24m":     (True,  0.15),
        "gold_up_beta": (True,  0.10),
        "volatility":   (False, 0.10), # False because lower is better
    }
    
    df["raw_score"] = 0.0
    applied_weight = pd.Series(0.0, index=df.index)
    
    for col, (higher_better, weight) in score_components.items():
        pctl = percentile_rank(df[col], higher_is_better=higher_better)
        pctl = pctl.fillna(40.0)
        df["raw_score"] += pctl * weight
        applied_weight += weight
        
    df["score"] = df["raw_score"] / applied_weight
    
    # Penalty for short history
    penalty = np.clip((df["data_days"] / 756) * 0.3 + 0.7, 0.5, 1.0)
    df["score"] = df["score"] * penalty
    df["score"] = df["score"].round(2)
    
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank")
    
    df.to_csv(ROOT_DIR / "results/experiments/exp3_results.csv", index=False)
    print(df[["rank", "score", "name", "mean_24m", "max_drawdown"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
