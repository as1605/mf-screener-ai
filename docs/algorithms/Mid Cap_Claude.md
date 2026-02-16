# Mid Cap Strategy - Claude

## Overview

**Author**: Claude
**Focus**: Multi-factor quantitative scoring with regime analysis and consistency metrics.
**Target**: Superior risk-adjusted returns over the next 1 year.

This model employs an advanced multi-factor quantitative scoring system designed to predict superior risk-adjusted returns. It computes approximately 25 metrics per fund across six research-backed categories, normalizes each to a peer-group percentile, and combines them using fixed weights derived from academic research on mutual fund performance persistence.

## Key Metrics & Weights

The scoring system is divided into the following categories:

| Category | Weight | Key Metrics |
| :--- | :--- | :--- |
| **Risk-Adjusted Returns** | **35%** | Sortino Ratio, Alpha, Treynor-Mazuy Alpha, Information Ratio, Omega Ratio, Treynor Ratio |
| **Downside Protection** | **20%** | Max Drawdown, Downside Capture, Ulcer Performance Index, Calmar Ratio, Recovery Speed |
| **Consistency** | **20%** | Rolling 1Y Beat %, Alpha Stability, Sharpe Stability |
| **Market Regime** | **10%** | Capture Spread, Beta Asymmetry, Bear Market Alpha |
| **Momentum** | **8%** | 12-1 Month Momentum, 6-Month Relative Momentum |
| **Return Magnitude** | **7%** | Blended CAGR (1Y/3Y/5Y) |

## Key Innovations

*   **Treynor-Mazuy Decomposition**: Separates stock-picking skill from market timing ability.
*   **Dual-Beta Model**: Captures asymmetric participation in up vs. down markets.
*   **Rolling Consistency**: Detects repeatable edges versus lucky streaks using rolling window analysis.
*   **Market Regime Conditioning**: Evaluates behavior specifically in bull vs. bear periods.
*   **Recovery Analysis**: Rewards funds that bounce back quickly from drawdowns.

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **HDFC Mid Cap Fund** (Score: 81.38)
2.  **Invesco India Midcap Fund** (Score: 78.79)
3.  **WOC Mid Cap Fund** (Score: 78.05)
4.  **Edelweiss Mid Cap Fund** (Score: 77.44)
5.  **HSBC Midcap Fund** (Score: 68.32)

[View Results CSV](../../results/Mid%20Cap_Claude.csv)
