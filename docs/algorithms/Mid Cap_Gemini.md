# Mid Cap Strategy - Gemini

## Overview

**Author**: Gemini
**Focus**: Consistency and Fundamental Risk Metrics.
**Target**: Balanced performance evaluation.

This strategy uses a straightforward, fundamental approach to scoring mutual funds. It prioritizes consistency of performance (win rate) and standard risk-adjusted return metrics, ensuring a balanced evaluation that rewards both returns and risk management.

## Key Metrics & Weights

The scoring model is a weighted average of normalized scores (0-100) for the following metrics:

| Metric | Weight | Description |
| :--- | :--- | :--- |
| **Win Rate (Consistency)** | **20%** | Percentage of rolling 1-year periods where the fund beat the benchmark. |
| **Sharpe Ratio** | **20%** | Measure of risk-adjusted return using standard deviation. |
| **Sortino Ratio** | **20%** | Measure of risk-adjusted return using downside deviation. |
| **Alpha** | **20%** | Jensen's Alpha, representing excess return over the risk-free rate adjusted for beta. |
| **Downside Capture** | **10%** | Ratio of fund returns to benchmark returns during down markets (Lower is better). |
| **Max Drawdown** | **10%** | Maximum observed loss from a peak to a trough (Closer to 0 is better). |

## Analysis Window
*   **Timeframe**: Last 5 years (standardized to ensure fair comparison).
*   **Minimum Data**: Requires at least ~1 year (250 trading days) of overlapping data with the benchmark.

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **HDFC Mid Cap Fund** (Score: 85.96)
2.  **Motilal Oswal Midcap Fund** (Score: 76.94)
3.  **Nippon India Growth Mid Cap Fund** (Score: 73.66)
4.  **Edelweiss Mid Cap Fund** (Score: 72.65)
5.  **Invesco India Midcap Fund** (Score: 69.86)

[View Results CSV](../../results/Mid%20Cap_Gemini.csv)
