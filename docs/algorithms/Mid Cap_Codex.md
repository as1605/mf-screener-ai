# Mid Cap Strategy - Codex

## Overview

**Author**: Codex
**Focus**: Forward 1-year excess return prediction.
**Target**: Indian Mid Cap funds.

This model is purpose-built for Indian Mid Cap funds and tuned specifically for forward 1-year excess return prediction against the Nifty Midcap 150 benchmark. It uses a walk-forward validation approach to tune weights and applies a market-regime tilt to the final live ranking.

## Key Metrics & Weights

The model uses a dynamic weighting system that is tuned based on historical performance, constrained within these ranges:

| Category | Weight Range | Key Metrics |
| :--- | :--- | :--- |
| **Skill & Active Return** | **22% - 55%** | Alpha (1Y), Information Ratio, Sortino Ratio, Rolling Alpha Stability, Rolling Active IR |
| **Market Regime** | **18% - 40%** | Up/Down Capture, Capture Spread, Bear Market Excess, High Volatility Excess |
| **Downside Pain** | **16% - 38%** | Max Drawdown, Ulcer Index, Recovery Speed, Rolling Beat Rate |
| **Momentum** | **8% - 26%** | Relative Momentum (6M, 12-1M), Overheat Penalty |
| **Anchors** | **7% - 22%** | CAGR (3Y, 5Y) |

## Methodology

1.  **Panel Construction**: Builds a historical cross-sectional factor panel at many past dates.
2.  **Backtesting**: Backtests each date against the next 1-year realized excess return.
3.  **Tuning**: Tunes weights using walk-forward cross-validation to maximize predictive power.
4.  **Regime Tilt**: Applies a small conservative market-regime tilt (Bull/Correction/High Volatility) to the final live ranking.

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **Invesco India Midcap Fund** (Score: 81.92)
2.  **ICICI Pru Midcap Fund** (Score: 81.34)
3.  **HDFC Mid Cap Fund** (Score: 79.44)
4.  **Nippon India Growth Mid Cap Fund** (Score: 77.58)
5.  **Mirae Asset Midcap Fund** (Score: 72.61)

[View Results CSV](../../results/Mid%20Cap_Codex.csv)
