# Total Market Strategy - Claude

## Overview

**Author**: Claude
**Focus**: Adaptive multi-horizon conviction model with path quality and tail risk analysis.
**Target**: Superior risk-adjusted returns over the next 1 year.
**Subsectors**: Contra Fund, Flexi Cap Fund, Focused Fund, Multi Cap Fund, Value Fund

This model is purpose-built for India's Total Market funds, which enjoy maximum portfolio flexibility across market caps, sectors, and styles. The key challenge is identifying fund managers who consistently convert that freedom into alpha. The model computes ~28 metrics per fund across six research-backed categories, normalises each to a peer-group percentile, and combines them via fixed weights.

## Key Metrics & Weights

| Category | Weight | Key Metrics |
| :--- | :--- | :--- |
| **Skill & Alpha Quality** | **28%** | Jensen's Alpha, Information Ratio, Treynor-Mazuy Alpha, Excess Return Autocorrelation, Active Divergence Score |
| **Path Quality & Tail Risk** | **18%** | Gain-to-Pain Ratio, Tail Ratio, CVaR, Ulcer Performance Index, Calmar Ratio |
| **Drawdown Resilience** | **16%** | Max Drawdown, Pain Index, Recovery Speed, Avg Drawdown Duration |
| **Regime Adaptability** | **14%** | Capture Spread, Regime Transition Alpha, Beta Asymmetry, Bear Outperformance |
| **Consistency & Stability** | **14%** | Rolling 1Y Beat %, Cross-Horizon Rank Consistency, Sortino Stability, Hit Rate |
| **Momentum & Acceleration** | **10%** | 6M Relative Momentum, Volatility-Normalised Momentum, Momentum Acceleration |

## Key Innovations

*   **Excess Return Autocorrelation**: Measures persistence of alpha generation. Positive autocorrelation predicts continued outperformance (Carhart 1997, Bollen & Busse 2005).
*   **Gain-to-Pain Ratio** (Schwager): Captures asymmetry of the full return stream, not just tail events.
*   **Tail Ratio**: Compares right-tail upside to left-tail downside, rewarding funds with positive skew.
*   **CVaR (Expected Shortfall)**: More robust than VaR for capturing extreme loss risk.
*   **Regime Transition Alpha**: Measures skill during market regime changes (bull-to-bear, bear-to-bull), a strong predictor of adaptive management.
*   **Cross-Horizon Rank Consistency**: Novel metric checking if a fund ranks well across 3M/6M/1Y/3Y horizons simultaneously, filtering out lucky streaks.
*   **Volatility-Normalised Momentum**: Penalises momentum earned from high-volatility bets (less persistent) and rewards low-risk alpha.
*   **Momentum Acceleration**: Forward-looking signal capturing whether relative strength is increasing or fading.
*   **Active Divergence Score**: Combines tracking error with alpha direction to reward skilled active management.

## Analysis Window

- **Benchmark**: Nifty 500 (Total Market)
- **Minimum Data**: 50 weeks for inclusion; confidence penalties for < 5Y history
- **Walk-Forward Validation**: 3Y training, 1Y forward test, 6M step

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **Mahindra Manulife Focused Fund** (Score: 81.84)
    - CAGR 3Y: 21.49%, CAGR 5Y: 20.11%
    - Alpha: 6.63%, Information Ratio: 1.434
    - Gain-to-Pain: 0.74, CVaR: -204.36, Max Drawdown: -16.47%
    - Cross-Horizon Consistency: 0.42, Rolling Beat %: 93.24%

2.  **LIC MF Multi Cap Fund** (Score: 74.74)
    - CAGR 3Y: 22.34%
    - Alpha: 5.67%, Information Ratio: 0.971
    - Gain-to-Pain: 0.62, CVaR: -228.47, Max Drawdown: -18.94%
    - Cross-Horizon Consistency: 0.39, Rolling Beat %: 91.67%

3.  **DSP Value Fund** (Score: 74.35)
    - CAGR 3Y: 21.12%, CAGR 5Y: 17.42%
    - Alpha: 6.05%, Information Ratio: 0.291
    - Gain-to-Pain: 0.76, CVaR: -160.02, Max Drawdown: -15.64%
    - Cross-Horizon Consistency: 0.58, Rolling Beat %: 72.60%

4.  **Axis Value Fund** (Score: 71.19)
    - CAGR 3Y: 24.45%
    - Alpha: 6.73%, Information Ratio: 1.460
    - Gain-to-Pain: 0.52, CVaR: -219.87, Max Drawdown: -19.44%
    - Cross-Horizon Consistency: 0.54, Rolling Beat %: 93.26%

5.  **Edelweiss Focused Fund** (Score: 71.08)
    - CAGR 3Y: 19.54%
    - Alpha: 4.01%, Information Ratio: 0.917
    - Gain-to-Pain: 0.60, CVaR: -198.58, Max Drawdown: -16.71%
    - Cross-Horizon Consistency: 0.34, Rolling Beat %: 91.73%

[View Results CSV](../../results/Total%20Market_Claude.csv)
