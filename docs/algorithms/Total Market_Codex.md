# Total Market Strategy - Codex

## Overview

**Author**: Codex
**Focus**: Walk-forward validation and weight tuning to optimize for forward returns.
**Target**: Identify funds with consistent risk-adjusted returns validated through historical backtesting.
**Subsectors**: Contra Fund, Flexi Cap Fund, Focused Fund, Multi Cap Fund, Value Fund

This model uses a sophisticated walk-forward backtesting approach to tune factor weights based on historical performance. Unlike fixed-weight models, Codex's approach validates which metrics actually predict forward returns by testing them across multiple time periods. The model also implements subsector-aware rank blending to reduce style overconcentration.

## Key Metrics & Factor Categories

The model evaluates funds across **22 factors** organized into six categories:

| Category | Factors | Focus |
| :--- | :--- | :--- |
| **Manager Skill** | Alpha (1Y), Information Ratio (1Y), Sortino (1Y), Rolling Alpha Stability (2Y), Rolling Active IR (2Y) | Active edge and skill persistence |
| **Market-Cycle Handling** | Up/Down Capture (1Y), Capture Spread (1Y), Bear Excess (1Y), High Vol Excess (1Y) | Performance across different market conditions |
| **Drawdown Resilience** | Max Drawdown (2Y), Ulcer Index (2Y), Recovery Speed (2Y), Rolling Beat Rate (2Y) | Downside protection and recovery ability |
| **Multi-Horizon Structure** | Cross-Horizon Consistency, Beta Distance (1Y) | Consistency across timeframes |
| **Momentum** | Momentum (6M relative), Momentum (12M-1M relative), Overheat Penalty | Recent performance trends with overheat detection |
| **Compounding Anchors** | CAGR (3Y), CAGR (5Y) | Long-term return validation |

## Key Innovations

*   **Walk-Forward Validation**: Uses rolling training/validation windows to tune factor weights. Tests 220+ weight combinations across multiple time periods to identify which metrics actually predict forward returns.
*   **Subsector-Aware Blending**: Blends 22% of within-subsector ranking into the total rank to prevent overconcentration in a single style (e.g., all Focused Funds).
*   **Market Regime Tilt**: Detects current market regime (bull/bear/mixed) and applies mild tilts to weights:
    - **Aggressive tilt** (bull markets): Emphasizes alpha, up-capture, momentum
    - **Defensive tilt** (bear markets): Emphasizes down-capture, bear excess, drawdown resilience
    - **Anchor tilt**: Always maintains weight on long-term CAGR anchors
*   **Factor Weight Caps**: Limits any single factor to maximum 22% weight to ensure diversification of signals.
*   **History Reliability**: Adjusts scores based on data history length and factor coverage, ensuring funds with limited data are appropriately penalized.
*   **Overheat Penalty**: Detects when momentum is too strong (potential mean reversion risk) and penalizes accordingly.

## Analysis Window

- **Benchmark**: Nifty 500 (Total Market)
- **Minimum Data**: 50 weeks for inclusion
- **Walk-Forward Setup**: 
  - Training: ~60% of historical data
  - Validation: ~20% of historical data  
  - Test: ~20% of historical data
  - Cross-validation: Multiple rolling folds
- **Forward Horizon**: 52 weeks (1 year)
- **Evaluation Step**: 8 weeks
- **Top K Selection**: Top 7 funds evaluated for backtest performance

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **ICICI Pru Focused Equity Fund** (Score: 89.6)
    - Alpha (1Y): 8.39%, Information Ratio: 2.079
    - CAGR 3Y: 24.69%, CAGR 5Y: 20.62%
    - Rolling Beat Rate: 100.00%, Cross-Horizon Consistency: 0.916
    - Recovery Speed: 0.514, Max Drawdown: -16.76%

2.  **HDFC Focused Fund** (Score: 82.71)
    - Alpha (1Y): 5.40%, Information Ratio: 0.677
    - CAGR 3Y: 22.97%, CAGR 5Y: 23.08%
    - Rolling Beat Rate: 100.00%, Cross-Horizon Consistency: 0.868
    - Recovery Speed: 0.517, Max Drawdown: -11.13%

3.  **HDFC Flexi Cap Fund** (Score: 82.11)
    - Alpha (1Y): 5.19%, Information Ratio: 0.594
    - CAGR 3Y: 22.67%, CAGR 5Y: 21.14%
    - Rolling Beat Rate: 100.00%, Cross-Horizon Consistency: 0.877
    - Recovery Speed: 0.484, Max Drawdown: -11.92%

4.  **Kotak Focused Fund** (Score: 81.23)
    - Alpha (1Y): 9.87%, Information Ratio: 2.004
    - CAGR 3Y: 19.72%, CAGR 5Y: 16.45%
    - Rolling Beat Rate: 92.41%, Cross-Horizon Consistency: 0.841
    - Recovery Speed: 0.383, Max Drawdown: -17.65%

5.  **Groww Multicap Fund** (Score: 79.94)
    - Alpha (1Y): 8.03%, Information Ratio: 1.644
    - CAGR 1Y: 22.34% (limited history)
    - Rolling Beat Rate: 91.43%, Cross-Horizon Consistency: N/A (insufficient history)
    - Recovery Speed: N/A, Max Drawdown: N/A

[View Results CSV](../../results/Total%20Market_Codex.csv)

## Backtest Outputs

The model generates several diagnostic files in `data/tmp/`:
- `Total Market_Codex_weights.csv`: Final tuned factor weights
- `Total Market_Codex_backtest.csv`: Historical performance of top-K selections
- `Total Market_Codex_tuning_trials.csv`: All weight combinations tested during tuning
- `Total Market_Codex_regime.csv`: Current market regime detection
