# Small Cap - Codex Model

**Author**: Codex
**Sector**: Small Cap Fund

## Strategy Overview

This model is explicitly optimized for forward 1-year outcomes using a validation-driven approach. Instead of using fixed weights based on theory, it tunes factor weights by backtesting against historical data.

### Methodology

1.  **Historical Panel**: Builds a panel of fund features at many past dates.
2.  **Backtesting**: Tests each date against the next 1-year forward return.
3.  **Tuning**: Tunes factor weights using walk-forward cross-validation.
4.  **Ranking**: Uses the tuned weights to rank funds on the latest date.

### Key Factors (Tuned)

The model dynamically selects weights, but typically favors:
*   **Alpha (1Y)**: Short-term skill.
*   **Information Ratio**: Consistency of excess returns.
*   **Sortino Ratio**: Downside-adjusted returns.
*   **Momentum (3M/6M/12M)**: Relative strength vs benchmark.
*   **Rolling Active Return**: Median active return over 2 years.

### Top 5 Funds

1.  **Invesco India Smallcap Fund** (Score: 71.18)
2.  **Bandhan Small Cap Fund** (Score: 69.60)
3.  **DSP Small Cap Fund** (Score: 67.44)
4.  **Sundaram Small Cap Fund** (Score: 65.88)
5.  **Mirae Asset Small Cap Fund** (Score: 65.13)

[View Full Results (CSV)](../../results/Small%20Cap_Codex.csv)
