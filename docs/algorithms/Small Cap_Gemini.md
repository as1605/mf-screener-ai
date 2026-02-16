# Small Cap - Gemini Model

**Author**: Gemini
**Sector**: Small Cap Fund

## Strategy Overview

This model is optimized for predicting performance over the next 1 year. It introduces advanced statistical metrics to capture non-normal return distributions and trend persistence common in small caps.

### Key Differentiators

1.  **Omega Ratio**: Captures all higher moments (skewness, kurtosis) of return distributions.
2.  **Hurst Exponent**: Identifies funds with persistent trending behavior (momentum).
3.  **Upside Potential Ratio**: Focuses on the asymmetry of returns (we want upside > downside).

### Key Metrics & Weights

*   **Momentum (3M/6M)**: 20% - High weight for short-term prediction.
*   **Alpha**: 20% - Skill persistence.
*   **Omega Ratio**: 15% - Better risk-adjusted measure for non-normal returns.
*   **Sortino Ratio**: 10% - Downside protection.
*   **Hurst Exponent**: 10% - Trend persistence.
*   **Upside Potential**: 10% - Asymmetry.
*   **CAGR (3Y)**: 10% - Medium term consistency.
*   **Max Drawdown**: 5% - Disaster avoidance.

### Top 5 Funds

1.  **Nippon India Small Cap Fund** (Score: 77.42)
2.  **Bandhan Small Cap Fund** (Score: 75.50)
3.  **DSP Small Cap Fund** (Score: 73.50)
4.  **Axis Small Cap Fund** (Score: 69.42)
5.  **Invesco India Smallcap Fund** (Score: 65.25)

[View Full Results (CSV)](../../results/Small%20Cap_Gemini.csv)
