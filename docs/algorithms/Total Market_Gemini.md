# Total Market Strategy - Gemini

## Overview

**Author**: Gemini
**Focus**: Multi-factor regression to isolate "pure alpha" from market-cap exposure.
**Target**: Identify funds generating excess returns beyond their market-cap bias.
**Subsectors**: Contra Fund, Flexi Cap Fund, Focused Fund, Multi Cap Fund, Value Fund

This model differentiates itself by recognizing that Total Market funds often generate returns simply by tilting towards Mid/Small caps. Instead of rewarding this "fake alpha," Gemini's model decomposes returns into Large, Mid, and Small cap factors to identify funds that generate true alpha *after* accounting for their cap bias.

## Key Metrics & Weights

| Category | Weight | Key Metrics |
| :--- | :--- | :--- |
| **Multi-Factor Alpha** | **30%** | Alpha from regression: R_fund = α + β₁R_large + β₂R_mid + β₃R_small |
| **Downside Resistance** | **20%** | Ulcer Index (depth + duration of drawdowns), Max Drawdown |
| **Consistency of Skill** | **20%** | Rolling Information Ratio (1-year windows, penalized by volatility) |
| **Upside Potential** | **15%** | Omega Ratio (probability-weighted gains vs losses) |
| **Momentum** | **15%** | EWMA of recent returns (26-week span) |

## Key Innovations

*   **Multi-Factor Alpha**: Performs regression to decompose fund returns into Large, Mid, and Small cap exposures. Rewards funds that generate alpha *after* accounting for their cap bias, penalizing "fake alpha" from high small-cap beta.
*   **Rolling Information Ratio**: Calculates IR stability across rolling 1-year windows. Rewards funds that consistently deliver risk-adjusted excess returns, rather than those with one lucky year.
*   **Omega Ratio**: Captures the non-normal distribution of returns better than Sharpe Ratio by using probability-weighted ratio of gains vs losses.
*   **Ulcer Index**: Measures both depth and duration of drawdowns (RMS of percentage drawdowns), focusing on the "pain" felt by investors during holding periods.
*   **Confidence Adjustment**: Scores are adjusted based on data history length (60% for <1Y, 75% for 1-2Y, 90% for 2-3Y, 100% for 3Y+).

## Analysis Window

- **Benchmark**: Nifty 500 (Total Market)
- **Factor Indices**: Large Cap (.NSEI), Mid Cap (.NIMI150), Small Cap (.NISM250)
- **Minimum Data**: 50 weeks for inclusion
- **Risk-Free Rate**: 6.5% annualized

## Top 5 Funds

Based on this strategy, the top 5 funds are:

1.  **Mahindra Manulife Focused Fund** (Score: 88.92)
    - Multi-Factor Alpha: 6.38%
    - CAGR 3Y: 21.49%, CAGR 5Y: 20.11%
    - Rolling IR: 0.78, Omega Ratio: 1.49

2.  **Kotak Multicap Fund** (Score: 83.78)
    - Multi-Factor Alpha: 5.40%
    - CAGR 3Y: 25.34%
    - Rolling IR: 0.85, Omega Ratio: 1.33

3.  **Axis Value Fund** (Score: 83.33)
    - Multi-Factor Alpha: 5.29%
    - CAGR 3Y: 24.45%
    - Rolling IR: 0.79, Omega Ratio: 1.31

4.  **ICICI Pru Flexicap Fund** (Score: 78.24)
    - Multi-Factor Alpha: 4.40%
    - CAGR 3Y: 20.48%
    - Rolling IR: 0.69, Omega Ratio: 1.31

5.  **Kotak Contra Fund** (Score: 77.39)
    - Multi-Factor Alpha: 3.82%
    - CAGR 3Y: 22.40%, CAGR 5Y: 18.73%
    - Rolling IR: 0.49, Omega Ratio: 1.30

[View Results CSV](../../results/Total%20Market_Gemini.csv)
