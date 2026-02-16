# Small Cap - Claude Model

**Author**: Claude
**Sector**: Small Cap Fund

## Strategy Overview

This model uses a multi-factor quantitative scoring approach designed to identify funds most likely to deliver superior risk-adjusted returns over the next 1 year. It emphasizes alpha generation and downside protection.

### Key Metrics & Weights

1.  **Alpha & Risk-Adjusted Returns (40%)**
    *   Jensen's Alpha
    *   Sharpe Ratio
    *   Information Ratio
    *   *Rationale*: Small Cap returns are driven more by stock-picking skill than market beta.

2.  **Downside Protection (20%)**
    *   Sortino Ratio
    *   Max Drawdown
    *   Down Capture Ratio
    *   *Rationale*: Large drawdowns in small caps can destroy compounding and take years to recover.

3.  **Consistency (15%)**
    *   Rolling 1-year benchmark beat percentage
    *   Win rate
    *   *Rationale*: Signals a repeatable strategy rather than lucky bets.

4.  **Momentum (10%)**
    *   3-month and 6-month relative returns
    *   *Rationale*: Captures regime persistence in small-cap rallies.

5.  **Return Magnitude (15%)**
    *   CAGR (3Y/5Y)
    *   *Rationale*: Included but down-weighted to avoid chasing past returns.

### Top 5 Funds

1.  **Nippon India Small Cap Fund** (Score: 75.25)
2.  **Quant Small Cap Fund** (Score: 73.75)
3.  **Invesco India Smallcap Fund** (Score: 70.28)
4.  **DSP Small Cap Fund** (Score: 68.28)
5.  **Bandhan Small Cap Fund** (Score: 67.96)

[View Full Results (CSV)](../../results/Small%20Cap_Claude.csv)
