# Small Cap Strategy

This strategy targets the Indian Small Cap mutual fund sector. It combines the insights from three different AI models (Claude, Gemini, Codex) to create a robust composite score.

## Overall Strategy

The Small Cap sector is characterized by high volatility and high potential returns. The composite strategy balances these factors by:
1.  **Rewarding Consistency**: Funds that consistently beat the benchmark (Nifty SmallCap 250).
2.  **Managing Risk**: Penalizing funds with deep drawdowns or poor downside protection.
3.  **Capturing Momentum**: Identifying funds that are currently in a favorable trend.

## Top 5 Funds (Composite Score)

Based on the latest analysis, the top 5 funds are:

1.  **Bandhan Small Cap Fund** (Score: 0.936)
2.  **Nippon India Small Cap Fund** (Score: 0.934)
3.  **DSP Small Cap Fund** (Score: 0.913)
4.  **Invesco India Smallcap Fund** (Score: 0.909)
5.  **Mahindra Manulife Small Cap Fund** (Score: 0.792)

[View Full Results (CSV)](../../results/Small%20Cap.csv)

## Model Breakdown

*   [**Claude**](Small%20Cap_Claude.md): Focuses on risk-adjusted returns and consistency.
*   [**Gemini**](Small%20Cap_Gemini.md): Uses advanced statistical metrics like Omega Ratio and Hurst Exponent.
*   [**Codex**](Small%20Cap_Codex.md): Optimizes for 1-year forward returns using backtested weights.
