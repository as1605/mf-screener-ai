# Mid Cap Strategy

This strategy targets the Indian Mid Cap mutual fund sector. It combines the insights from three different AI models (Claude, Gemini, Codex) to create a robust composite score.

## Overall Strategy

The Mid Cap sector is known for its growth potential, balancing the stability of large caps with the high growth of small caps. The composite strategy evaluates funds based on:

1.  **Risk-Adjusted Performance**: Ensuring returns are commensurate with the risks taken.
2.  **Consistency**: Favoring funds that consistently outperform the benchmark (Nifty Midcap 150) across different market cycles.
3.  **Downside Protection**: identifying funds that manage capital well during market downturns.
4.  **Momentum & Forward Potential**: Using predictive modeling to identify funds with strong future outlooks.

## Top 5 Funds (Composite Score)

Based on the latest analysis, the top 5 funds are:

1.  **HDFC Mid Cap Fund** (Score: 0.989)
2.  **Invesco India Midcap Fund** (Score: 0.917)
3.  **Edelweiss Mid Cap Fund** (Score: 0.879)
4.  **Nippon India Growth Mid Cap Fund** (Score: 0.829)
5.  **ICICI Pru Midcap Fund** (Score: 0.795)

[View Full Results (CSV)](../../results/Mid%20Cap.csv)

## Model Breakdown

*   [**Claude**](Mid%20Cap_Claude.md): A comprehensive multi-factor model focusing on risk-adjusted returns, consistency, and market regime behavior.
*   [**Codex**](Mid%20Cap_Codex.md): A research-led model optimized for predicting 1-year forward excess returns using walk-forward validation.
*   [**Gemini**](Mid%20Cap_Gemini.md): A balanced scoring system prioritizing consistency (win rate) and standard risk metrics like Sharpe and Sortino ratios.
