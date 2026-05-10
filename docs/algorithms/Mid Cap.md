# Mid Cap Strategy

This strategy targets the Indian Mid Cap mutual fund sector. It combines the insights from four AI models (Claude, Gemini, GPT, Grok) into a composite score.

## Overall Strategy

The Mid Cap sector is known for its growth potential, balancing the stability of large caps with the high growth of small caps. The composite strategy evaluates funds based on:

1.  **Risk-Adjusted Performance**: Ensuring returns are commensurate with the risks taken.
2.  **Consistency**: Favoring funds that consistently outperform the benchmark (Nifty Midcap 150) across different market cycles.
3.  **Downside Protection**: identifying funds that manage capital well during market downturns.
4.  **Momentum & Forward Potential**: Using predictive modeling to identify funds with strong future outlooks.

## Top 5 Funds (Composite Score)

Based on the latest analysis, the top 5 funds are:

1.  **Invesco India Midcap Fund** (Score: 0.960)
2.  **Edelweiss Mid Cap Fund** (Score: 0.927)
3.  **HDFC Mid Cap Fund** (Score: 0.762)
4.  **Union Midcap Fund** (Score: 0.739)
5.  **Baroda BNP Paribas Mid Cap Fund** (Score: 0.739)

[View Full Results (CSV)](../../results/Mid%20Cap.csv)

## Model Breakdown

*   [**Claude**](Mid%20Cap_Claude.md): A comprehensive multi-factor model focusing on risk-adjusted returns, consistency, and market regime behavior.
*   [**GPT**](Mid%20Cap_GPT.md): SIP-forward ranking with a mid-cap prior, seventeen NAV-derived features, and weights learned from look-ahead-safe historical SIP alpha versus Nifty Midcap 150.
*   [**Gemini**](Mid%20Cap_Gemini.md): A balanced scoring system prioritizing consistency (win rate) and standard risk metrics like Sharpe and Sortino ratios.
*   [**Grok**](Mid%20Cap_Grok.md): Nine-factor theory blend (swing elasticity, downside capture, rebound, block alpha, timing convexity, style purity/drift, recovery speed, momentum quality) with signed weights and CDF spread.
