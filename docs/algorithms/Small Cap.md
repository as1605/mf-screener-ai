# Small Cap Strategy

This strategy targets the Indian Small Cap mutual fund sector. It combines the insights from three different AI models (Claude, Gemini, GPT) to create a robust composite score.

## Overall Strategy

The Small Cap sector is characterized by high volatility and high potential returns. The composite strategy balances these factors by:

1. **Rewarding consistency**: Funds that repeatedly beat the benchmark (Nifty SmallCap 250) on rolling horizons and in historical monthly SIP windows.
2. **Managing risk**: Penalizing deep drawdowns and weak downside capture relative to the benchmark.
3. **Capturing participation**: Favoring funds that participate in recoveries and benchmark rebounds without chasing raw volatility.
4. **Blending research with data**: The GPT model mixes a fixed research prior on feature importance with weights partially learned from look-ahead-safe historical 12-month SIP outcomes.

## Top 5 Funds (Composite Score)

Based on the latest analysis, the top 5 funds are:

1. **Bank of India Small Cap Fund** (Score: 0.984)
2. **Union Small Cap Fund** (Score: 0.892)
3. **TRUSTMF Small Cap Fund** (Score: 0.827)
4. **DSP Small Cap Fund** (Score: 0.795)
5. **Bandhan Small Cap Fund** (Score: 0.742)

[View Full Results (CSV)](../../results/Small%20Cap.csv)

## Model Breakdown

* [**Claude**](Small%20Cap_Claude.md): Focuses on risk-adjusted returns and consistency.
* [**Gemini**](Small%20Cap_Gemini.md): Uses advanced statistical metrics like Omega Ratio and Hurst Exponent.
* [**GPT**](Small%20Cap_GPT.md): SIP-aligned NAV features, benchmark-relative capture and recovery signals, AUM-aware confidence, and blended prior + correlation-based feature weights.
