# Total Market Strategies

This section covers mutual funds that invest across the entire market capitalization spectrum, including:
- **Flexi Cap Funds**: Can invest anywhere (Large, Mid, Small) dynamically.
- **Multi Cap Funds**: Must invest a minimum percentage in Large, Mid, and Small caps.
- **Focused Funds**: Concentrated portfolio (max 30 stocks).
- **Value Funds**: Invest in undervalued stocks.
- **Contra Funds**: Invest in out-of-favor sectors/stocks.

These funds offer the highest flexibility to fund managers to generate alpha.

## Available Algorithms

### 1. [Claude's Adaptive Conviction Model](./Total%20Market_Claude.md)
*Focus: Skill persistence, path quality, and regime adaptability.*

**Top 5 Funds:**
1. Mahindra Manulife Focused Fund (Score: 81.84)
2. LIC MF Multi Cap Fund (Score: 74.74)
3. DSP Value Fund (Score: 74.35)
4. Axis Value Fund (Score: 71.19)
5. Edelweiss Focused Fund (Score: 71.08)

**Key Metrics**: Gain-to-Pain Ratio, CVaR, Excess Return Autocorrelation, Regime Transition Alpha, Cross-Horizon Rank Consistency.

[View Results CSV](../../results/Total%20Market_Claude.csv)

### 2. [Gemini's Multi-Factor Alpha Model](./Total%20Market_Gemini.md)
*Focus: Pure alpha isolation from market-cap exposure using multi-factor regression.*

**Top 5 Funds:**
1. Mahindra Manulife Focused Fund (Score: 88.92)
2. Kotak Multicap Fund (Score: 83.78)
3. Axis Value Fund (Score: 83.33)
4. ICICI Pru Flexicap Fund (Score: 78.24)
5. Kotak Contra Fund (Score: 77.39)

**Key Metrics**: Multi-Factor Alpha (Large/Mid/Small cap decomposition), Ulcer Index, Rolling Information Ratio, Omega Ratio, Momentum EWMA.

[View Results CSV](../../results/Total%20Market_Gemini.csv)

### 3. [Codex's Walk-Forward Tuned Model](./Total%20Market_Codex.md)
*Focus: Validation-driven weight optimization with subsector-aware blending.*

**Top 5 Funds:**
1. ICICI Pru Focused Equity Fund (Score: 89.6)
2. HDFC Focused Fund (Score: 82.71)
3. HDFC Flexi Cap Fund (Score: 82.11)
4. Kotak Focused Fund (Score: 81.23)
5. Groww Multicap Fund (Score: 79.94)

**Key Metrics**: Rolling Alpha Stability, Cross-Horizon Consistency, Recovery Speed, Rolling Beat Rate, Volatility-Adjusted Momentum.

[View Results CSV](../../results/Total%20Market_Codex.csv)

## Comparative Performance

All three models identify high-quality funds, but with different emphases:

- **Claude** rewards funds that adapt well to market regimes with persistent alpha generation and strong tail risk management.
- **Gemini** isolates "pure alpha" by decomposing returns into market-cap factors, rewarding funds that generate excess returns beyond their cap bias.
- **Codex** uses walk-forward validation to tune weights based on historical performance, favoring funds with consistent risk-adjusted returns across different market conditions.

Interestingly, **Mahindra Manulife Focused Fund** appears in the top picks of both Claude and Gemini, suggesting strong consensus on its quality. Codex's top pick, **ICICI Pru Focused Equity Fund**, demonstrates exceptional consistency metrics validated through backtesting.

[View All Results](../../results/)
