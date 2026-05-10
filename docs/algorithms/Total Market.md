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
1. Axis Value Fund (Score: 81.53)
2. Kotak Multicap Fund (Score: 80.54)
3. ICICI Pru Value Fund (Score: 80.13)
4. ICICI Pru Focused Equity Fund (Score: 78.84)
5. HSBC Value Fund (Score: 77.48)

**Key Metrics**: Gain-to-Pain Ratio, CVaR, Excess Return Autocorrelation, Regime Transition Alpha, Cross-Horizon Rank Consistency.

[View Results CSV](../../results/Total%20Market_Claude.csv)

### 2. [Gemini's Multi-Factor Alpha Model](./Total%20Market_Gemini.md)
*Focus: Pure alpha isolation from market-cap exposure using multi-factor regression.*

**Top 5 Funds:**
1. ICICI Pru Flexicap Fund (Score: 86.96)
2. Axis Value Fund (Score: 86.78)
3. DSP Value Fund (Score: 84.65)
4. Invesco India Flexi Cap Fund (Score: 83.13)
5. Canara Rob Focused Fund (Score: 82.39)

**Key Metrics**: Multi-Factor Alpha (Large/Mid/Small cap decomposition), Ulcer Index, Rolling Information Ratio, Omega Ratio, Momentum EWMA.

[View Results CSV](../../results/Total%20Market_Gemini.csv)

### 3. [GPT's Hybrid SIP-Hold XIRR Model](./Total%20Market_GPT.md)
*Focus: Rolling 12+12 month SIP-hold paths, recovery and resilience vs benchmark and peers, factor-adjusted skill.*

**Top 5 Funds:**
1. Axis Value Fund (Score: 76.59)
2. HSBC Value Fund (Score: 76.52)
3. HDFC Focused Fund (Score: 74.46)
4. Kotak Multicap Fund (Score: 74.13)
5. ICICI Pru Focused Equity Fund (Score: 73.98)

**Key Metrics**: Hybrid XIRR median/tails, hit rates vs benchmark and peers, recovery capture, Ulcer/tail loss, factor alpha and IR, capacity/history confidence.

[View Results CSV](../../results/Total%20Market_GPT.csv)

### 4. [Grok's Scenario XIRR + Skill Model](./Total%20Market_Grok.md)
*Focus: Hybrid scenario XIRR, information ratio, downside capture, theme tilt, conviction.*

**Top 5 Funds:**
1. ITI Focused Fund (Score: 64.02)
2. WOC Multi Cap Fund (Score: 62.49)
3. Axis Value Fund (Score: 62.18)
4. Helios Flexi Cap Fund (Score: 59.00)
5. Bajaj Finserv Flexi Cap Fund (Score: 58.68)

**Key Metrics**: Mean/min scenario XIRR, recovery days, downside capture, information ratio, size alpha, theme tilt, conviction.

[View Results CSV](../../results/Total%20Market_Grok.csv)

## Comparative Notes

All four models target diversified equity mandates but emphasize different evidence:

- **Claude** rewards adaptive skill, tail-aware path quality, and regime behaviour.
- **Gemini** stresses alpha after factor decomposition and downside-aware ratios.
- **GPT** aligns explicitly with the **12 SIP + 12 hold + exit** cashflow and blends hybrid XIRR evidence with resilience and factor skill.
- **Grok** stress-tests hybrid XIRR scenarios and combines IR, recovery timing, and macro-theme tilts.

**Axis Value Fund** ranks highly across Claude, Gemini, and GPT, and appears in Grok’s top three—useful consensus for deeper research. Composite rankings in `results/Total Market.csv` blend normalized scores from all models.

[View composite results](../../results/Total%20Market.csv) · [View all sector CSVs](../../results/)
