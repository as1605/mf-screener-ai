# Financial Research Synthesis — Claude Retrospective

## Sources Consulted
- SPIVA Persistence Scorecards (2022-2024, India & US)
- Carhart 1997, Berk-Green 2004, Fama-French 2010, Cremers-Petajisto 2009
- SEBI small-cap stress test disclosures (March 2024)
- Morningstar India fund selection research (Kinnel 2010-2019)
- NSE impact cost data for mid/small-cap stocks
- RBI monetary policy outlook (2024-2026)

## Key Findings Applied to Algorithm Design

### 1. What Predicts Forward Outperformance (1-3 Year)?
| Metric | Forward Predictive Power | Action |
|--------|-------------------------|--------|
| Trailing CAGR (1Y/3Y) | Near zero (r ≈ -0.05 to +0.08) | DO NOT use as scoring signal |
| Downside Capture Ratio | Strong & persistent (r ≈ 0.35-0.44) | CORE predictor, weight 2x vs upside |
| Sortino Ratio | Moderate-strong (r ≈ 0.25-0.32) | Keep in all algorithms |
| Information Ratio | Moderate only at 5Y+ | Use with long windows only |
| Expense Ratio (TER) | Strongest predictor overall | Not available in data — acknowledged gap |
| Rolling Alpha Consistency | Persists at 36M forward | Use hit-rate across rolling windows |

### 2. Downside Protection is 2x More Important for 24M SIP+Hold
- Year 1 (SIP): Rupee-cost averaging cushions volatility; drawdowns can actually help
- Year 2 (Hold): Full corpus exposed with zero averaging; drawdowns destroy directly
- Implication: Weight downside metrics (DCR, Sortino, CDaR) more than upside (UCR, raw return)

### 3. Recovery Half-Life is Mostly Beta Noise
- Fast recovery ≈ high beta (drops more, bounces more)
- Valid only as VETO: very slow recovery (>75th percentile of peers) indicates real problems
- Should NOT be scored as positive signal for fast recovery

### 4. Augmented Carhart (Peer-Relative Alpha) Doubles Persistence
- Fund beating both benchmark AND peer median → 2x forward persistence vs single-alpha screen
- Already in Total Market → propagate to Mid Cap, Small Cap

### 5. AUM Capacity Constraints (India-Specific)
- Small Cap: Impact costs surge above ₹15,000-25,000 Cr AUM
- Mid Cap: Inflection at ₹20,000-35,000 Cr
- Empirical: Large small-cap funds need 27-60 days to liquidate 50% of portfolio
- Current Mid Cap thresholds (15k/25k/40k/60k Cr) are roughly appropriate

### 6. Turnover Negatively Correlated with Performance in India
- High turnover (>200%) in mid/small caps loses 150-250 bps in execution friction
- STT (0.1%), impact cost (0.15-1.0%), brokerage compound significantly
- Exception: Pure quantitative momentum strategies (Quant MF) — but capacity-constrained
- Confirms: Mid Cap P5 should PENALIZE high turnover, not reward it

### 7. James-Stein Shrinkage Parameters
- True skill dispersion (τ) in Indian mid/small caps: 1.8-2.8% annualized
- At T=3Y: ~67.5% shrinkage toward peer mean
- At T=1Y: ~86% shrinkage
- At T=5Y: ~55% shrinkage
- Shrinkage target: Category peer median (not zero)

### 8. Multi-Asset: Gold's Counter-Cyclical Hedge via Rupee Depreciation
- Gold (INR) vs Nifty 50: -0.08 to +0.12 correlation (near zero/negative)
- During crises: FPI outflows → INR depreciation → amplifies gold returns in INR
- 2008: Nifty -52%, Gold (INR) +28.4%
- Confirms: Reward demonstrated allocation skill, not subjective macro view

### 9. Indian Market Valuations (Sep 2026 context)
- Midcap P/E at +2.0σ above 10Y median (32-36x vs 24.5x median)
- Smallcap P/E at +1.8σ (28.5-33x vs 20x median)
- Buffett indicator (Market Cap/GDP) above 130% vs 85-90% historical
- Implication: Downside resilience metrics should be prioritized over upside capture
