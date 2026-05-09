# Small Cap - GPT Model

**Author**: GPT  
**Sector**: Small Cap Fund  
**Implementation**: `src/algorithms/Small Cap_GPT.py`

## Strategy Overview

The model ranks funds for the **next 12 months of monthly SIP** (invest on the 1st of each month), using only NAV history and fund metadata. It encodes a **market prior** for small caps (earnings-led recovery narrative, liquidity and valuation risks) into observable signals, then **learns feature weights** from a look-ahead-safe panel of historical 12-month SIP XIRR outcomes versus the benchmark.

### Benchmark

Scores are computed relative to the **Small Cap** index from `MfDataProvider` (`BENCHMARK_INDEX = "Small Cap"`).

### Feature set (composite inputs)

Eleven features are z-scored cross-sectionally (robust MAD-based z-scores with clipping), then combined with learned weights:

| Feature | Role |
|--------|------|
| `sip_hit_rate` | Share of rolling 12-month SIP windows where fund XIRR exceeded benchmark XIRR |
| `alpha_consistency` | Fraction of 13-week rolling periods with positive excess return vs benchmark |
| `momentum_quality` | Blend of 13w / 26w / 52w returns and 26w relative strength, normalized by trailing volatility |
| `recovery_strength` | Fund vs benchmark performance from benchmark trough in recent window |
| `swing_participation` | Upside capture in benchmark rebound weeks (drawdown recovery episodes); falls back to upside capture |
| `drawdown_control` | Average of 52w max drawdown, ~3y max drawdown, and current drawdown (more negative is worse) |
| `downside_resilience` | Combines downside deviation of weekly returns with down-market capture |
| `vol_normalization` | Penalizes rising short-term vol vs longer-term vol (vol regime) |
| `upside_capture` | Upside capture ratio vs benchmark over ~2y of weekly returns |
| `cagr_stability` | Rewards stable trajectory between 3y and 5y rolling CAGR-style returns |
| `aum_liquidity_score` | Soft penalty for very small or very large AUM vs peer distribution (liquidity / capacity) |

### Weight learning

1. **Prior weights** (`PRIOR_WEIGHTS`): Fixed research blend summing to 1.0 across the eleven features.
2. **Training panel**: For each fund and many historical month-end dates, features are computed **as of** that date; the target is the **realized** 12-month SIP XIRR starting that month (and benchmark SIP XIRR). The model prefers **`target_alpha`** (fund minus benchmark SIP XIRR) when enough observations exist.
3. **Spearman correlation**: Each feature’s Spearman correlation with the target is taken as a weight signal (only if correlation exceeds a minimum threshold and enough panel rows exist).
4. **Blend**: Learned weights are mixed **60% learned / 40% prior**, then **capped per feature** so no single signal dominates.

### Final score

Cross-sectional composite z-scores are mapped to percentiles, then scaled by a **confidence** score derived from history length and AUM score (longer history and balanced AUM raise confidence).

### Top 5 Funds (GPT rank)

Based on the latest `results/Small Cap_GPT.csv`:

1. **Union Small Cap Fund** (Score: 96.67)
2. **Bank of India Small Cap Fund** (Score: 93.33)
3. **ITI Small Cap Fund** (Score: 90.00)
4. **TRUSTMF Small Cap Fund** (Score: 82.18)
5. **Bandhan Small Cap Fund** (Score: 81.83)

[View Full Results (CSV)](../../results/Small%20Cap_GPT.csv)
