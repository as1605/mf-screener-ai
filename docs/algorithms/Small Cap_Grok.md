# Small Cap - Grok Model

**Author**: Grok  
**Sector**: Small Cap Fund  
**Implementation**: `src/algorithms/Small Cap_Grok.py`

## Strategy Overview

The model ranks funds for **forward-looking monthly SIP behaviour** using only NAV history and fund metadata, aligned to the **Small Cap** index from `MfDataProvider`. It emphasises **earnings-rebound participation**, **SIP stability**, **recovery after benchmark drawdowns**, and **drawdown avoidance** rather than raw momentum. Eight NAV-derived features are **robust z-scored** (median / MAD) cross-sectionally, combined with **fixed theory weights**, then mapped through a **normal CDF** to spread scores. History length and a **tiered AUM factor** adjust the final score (shorter history is penalised; very small or very large AUM is nudged).

### Benchmark

**Small Cap** index chart from `MfDataProvider` (aligned weekly returns vs fund NAV).

### Feature set

| Feature | Role |
|--------|------|
| `rebound_capture` | Excess fund return vs benchmark on weeks the benchmark is up (recent window). |
| `sip_stability` | Inverse dispersion of rolling ~12-month SIP XIRR outcomes over staggered windows (consistency). |
| `recovery_ratio` | Median post-trough excess return after benchmark drawdown episodes. |
| `vol_alpha` | Mean excess weekly return scaled by volatility of excess returns. |
| `downside_resilience` | Relative behaviour vs benchmark on down weeks. |
| `momentum_persist` | Autocorrelation of weekly returns (persistence of moves). |
| `dd_avoidance` | Fund vs benchmark max drawdown ratio (better avoidance scores higher). |
| `earnings_accel` | Short vs long rolling mean of NAV returns as a simple acceleration proxy. |

Theory weights on z-scores (sum 1.0): rebound and SIP/recovery weighted slightly higher than tail features; see script for exact vector.

### Final score

Weighted z-sum is scaled, passed through `norm.cdf`, multiplied by a **history confidence** and **`aum_factor`**, then clipped to a bounded range.

### Top 5 Funds (Grok rank)

Based on the latest `results/Small Cap_Grok.csv`:

1. **Sundaram Small Cap Fund** (Score: 60.23)
2. **Axis Small Cap Fund** (Score: 57.95)
3. **Bank of India Small Cap Fund** (Score: 56.33)
4. **TRUSTMF Small Cap Fund** (Score: 56.21)
5. **Edelweiss Small Cap Fund** (Score: 56.05)

[View Full Results (CSV)](../../results/Small%20Cap_Grok.csv)
