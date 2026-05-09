# Mid Cap - GPT Model

**Author**: GPT  
**Sector**: Mid Cap Fund  
**Implementation**: `src/algorithms/Mid Cap_GPT.py`

## Strategy Overview

The model ranks funds for the **next 12 months of monthly SIP** (invest on the 1st of each month), using NAV history, the mid-cap benchmark from `MfDataProvider`, and fund metadata. It bakes in a **market prior** tuned to mid caps (earnings-led cycles, liquidity and valuation asymmetry, preference for persistent SIP alpha versus drawdown discipline), then **learns feature weights** from a look-ahead-safe panel of historical 12-month SIP outcomes versus the benchmark.

### Benchmark

Scores are computed relative to the **Mid Cap** index from `MfDataProvider` (`BENCHMARK_INDEX = "Mid Cap"`, Nifty Midcap 150).

### Feature set (composite inputs)

Seventeen features are direction-aligned, robust z-scored cross-sectionally (MAD-based with clipping), then combined with learned weights shrunk toward the prior:

| Feature | Role |
|--------|------|
| `sip_xirr_median` | Central tendency of rolling 12-month SIP XIRR |
| `sip_xirr_p25` | Lower-tail SIP outcome (stress on typical paths) |
| `sip_alpha_hit_rate` | Share of rolling SIP windows where fund XIRR beat benchmark |
| `alpha_consistency` | Stability of positive excess return vs benchmark |
| `information_ratio` | Active return per unit of tracking error |
| `up_capture` | Upside capture vs benchmark |
| `down_capture` | Downside capture (lower is better after direction flip) |
| `downside_beta` | Sensitivity in weak benchmark periods |
| `drawdown_control` | Depth and severity of drawdowns |
| `recovery_strength` | Participation after benchmark corrections |
| `correction_defense` | Behaviour in correction regimes |
| `momentum_quality` | Risk-adjusted momentum blend |
| `omega_ratio` | Gain-to-loss asymmetry of returns |
| `ulcer_index` | Drawdown pain (lower is better) |
| `cdar_5` | Conditional drawdown-at-risk style tail metric |
| `style_purity` | Alignment with mid-cap mandate / style stability |
| `aum_score` | Peer-relative AUM sanity (capacity / liquidity) |

### Weight learning

1. **Prior weights** (`PRIOR_WEIGHTS`): Fixed blend across the seventeen features (sums to 1.0).
2. **Training panel**: Many historical month-ends; features as-of each date; target is realized **12-month SIP alpha** (fund minus benchmark) when enough observations exist.
3. **Spearman signal**: Each feature’s correlation with the target informs learned weights (thresholds on correlation and panel size).
4. **Shrinkage**: Learned weights are blended **toward the prior** and **per-feature capped** so no single signal dominates.

### Final score

Composite z-scores map to percentiles and are scaled by a **confidence** score from history length and data quality (longer, cleaner histories raise confidence).

### Top 5 Funds (GPT rank)

Based on the latest `results/Mid Cap_GPT.csv`:

1. **WOC Mid Cap Fund** (Score: 79.46)
2. **HDFC Mid Cap Fund** (Score: 78.55)
3. **Edelweiss Mid Cap Fund** (Score: 78.15)
4. **Invesco India Midcap Fund** (Score: 77.79)
5. **Kotak Midcap Fund** (Score: 73.35)

[View Full Results (CSV)](../../results/Mid%20Cap_GPT.csv)
