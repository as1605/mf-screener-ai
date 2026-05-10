# Total Market - Grok Model

**Author**: Grok  
**Sector**: Total Market (Flexi Cap, Multi Cap, Focused, Value, Contra)  
**Implementation**: `src/algorithms/Total Market_Grok.py`

## Strategy Overview

Designed for the **24-month horizon** (12 SIP + 12 hold + exit) with emphasis on **hybrid cashflow XIRR**, manager skill (information ratio, alpha, consistency), and resilience through the hold phase. Uses NAV from `MfDataProvider`, size indices for attribution-style tilts, and theme proxies aligned to stated macro themes (e.g. financials, infra, consumption, manufacturing).

### Benchmark

Scores are anchored to the broad market via **`BENCHMARK_INDEX = ".NIFTY500"`** (Nifty 500) with **Large / Mid / Small** paths from `SIZE_INDICES` for size-adjusted behaviour.

### Composite weights

Nine components (`WEIGHTS`) combine theory-weighted scores:

| Component | Weight | Role |
|-----------|--------|------|
| `xirr_mean` | 22% | Central tendency of rolling hybrid SIP-hold XIRR scenarios |
| `xirr_min` | 13% | Stress tail of scenario XIRR |
| `xirr_consistency` | 8% | Stability of scenario outcomes |
| `recovery_days` | 12% | Drawdown recovery timing |
| `downside_capture` | 10% | Weak-market capture vs benchmark |
| `info_ratio` | 15% | Active return per unit of risk |
| `size_alpha` | 8% | Alpha versus size benchmarks |
| `theme_tilt` | 7% | Sector/theme alignment vs growth-theme buckets |
| `conviction` | 5% | Concentration / conviction proxy |

### Top 5 Funds (Grok rank)

From `results/Total Market_Grok.csv`:

1. **ITI Focused Fund** (Score: 64.02)
2. **WOC Multi Cap Fund** (Score: 62.49)
3. **Axis Value Fund** (Score: 62.18)
4. **Helios Flexi Cap Fund** (Score: 59.00)
5. **Bajaj Finserv Flexi Cap Fund** (Score: 58.68)

[View Results CSV](../../results/Total%20Market_Grok.csv)
