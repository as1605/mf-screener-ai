# Total Market - GPT Model

**Author**: GPT  
**Sector**: Total Market (Flexi Cap, Multi Cap, Focused, Value, Contra)  
**Implementation**: `src/algorithms/Total Market_GPT.py`

## Strategy Overview

The model ranks funds for the user cashflow in the task brief: **12 monthly SIPs**, **12 months hold**, **single exit at month 24**. It optimises signals aligned with that **hybrid SIP-hold XIRR**, using weekly NAV resampling, rolling scenario windows, benchmark and peer comparison on XIRR paths, and multi-index factor adjustment (large, mid, small, total market).

### Benchmark and factors

- **Benchmark**: Total Market index from `MfDataProvider` (`BENCHMARK_INDEX = "Total Market"`).
- **Factor indices**: Large Cap, Mid Cap, Small Cap, and Total Market for alpha and style tilt.

### Score pillars

Fixed pillar weights (`PILLAR_WEIGHTS`) combine into the final score:

| Pillar | Weight | Role |
|--------|--------|------|
| **Hybrid** | 34% | Rolling SIP-then-hold XIRR distribution (median, lower tail, recent window), hit rates vs benchmark and peers |
| **Recovery** | 18% | Participation in rebounds after stress; bull recovery excess; momentum-style relatives |
| **Resilience** | 22% | Drawdown depth (e.g. 2Y max drawdown), Ulcer-style pain, downside capture, bear regime behaviour, recovery speed |
| **Active skill** | 18% | Factor-adjusted alpha and IR, alpha stability, market beta and small/mid tilt versus mandate |
| **Confidence** | 8% | History length, XIRR scenario coverage, data freshness, capacity-style multiplier |

Outputs include hybrid XIRR stats, recovery and resilience diagnostics, factor alpha columns, and regime labels (e.g. sideways) for transparency.

### Top 5 Funds (GPT rank)

From `results/Total Market_GPT.csv`:

1. **Axis Value Fund** (Score: 76.59)
2. **HSBC Value Fund** (Score: 76.52)
3. **HDFC Focused Fund** (Score: 74.46)
4. **Kotak Multicap Fund** (Score: 74.13)
5. **ICICI Pru Focused Equity Fund** (Score: 73.98)

[View Results CSV](../../results/Total%20Market_GPT.csv)
