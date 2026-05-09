# Multi Asset Strategy — Claude

## Overview

**Author:** Claude  
**Implementation:** `src/algorithms/Multi Asset_Claude.py`  
**Sector:** Multi Asset Allocation Fund  

Unlike single-benchmark equity strategies, this model assumes **alpha comes from the asset mix**: equity, debt, gold, and silver, inferred from NAVs alone via **returns-based style analysis (RBSA)**. It combines inferred weights with **Brinson-style** strategic vs tactical decomposition, **downside-first** risk metrics, a **forward-looking “outlook fit”** versus a configurable target mix, and a modest weight on **rolling SIP XIRR** consistency.

## Pillars (conceptual)

| Pillar | Role |
|--------|------|
| **Allocation profile** | Diversification across sleeves, inferred equity/debt/gold/silver weights (silver omitted automatically when history is too short). |
| **Outlook fit** | Squared distance from a stated target mix (penalises overweight to metals after strong rallies, among other tilts). |
| **Allocation skill** | Strategic vs tactical value-add from rolling RBSA snapshots; strategic dominates per design. |
| **Downside** | Sortino, CDaR (5%), Calmar, pain index, current drawdown. |
| **SIP history** | Rolling 1Y SIP XIRR median and consistency (intentionally smaller weight than structural pillars). |

Short track records receive a **haircut**; very short history may yield **score 0** while still emitting a row.

## Proxies

- Equity: `_NIFTY500` (index chart)  
- Gold: `M_SBIGL` (fund chart)  
- Silver: `M_ICPVF` (fund chart; limited history from ~2022)

## Top 5 Funds

From `results/Multi Asset_Claude.csv`:

1. **Quant Multi Asset Allocation Fund** (Score: 73.55) — CAGR 3Y: 24.76%, CAGR 5Y: 21.46%
2. **Aditya Birla SL Multi Asset Allocation Fund** (Score: 73.49) — CAGR 3Y: 18.69%
3. **SBI Multi Asset Allocation Fund** (Score: 71.93) — CAGR 3Y: 18.83%, CAGR 5Y: 15.34%
4. **DSP Multi Asset Allocation Fund** (Score: 69.20)
5. **Nippon India Multi Asset Allocation Fund** (Score: 69.00) — CAGR 3Y: 22.04%, CAGR 5Y: 17.85%

[View Full Results (CSV)](../../results/Multi%20Asset_Claude.csv)
