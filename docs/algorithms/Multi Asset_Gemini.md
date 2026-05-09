# Multi Asset Strategy — Gemini

## Overview

**Author:** Gemini  
**Implementation:** `src/algorithms/Multi Asset_Gemini.py`  
**Sector:** Multi Asset Allocation Fund  

This variant emphasises **cycle-aware equity behaviour** and **precious metals upside**, alongside **downside-focused Sortino** and **stability of rolling SIP returns**—aligned with mandates that should participate in equity and metal rallies while managing drawdowns.

## Key metrics

| Metric | Role |
|--------|------|
| **Cycle agility** | Difference between equity **up-market** and **down-market beta** (reward shifting exposure through cycles). |
| **Precious metals capture** | How much of gold/silver upside the fund captures when those proxies rally. |
| **Sortino (3Y)** | Return per unit of downside deviation. |
| **SIP stability** | Consistency of realised 1-year SIP outcomes over rolling windows. |

Equity benchmark exposure uses the provider’s broad equity index mapping; gold and silver use the standard multi-asset proxies (`M_SBIGL`, `M_ICPVF`).

## Top 5 Funds

From `results/Multi Asset_Gemini.csv`:

1. **SBI Multi Asset Allocation Fund** (Score: 64.30)
2. **ICICI Pru Multi-Asset Fund** (Score: 53.44)
3. **HDFC Multi-Asset Allocation Fund** (Score: 48.83)
4. **Tata Multi Asset Allocation Fund** (Score: 48.72)
5. **Nippon India Multi Asset Allocation Fund** (Score: 46.55)

[View Full Results (CSV)](../../results/Multi%20Asset_Gemini.csv)
