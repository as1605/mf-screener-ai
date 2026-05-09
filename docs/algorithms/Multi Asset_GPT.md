# Multi Asset Strategy — GPT

## Overview

**Author:** GPT  
**Implementation:** `src/algorithms/Multi Asset_GPT.py`  
**Sector:** Multi Asset Allocation Fund  

The model ranks funds for **next 12 months of monthly SIP** using NAV-derived features tailored to **multi-asset** behaviour: rolling SIP outcomes, participation in equity and metal regimes, drawdown and tail behaviour, and **NAV-regression-based** exposure inference. It **does not** assume last year’s winners repeat; it stresses **consistency across many SIP start dates** and **stress-period resilience**.

## Benchmark & proxies

Signals are built relative to equity and precious-metal proxies available from `MfDataProvider` (broad equity index plus gold/silver fund proxies as in the task spec).

## Feature set

Twenty observables (examples from the script) are robust z-scored cross-sectionally, then combined with weights that blend:

1. **Prior weights** — research-style defaults summing to 1.0.  
2. **Learned adjustment** — Spearman correlation vs a **look-ahead-safe** target (historical 12-month SIP alpha vs benchmark), capped per feature and **shrunk** toward priors when data are thin.

Representative features include:

| Feature | Role |
|--------|------|
| `sip_p50`, `sip_p25`, `sip_hit_rate`, `sip_consistency` | Distribution of rolling 12-month SIP outcomes vs hurdle / peers |
| `risk_on_alpha`, `metal_capture`, `mixed_alpha` | Regime participation vs equity and metals |
| `stress_alpha`, `downside_protection` | Behaviour when correlations spike or markets stress |
| `sortino_3y`, `calmar_3y`, `drawdown_control`, `ulcer_control`, `cvar_control`, `recovery_speed` | Downside and recovery |
| `timing_alignment`, `regime_fit`, `allocation_balance`, `exposure_stability` | Inferred mix dynamics |
| `durability_score` | History / reliability of estimates |

Final scores incorporate a **confidence** adjustment from history length and stability.

## Top 5 Funds

From `results/Multi Asset_GPT.csv`:

1. **SBI Multi Asset Allocation Fund** (Score: 74.78) — CAGR 3Y: 18.9%, CAGR 5Y: 15.4%
2. **DSP Multi Asset Allocation Fund** (Score: 69.36) — CAGR 3Y: 21.54%
3. **WOC Multi Asset Allocation Fund** (Score: 68.53) — CAGR 3Y: 17.38%
4. **Nippon India Multi Asset Allocation Fund** (Score: 67.02) — CAGR 3Y: 22.13%, CAGR 5Y: 17.92%
5. **Mirae Asset Multi Asset Allocation Fund** (Score: 62.37)

[View Full Results (CSV)](../../results/Multi%20Asset_GPT.csv)
