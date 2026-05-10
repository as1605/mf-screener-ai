# Multi Asset Allocation - Grok Model

**Author**: Grok  
**Sector**: Multi Asset Allocation  
**Implementation**: `src/algorithms/Multi Asset_Grok.py`

## Strategy Overview

The model scores **multi-asset allocation** funds using weekly NAV alignment against **gold** (SBI Gold ETF FoF, `M_SBIGL`), **silver** (ICICI Pru Silver ETF FoF, `M_ICPVF` where history allows), and **broad equity** (**Nifty 500**). It stresses **commodity rally capture**, **equity participation on up weeks**, **downside behaviour vs equity**, **rebound elasticity**, **drawdown pain**, **inferred allocation stability**, and **penalties for tactical noise**—with **median/MAD z-scores**, **signed theory weights**, and a **normal CDF** mapping. History and AUM feed **confidence** and **`aum_penalty`**.

### Proxies

Per script: gold, silver, and Nifty 500 charts from `MfDataProvider`.

### Feature set

| Feature | Role |
|--------|------|
| `gold_up_excess` | Fund excess return on weeks gold is up. |
| `equity_up_excess` | Excess on equity-up weeks (participation). |
| `silver_beta_recent` | Recent sensitivity to silver (shorter silver history tolerated). |
| `downside_vs_equity` | Pain vs equity in weak equity regimes (negative weight: lower pain better). |
| `rebound_elasticity_multi` | Multi-benchmark rebound participation. |
| `max_drawdown_pain` | Drawdown severity (negative weight). |
| `allocation_stability` | Stability of inferred mix / path (negative weight on instability). |
| `tactical_noise_penalty` | Penalty for choppy tactical trading signature (negative weight). |
| `regime_persistence` | Persistence across regimes. |

### Final score

Weighted z-composite → CDF → confidence × AUM adjustment → clipped score.

### Top 5 Funds (Grok rank)

Based on the latest `results/Multi Asset_Grok.csv`:

1. **DSP Multi Asset Allocation Fund** (Score: 54.85)
2. **WOC Multi Asset Allocation Fund** (Score: 50.17)
3. **Kotak Multi Asset Allocation Fund** (Score: 46.57)
4. **Mahindra Manulife Multi Asset Allocation Fund** (Score: 43.76)
5. **Bandhan Multi Asset Allocation Fund** (Score: 43.73)

[View Full Results (CSV)](../../results/Multi%20Asset_Grok.csv)
