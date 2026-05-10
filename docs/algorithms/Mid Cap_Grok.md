# Mid Cap - Grok Model

**Author**: Grok  
**Sector**: Mid Cap Fund  
**Implementation**: `src/algorithms/Mid Cap_Grok.py`

## Strategy Overview

The model targets **1–2 year forward SIP-style outcomes** using a **theory-heavy hybrid**: nine weekly NAV features vs **Midcap 150** and style references (large / small cap index charts where used in the script for style purity and block alpha). Features are **winsorised**, **median/MAD z-scored**, combined with **signed theory weights** (some features are “lower is better”, encoded as negative weights), then mapped with a **normal CDF** for spread. Light **history** and **AUM** penalties apply.

### Benchmarks

Primary **Mid Cap** index; additional index series support style and participation metrics inside the script.

### Feature set

| Feature | Role |
|--------|------|
| `swing_elasticity` | Participation vs benchmark around large benchmark swings. |
| `downside_capture` | Down-market behaviour vs benchmark (weight negative: less loss vs benchmark is better). |
| `rebound_elasticity` | Recovery after benchmark stress. |
| `block_alpha` | Alpha-style block vs style-adjusted reference. |
| `timing_convexity` | Convexity / timing asymmetry of returns vs benchmark. |
| `style_purity` | How cleanly the path tracks mid-cap vs large/small tilts. |
| `style_drift` | Drift of style exposure (lower drift preferred; negative weight). |
| `recovery_weeks` | Speed of recovery after drawdowns (faster preferred; negative weight). |
| `momentum_quality` | Risk-adjusted momentum quality vs noise. |

### Final score

Weighted z-composite → normal CDF scaling → **confidence** (full year of data vs not) × **`aum_penalty`** → clipped output.

### Top 5 Funds (Grok rank)

Based on the latest `results/Mid Cap_Grok.csv`:

1. **Invesco India Midcap Fund** (Score: 65.10)
2. **Edelweiss Mid Cap Fund** (Score: 61.59)
3. **Baroda BNP Paribas Mid Cap Fund** (Score: 57.08)
4. **Tata Mid Cap Fund** (Score: 56.44)
5. **HSBC Midcap Fund** (Score: 55.49)

[View Full Results (CSV)](../../results/Mid%20Cap_Grok.csv)
