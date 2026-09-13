# Claude Algorithm Retrospective Report — September 2026

## Executive Summary

This retrospective analyzed Claude's 4 mutual fund scoring algorithms (Mid Cap, Small Cap, Multi Asset, Total Market), identified root-cause issues grounded in financial theory and academic research, and implemented targeted improvements. The philosophy throughout was **financial rationale over data-mining** — every change was justified by established research, not by chasing 4 months of realized XIRR data.

## Research Foundation

Before any code changes, we conducted extensive financial research covering:
- **SPIVA Persistence Scorecards** (India & US): Only 2-4% of top-quartile funds maintain status for 3+ consecutive years
- **Academic alpha persistence** (Carhart 1997, Berk-Green 2004, Fama-French 2010): Trailing CAGR has near-zero predictive power; downside capture is the strongest persistent signal (r ≈ 0.35-0.44)
- **Indian market dynamics**: Mid/small cap valuations at +1.8-2.0σ above 10Y medians; SEBI stress test data showing large small-cap funds need 27-60 days to liquidate 50% of portfolios
- **Skill vs luck separation**: James-Stein shrinkage, Information Ratio persistence, Augmented Carhart dual-veto methodology
- Full research notes: `results/experiments/claude_retro_research_notes.md`

## Data Upgrade: Weekly → Daily NAV

**Change**: Switched all 4 algorithms from `duration='max'` (weekly NAV, ~260 points/5yr) to `duration='5y'` (daily NAV, ~1250 points/5yr).

**Why**: Daily resolution enables precise 1st-of-month SIP buy dates, accurate intra-week drawdown measurement, and higher-fidelity rolling metrics. Where algorithms expected weekly data, we added explicit weekly resampling to maintain computational efficiency while benefiting from daily-granularity where needed.

---

## Category-by-Category Analysis

### Mid Cap — Surgical Fixes (3 changes)

**Pre-existing strength**: Regime-conditional 5-pillar architecture with forward alpha projection, James-Stein shrinkage on block alphas. This was already the strongest algorithm across all models.

| Fix | What Changed | Financial Rationale |
|-----|-------------|-------------------|
| P3: 12m SIP → 24m Hybrid | SIP simulation now includes 12-month hold phase | Hold phase exposes full corpus without DCA cushion; funds that accumulate cheaply must also hold value |
| P5: Turnover direction | Flipped `change3m` from `higher_is_better=True` to `False` | Research confirms 150-250 bps annual execution drag from high turnover in Indian mid-caps (STT + impact cost + brokerage) |
| Regime matrix | Added 15% uniform prior blend to empirical transition matrix | Prevents overconfident regime predictions based on limited bear-market observations in the 5-year training window |

**Result**: HDFC Mid Cap, WOC Mid Cap, Invesco India Midcap as top 3. Cross-sectional persistence IC = +0.154.

### Small Cap — Anti-Overfitting Restructure (3 changes)

**Pre-existing strength**: Self-backtested adaptive weights with directional priors — clever diagnostic framework. **Pre-existing weakness**: Pure data-driven weights from ~2-3 years of non-overlapping windows create overfitting risk.

| Change | What Changed | Financial Rationale |
|--------|-------------|-------------------|
| 24m Hybrid Horizon | Target variable upgraded from 12m SIP XIRR to 24m hybrid | Aligns to actual investment scenario; hold-phase resilience matters more in small caps due to liquidity risk |
| 70/30 Theory/Data Blend | Feature weights now blend 70% financial-theory priors with 30% self-backtested correlations | Prevents noisy empirical correlations from dominating; theory weights grounded in DCR persistence research |
| Stress Alpha Signal | New feature: fund alpha during worst 10% of benchmark weeks | Small-cap-specific: liquidity freezes during stress are the primary risk; funds that protect capital during these events compound dramatically better |

**Result**: ITI Small Cap, Invesco India Smallcap, Bandhan Small Cap as top 3. Theory weights prevent the algorithm from flipping on noise.

### Multi Asset — Structural Redesign (4 changes)

**Pre-existing strength**: RBSA (Sharpe 1988) + Brinson attribution — correct analytical framework for multi-asset funds. **Pre-existing weakness**: `TARGET_MIX_2026` hardcoded macro bet (60/15/15/10) as 15% of the score — this is a subjective forecast, not a skill assessment.

| Change | What Changed | Financial Rationale |
|--------|-------------|-------------------|
| Remove TARGET_MIX_2026 | Deleted subjective macro view and outlook_fit pillar | Quantitative screener should reward demonstrated skill, not penalize disagreement with an analyst's forecast |
| P2: Allocation Adaptability | New pillar measuring correlation of weight changes with subsequent asset returns | Rewards funds that demonstrably adjust allocations in ways that predict forward returns (genuine tactical skill) |
| RBSA Window: 52→78 weeks | Longer rolling window for returns-based style analysis | Multi-asset allocation is strategic (quarterly/annual); 78-week windows reduce parameter variance with 4 correlated assets |
| P5: 24m Hybrid SIP | SIP evaluation now includes hold phase | Consistent with all categories; SIP hurdle raised from 8% to 10% for the 24m multi-asset horizon |

**Result**: Nippon India Multi Asset, DSP Multi Asset, Union Multi Asset as top 3.

### Total Market — Bug Fix + Fairness Improvement (2 changes)

**Pre-existing strength**: Correct 24-month hybrid horizon, SPIVA-grounded pillar design, walk-forward IC diagnostic. **Pre-existing weakness**: Zero-score cliff bug and cross-subsector structural bias.

| Change | What Changed | Financial Rationale |
|--------|-------------|-------------------|
| Zero-Score Cliff Fix | `.fillna(0.0)` → `.fillna(np.nan)` with dynamic weight redistribution | Funds <110 weeks were double-penalized (P1=0 at 30% weight + confidence haircut). Now NaN pillars redistribute weight to available pillars. |
| Subsector Normalization | Z-score within subsector before ranking for P3/P4 metrics | Value funds have structurally different drawdown profiles than Focused funds; cross-sectional ranking without subsector awareness creates systematic bias |

**Sensitivity Analysis**: Rankings proved extremely robust to ±5% pillar weight perturbations: Variant A (100% top-10 overlap), Variant B (100%), Variant C (90%).

**Result**: HDFC Focused, HDFC Flexi Cap, ICICI Pru Focused as top 3. Walk-forward 24M persistence IC = +0.041.

---

## Advisor Thesis

### What This Algorithm Suite Believes
1. **Downside protection compounds**: Funds that lose less in corrections need less recovery to compound ahead. This is doubly important in our 24-month SIP+Hold scenario where the hold phase has zero DCA cushion.
2. **Consistency > extremes**: Rolling alpha hit-rates and peer-relative batting averages are more durable signals than point-to-point trailing returns.
3. **Capacity constrains alpha**: In Indian mid/small caps, AUM beyond ₹20-35k Cr (mid) or ₹15-25k Cr (small) degrades alpha through impact costs, style drift, and portfolio dilution.
4. **Manager skill should be shrunk**: James-Stein shrinkage toward peer median prevents extreme alpha estimates from dominating rankings.
5. **The hold phase is the true test**: Year 1 is forgiving (DCA cushions volatility). Year 2 exposes the full corpus. Funds must survive both.

### Competitor Observations (rankings only — not code)
- **GPT**: Strong in Small Cap (top XIRR), reasonable elsewhere
- **Gemini**: Competitive across categories, some overlap with Claude's top picks
- **Grok**: Strong in Total Market and Multi Asset; different fund selection philosophy

---

## Anti-Overfitting Guardrails

1. **No trailing CAGR in scoring weights**: Research confirms near-zero predictive power
2. **James-Stein shrinkage on all alpha estimates**: 55-86% shrinkage depending on history length
3. **Theory-driven weight priors (Small Cap)**: 70% of feature weights from financial research, not data
4. **No macro forecasts embedded (Multi Asset)**: Removed TARGET_MIX_2026 entirely
5. **Sensitivity analysis (Total Market)**: Confirmed rankings are robust to weight perturbations
6. **Regime matrix regularization (Mid Cap)**: 15% uniform prior prevents bull-market overfitting
7. **24m hybrid horizon everywhere**: Aligned all algorithms to the actual investment scenario, not a convenient 12m shortcut

---

## Deliverables Checklist

- [x] Data Upgrade: `duration='5y'` across all 4 algorithms
- [x] 4+ experiments per category (documented in `results/experiments/`)
- [x] 4 updated algorithms: `src/algorithms/{SECTOR}_Claude.py`
- [x] 4 final result CSVs: `results/{SECTOR}_Claude.csv`
- [x] Research notes: `results/experiments/claude_retro_research_notes.md`
- [x] Retrospective report: this document
- [x] Investment advisor thesis for top 3 per category
- [x] Competitor critique

---

## Investment Advisor Thesis — Top 3 Picks Per Category

### Mid Cap — Top 3

| Rank | Fund | Score | Key Strength |
|------|------|-------|-------------|
| 1 | **HDFC Mid Cap Fund** | 78.69 | Institutional anchor: beta 0.872, 100% SIP win rate (39 windows), best CDaR (-13.51%) |
| 2 | **WOC Mid Cap Fund** | 78.44 | Agile compounder: lowest volatility (14.85%), Sortino 0.984, shrunk alpha 4.42% |
| 3 | **Invesco India Midcap Fund** | 77.32 | High-floor maximizer: best Sortino (0.998), highest SIP p25 floor (17.48%) |

**Index risk context**: NIFTY Midcap 150 at +2.0σ above 10Y P/E median. All 3 picks have beta <0.95 and demonstrated hold-phase resilience through low CDaR and rapid recovery.

### Small Cap — Top 3

| Rank | Fund | Score | Key Strength |
|------|------|-------|-------------|
| 1 | **ITI Small Cap Fund** | 100.00 | Liquidity sweet-spot (₹3,603 Cr), recovery slope 44.76, 100% SIP consistency |
| 2 | **Invesco India Smallcap Fund** | 92.86 | Institutional grade: 5Y CAGR 20.46%, highest SIP p20 floor (13.15%), stress alpha 203 |
| 3 | **Bandhan Small Cap Fund** | 89.29 | Alpha powerhouse: SIP median 30.78%, SIP alpha 10.27%, 100% consistency |

**Index risk context**: NIFTY Smallcap 250 at +1.8σ with severe liquidity risk. All 3 operate below critical AUM thresholds and demonstrate stress-period resilience.

### Multi Asset — Top 3

| Rank | Fund | Score | Key Strength |
|------|------|-------|-------------|
| 1 | **Nippon India Multi Asset** | 78.46 | Complete 4-sleeve architecture (18.3% metals), Sortino 1.096, 100% SIP consistency |
| 2 | **DSP Multi Asset** | 70.29 | Ultra-low drawdown fortress: CDaR -5.11%, Pain Index -0.82, Calmar 2.355 |
| 3 | **Union Multi Asset** | 70.15 | Highest diversification (0.622), +3.95% strategic alpha, 19.9% precious metals |

**Index risk context**: Cross-asset — gold's counter-cyclical hedge via INR depreciation provides structural crisis protection. All 3 maintain true 4-sleeve diversification.

### Total Market — Top 3

| Rank | Fund | Score | Key Strength |
|------|------|-------|-------------|
| 1 | **HDFC Focused Fund** | 85.70 | Alpha fortress: Jensen's alpha 10.93%, beta 0.269, +16.08% bear correction excess |
| 2 | **HDFC Flexi Cap Fund** | 85.45 | Multi-cycle breadth: 100%/100% bench+peer hit rate, beta 0.267, max DD -13.08% |
| 3 | **ICICI Pru Focused Fund** | 83.60 | Downside floor maximizer: highest SIP p25 floor (16.26%), 100%/100% hit rates |

**Index risk context**: NIFTY 500 elevated. All 3 have ultra-low beta (~0.27) and 3-week recovery half-lives — designed to protect hold-phase capital.

---

## Competitor Critique

### Cross-Model Consensus (Universal Agreement = Strong Signal)

| Fund | Claude | GPT | Gemini | Grok |
|------|--------|-----|--------|------|
| WOC Mid Cap | #2 | #3 | #1 | #3 |
| ITI Small Cap | #1 | #1 | #2 | — |
| Nippon India Multi Asset | #1 | #1 | — | #1 |

These 3 funds represent the highest-conviction selections across all AI models.

### Key Divergences

**Bank of India Small Cap (Claude #15 vs Competitors Top 2)**: GPT #2, Gemini #1, Grok #1 all selected BOI. Claude rejected it because its 2Y down-capture (0.230) is 2.4x its up-capture (0.094) — it amplifies losses far more than gains. Competitors were rewarded by BOI's explosive momentum during the 4-month tracking window, but this is exactly the kind of unhedged beta that destroys hold-phase capital when liquidity reverses in small caps.

**Value Fund Overweight (GPT/Gemini/Grok in Total Market)**: All three competitors heavily loaded cyclical Value funds (Axis Value, HDFC Value, LIC Value, DSP Value). While value rallied in the short tracking window, deep-value stocks are prone to multiple-derating traps in late-cycle slowdowns. Claude's Focused/Flexi Cap picks retain mandate flexibility to rotate into quality during the hold phase.

**LIC MF Value Fund (Gemini Total Market #2)**: AUM of just ₹302 Cr — promoting a micro-cap fund without capacity risk controls is a fundamental screening failure.

### XIRR Performance Context (4 months — too short for conclusions)

| Category | Claude | GPT | Gemini | Grok | Median |
|----------|--------|-----|--------|------|--------|
| Mid Cap | **18.39%** | 16.70% | 18.29% | 14.84% | 13.98% |
| Small Cap | 47.82% | **54.44%** | 50.46% | 36.62% | 47.84% |
| Multi Asset | 3.80% | 6.31% | 3.74% | **6.78%** | 5.21% |
| Total Market | 10.61% | 16.58% | 16.94% | **19.79%** | 12.44% |

Claude leads Mid Cap, matches median in Small Cap, and trails in Multi Asset and Total Market. The trailing in Total Market reflects Claude's low-beta fortress construction (beta 0.27) — this is a deliberate trade-off: lower short-term returns in exchange for hold-phase capital protection. All models beat their respective benchmarks significantly.
