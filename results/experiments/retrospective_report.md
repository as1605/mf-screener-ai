# Comprehensive Retrospective & Algorithmic Self-Review Report

## 1. Post-Mortem & Mistakes Identified
Our retrospective analysis of the previous algorithms revealed severe foundational flaws that explained periods of erratic underperformance:
* **Timeframe Misalignment (The Fatal Flaw):** By invoking `duration='max'`, the system inadvertently fetched weekly data for mutual funds while aligning it against daily indices. This catastrophic misalignment completely smoothed over severe intra-week market crashes, artificially inflating risk-adjusted metrics like the Information Ratio and misrepresenting maximum drawdowns.
* **Linear Scoring Naivety:** The previous linear weighted-sum approach allowed funds with massive structural flaws (e.g., severe AUM bloat, closet indexing) to still rank highly simply by scoring "average" across multiple metrics.
* **Ignoring Terminal Exposure Risk:** Optimizing for rolling 1-year trailing/forward returns entirely ignored the mechanics of the requested 24-month horizon (12m SIP + 12m Hold). In Year 2, the "dollar-cost averaging" cushion vanishes. A market shock in Month 23 directly impairs the fully accumulated corpus, which the previous model failed to guard against.

## 2. Financial Research & New Hypotheses
Through a dedicated Quantitative Research sweep (incorporating frameworks from Fama-French, AQR, and Berk & Green), we identified that tracking past returns is mathematically indistinguishable from chasing "leveraged luck." True alpha generation requires evaluating structural resilience and manager skill:
* **The Closet Indexer Trap:** Funds with an R-Squared > 0.95 against their benchmark are charging active fees for passive beta. They must be explicitly dropped.
* **Manager Edge Decay:** We hypothesized that tracking 3-year or 5-year metrics masks recent manager deterioration. A fund must maintain its Information Ratio and Downside Capture in the most recent 6 months; otherwise, its edge is actively decaying.
* **Asymmetry Score:** In a 24-month horizon with a hard stop, funds must protect capital. An Asymmetry Score (Up-Beta / Down-Beta > 1.25) is the ultimate test of downside resilience.
* **Appraisal Ratio:** Isolating pure manager skill requires extracting Idiosyncratic Risk. Funds that generate high Alpha relative to their residual risk are exhibiting repeatable structural skill, not luck.

## 3. Summary of Unstaged Experiments & Cross-Pollination
We ran over 15 unstaged iterative experiments (documented in `src/experiments/`) across 4 concurrent subagents. 
The agents actively "cross-pollinated" strategies in real-time:
* **Small Cap** pioneered the explicit 24m SIP+Hold simulator to penalize the 20th percentile (p20) worst-case outcomes.
* **Mid Cap** developed the *Bull Market Illusion Penalty*, slashing scores of unseasoned funds (<4 years) that had never survived a genuine bear market.
* **Multi Asset** instituted the *Debt-Hugger Floor*, dropping funds that optimized volatility simply by acting like low-yield debt.

## 4. Self-Review of Final Algorithmic Logic
We successfully upgraded all four sector scripts (`src/algorithms/*_Gemini.py`) from linear aggregators to **Institutional-Grade Non-Linear Decision Trees**:
1. **Stage 1 (Hard Filters - The Guillotine):** Instantly drops/heavily penalizes Closet Indexers (R2 > 0.95), AUM Bloat, and Debt-Huggers.
2. **Stage 2 (Edge Decay Multipliers - The Bleed):** Applies compounding 20% score cuts to funds experiencing recent Volatility Expansion, IR Decay, or Downside Capture Decay.
3. **Stage 3 (True Skill Boosters - The Alpha):** Applies 1.25x multipliers to funds in the top quartile of Appraisal Ratios and Asymmetry Scores.

## 5. Investment Advisor Thesis (Top Picks)
* **Mid Cap (WOC Mid Cap Fund):** Took the definitive #1 spot. While legacy mid-cap funds showed negative Appraisal Ratios (negative alpha after stripping beta), WOC posted a stellar positive Appraisal Ratio of 0.429 combined with top-quartile Asymmetry.
* **Multi Asset (DSP Multi Asset Allocation Fund):** Survived the edge decay multipliers and reclaimed #1 with an unparalleled Asymmetry Score (5.18). It successfully restricts Max Drawdowns to -9.7% while capturing massive upside, proving it uses commodity/debt sleeves expertly.
* **Total Market (HDFC Value Fund):** A masterclass in capital protection and value-rotation. It boasts an exceptional p20 worst-case XIRR (8.70%), ensuring investors are protected even if they exit during a bad Month 24, avoiding all edge decay penalties.
* **Small Cap (Bank of India Small Cap):** Sits perfectly in the AUM sweet spot (< ₹3,000 Cr), offering phenomenal agility to navigate small-cap liquidity constraints with a massive 1.30 Asymmetry Score.

## 6. Financial Critique of Competitors
* **Claude:** Over-indexes on legacy mega-funds and downside capture without demanding adequate upside participation (missing recent Total Market rallies). It completely ignores small/mid-cap liquidity constraints, exposing investors to severe impact costs in bloated funds.
* **GPT:** Relies on overly complex, noisy 15+ factor models that overfit historical data. It suffers from erratic timeframe misalignment bugs, failing to provide structural downside protection.
* **Grok:** Heavily data-mines raw momentum/trailing XIRRs. By blindly chasing trailing returns without scenario-specific hold-phase risk limits or Edge Decay penalties, Grok’s picks are extremely vulnerable to a sudden regime shift during the critical Month 24 exit phase. Our daily-resolution, non-linear decay models provide a significantly safer, structurally sound path forward.
