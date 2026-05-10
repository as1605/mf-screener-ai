# Algorithms

This project uses a "vibecoding" approach where different AI models are tasked with creating strategies to score mutual funds. The core idea is to leverage the diverse reasoning capabilities of different LLMs to identify high-potential funds.

## Concept

Instead of relying on a single fixed strategy, we ask multiple AI models to act as quantitative analysts. We provide them with the data and the goal (ranking funds for a stated horizon—typically a hybrid SIP-then-hold path such as 12 months of SIP plus 12 months hold and exit), and they design the scoring logic.

## Models Used

*   **Claude**: For Small/Mid/Total Market funds, focuses on adaptive multi-horizon conviction with path quality and regime adaptability (skill persistence, tail risk). For **Multi Asset**, uses returns-based style analysis on NAVs to infer sleeve weights, Brinson-style strategic vs tactical decomposition, downside metrics (Sortino, CDaR, Calmar), and an outlook-fit pillar versus a target mix.
*   **Gemini**: For Small/Mid/Total Market funds, uses multi-factor regression to isolate "pure alpha" from market-cap exposure (Omega Ratio, downside resistance, consistency). For **Multi Asset**, emphasises equity cycle agility (up vs down beta), precious-metals upside capture, Sortino, and SIP return stability.
*   **GPT** (Small Cap, Mid Cap, Multi Asset, Total Market): Blends NAV-derived SIP alignment, benchmark capture and recovery metrics, and AUM-aware confidence with feature weights learned from look-ahead-safe historical SIP panels where applicable (Spearman vs forward SIP alpha), mixed with a fixed research prior. **Mid Cap** extends the family with a richer factor set (information ratio, downside beta, correction defence, Omega, Ulcer, CDaR, style purity). **Multi Asset** adds sleeve/regime features (metals capture, stress alphas, exposure stability). **Total Market** emphasises rolling hybrid SIP-hold XIRR (median and tails vs benchmark and peers), recovery participation, hold-phase drawdown control, multi-factor alpha versus large/mid/small and broad indices, and history/capacity confidence.
*   **Grok** (Small Cap, Mid Cap, Multi Asset, Total Market): Fixed **theory-weighted** composites on **robust cross-sectional z-scores** (median/MAD), mapped through a **normal CDF** for score spread where that pattern applies. **Small Cap** stresses rebound capture, SIP stability, recovery after drawdowns, vol-adjusted alpha, downside resilience, and drawdown avoidance. **Mid Cap** adds nine-factor hybrid vs mid benchmark plus **large/small** paths for style purity, drift, and timing. **Multi Asset** aligns fund NAV to **gold**, **silver**, and **Nifty 500** for cycle capture, metals beta, equity participation, drawdown pain, and allocation stability. **Total Market** combines hybrid scenario XIRR statistics, information ratio and size-adjusted alpha, recovery timing, downside capture, sector theme tilt, and conviction-style concentration.

## Disclaimer

These algorithms are AI-generated and experimental. They rely on historical data which is not a guarantee of future performance. The scores should be used as a starting point for research, not as the sole basis for investment decisions.

## Navigation

*   [Small Cap Strategy Overview](algorithms/Small%20Cap.md)
    *   [Claude Strategy](algorithms/Small%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Small%20Cap_Gemini.md)
    *   [GPT Strategy](algorithms/Small%20Cap_GPT.md)
    *   [Grok Strategy](algorithms/Small%20Cap_Grok.md)

*   [Mid Cap Strategy Overview](algorithms/Mid%20Cap.md)
    *   [Claude Strategy](algorithms/Mid%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Mid%20Cap_Gemini.md)
    *   [GPT Strategy](algorithms/Mid%20Cap_GPT.md)
    *   [Grok Strategy](algorithms/Mid%20Cap_Grok.md)

*   [Total Market Strategy Overview](algorithms/Total%20Market.md)
    *   [Claude Strategy](algorithms/Total%20Market_Claude.md)
    *   [Gemini Strategy](algorithms/Total%20Market_Gemini.md)
    *   [GPT Strategy](algorithms/Total%20Market_GPT.md)
    *   [Grok Strategy](algorithms/Total%20Market_Grok.md)

*   [Multi Asset Strategy Overview](algorithms/Multi%20Asset.md) ⭐ *New*
    *   [Claude Strategy](algorithms/Multi%20Asset_Claude.md)
    *   [Gemini Strategy](algorithms/Multi%20Asset_Gemini.md)
    *   [GPT Strategy](algorithms/Multi%20Asset_GPT.md)
    *   [Grok Strategy](algorithms/Multi%20Asset_Grok.md)
