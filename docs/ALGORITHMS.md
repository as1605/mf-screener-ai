# Algorithms

This project uses a "vibecoding" approach where different AI models are tasked with creating strategies to score mutual funds. The core idea is to leverage the diverse reasoning capabilities of different LLMs to identify high-potential funds.

## Concept

Instead of relying on a single fixed strategy, we ask multiple AI models to act as quantitative analysts. We provide them with the data and the goal (predicting 1-year returns), and they design the scoring logic.

## Models Used

*   **Claude**: For Small/Mid/Total Market funds, focuses on adaptive multi-horizon conviction with path quality and regime adaptability (skill persistence, tail risk). For **Multi Asset**, uses returns-based style analysis on NAVs to infer sleeve weights, Brinson-style strategic vs tactical decomposition, downside metrics (Sortino, CDaR, Calmar), and an outlook-fit pillar versus a target mix.
*   **Gemini**: For Small/Mid/Total Market funds, uses multi-factor regression to isolate "pure alpha" from market-cap exposure (Omega Ratio, downside resistance, consistency). For **Multi Asset**, emphasises equity cycle agility (up vs down beta), precious-metals upside capture, Sortino, and SIP return stability.
*   **GPT** (Small Cap, Multi Asset): Blends NAV-derived SIP alignment, benchmark capture and recovery metrics, and AUM-aware confidence with feature weights learned from a look-ahead-safe historical SIP panel (Spearman vs forward SIP alpha), mixed with a fixed research prior. The Multi Asset variant adds sleeve/regime features (metals capture, stress alphas, exposure stability).
*   **Codex** (Mid Cap, Total Market): Uses walk-forward validation and weight tuning to optimize for forward returns. Implements subsector-aware rank blending to reduce style overconcentration.

## Disclaimer

These algorithms are AI-generated and experimental. They rely on historical data which is not a guarantee of future performance. The scores should be used as a starting point for research, not as the sole basis for investment decisions.

## Navigation

*   [Small Cap Strategy Overview](algorithms/Small%20Cap.md)
    *   [Claude Strategy](algorithms/Small%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Small%20Cap_Gemini.md)
    *   [GPT Strategy](algorithms/Small%20Cap_GPT.md)

*   [Mid Cap Strategy Overview](algorithms/Mid%20Cap.md)
    *   [Claude Strategy](algorithms/Mid%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Mid%20Cap_Gemini.md)
    *   [Codex Strategy](algorithms/Mid%20Cap_Codex.md)

*   [Total Market Strategy Overview](algorithms/Total%20Market.md)
    *   [Claude Strategy](algorithms/Total%20Market_Claude.md)
    *   [Gemini Strategy](algorithms/Total%20Market_Gemini.md)
    *   [Codex Strategy](algorithms/Total%20Market_Codex.md)

*   [Multi Asset Strategy Overview](algorithms/Multi%20Asset.md) ⭐ *New*
    *   [Claude Strategy](algorithms/Multi%20Asset_Claude.md)
    *   [Gemini Strategy](algorithms/Multi%20Asset_Gemini.md)
    *   [GPT Strategy](algorithms/Multi%20Asset_GPT.md)
