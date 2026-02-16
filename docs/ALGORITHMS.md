# Algorithms

This project uses a "vibecoding" approach where different AI models are tasked with creating strategies to score mutual funds. The core idea is to leverage the diverse reasoning capabilities of different LLMs to identify high-potential funds.

## Concept

Instead of relying on a single fixed strategy, we ask multiple AI models (Claude, Gemini, Codex) to act as quantitative analysts. We provide them with the data and the goal (predicting 1-year returns), and they design the scoring logic.

## Models Used

*   **Claude**: Focuses on adaptive multi-horizon conviction with path quality and regime adaptability analysis. Emphasizes skill persistence and tail risk management.
*   **Gemini**: Uses multi-factor regression to isolate "pure alpha" from market-cap exposure. Emphasizes Omega Ratio, downside resistance, and consistency of skill.
*   **Codex**: Uses walk-forward validation and weight tuning to optimize for forward returns. Implements subsector-aware rank blending to reduce style overconcentration.

## Disclaimer

These algorithms are AI-generated and experimental. They rely on historical data which is not a guarantee of future performance. The scores should be used as a starting point for research, not as the sole basis for investment decisions.

## Navigation

*   [Small Cap Strategy Overview](algorithms/Small%20Cap.md)
    *   [Claude Strategy](algorithms/Small%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Small%20Cap_Gemini.md)
    *   [Codex Strategy](algorithms/Small%20Cap_Codex.md)

*   [Mid Cap Strategy Overview](algorithms/Mid%20Cap.md)
    *   [Claude Strategy](algorithms/Mid%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Mid%20Cap_Gemini.md)
    *   [Codex Strategy](algorithms/Mid%20Cap_Codex.md)

*   [Total Market Strategy Overview](algorithms/Total%20Market.md) ⭐ *New*
    *   [Claude Strategy](algorithms/Total%20Market_Claude.md)
    *   [Gemini Strategy](algorithms/Total%20Market_Gemini.md)
    *   [Codex Strategy](algorithms/Total%20Market_Codex.md)
