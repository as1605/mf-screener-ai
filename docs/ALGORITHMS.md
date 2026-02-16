# Algorithms

This project uses a "vibecoding" approach where different AI models are tasked with creating strategies to score mutual funds. The core idea is to leverage the diverse reasoning capabilities of different LLMs to identify high-potential funds.

## Concept

Instead of relying on a single fixed strategy, we ask multiple AI models (Claude, Gemini, Codex) to act as quantitative analysts. We provide them with the data and the goal (predicting 1-year returns), and they design the scoring logic.

## Models Used

*   **Claude**: Focuses on a balanced approach, weighing Alpha, Downside Protection, and Consistency.
*   **Gemini**: Emphasizes advanced statistical metrics like Omega Ratio, Hurst Exponent, and Upside Potential.
*   **Codex**: Uses a validation-driven approach, tuning weights based on historical backtesting to optimize for forward returns.

## Disclaimer

These algorithms are AI-generated and experimental. They rely on historical data which is not a guarantee of future performance. The scores should be used as a starting point for research, not as the sole basis for investment decisions.

## Navigation

*   [Small Cap Strategy Overview](algorithms/Small%20Cap.md)
    *   [Claude Strategy](algorithms/Small%20Cap_Claude.md)
    *   [Gemini Strategy](algorithms/Small%20Cap_Gemini.md)
    *   [Codex Strategy](algorithms/Small%20Cap_Codex.md)
