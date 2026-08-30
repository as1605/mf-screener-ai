# Methodology: The LLM Arena for Mutual Funds

MF Screener AI takes a unique approach to quantitative mutual fund screening. Instead of relying on a single, fixed set of rules designed by one human, it acts as an **LLM Arena**. We provide raw historical NAV data to four distinct AI models (Claude, Gemini, GPT, and Grok) and task them with independently designing the best quantitative scoring strategy for Indian mutual funds.

## Core Concept

We supply the models with TickerTape data (daily/weekly NAVs) and the goal of identifying high-potential, resilient funds for long-term investors (e.g., heavily weighting SIP outcomes). Each model generates a Python script containing its unique scoring logic. The final ranking is a composite of these four distinct "expert opinions."

## The AI Quantitative Strategies

*   **Claude**: Focuses on adaptive multi-horizon conviction. It prioritizes path quality, regime adaptability (skill persistence across bull/bear markets), and tail risk management. For Multi Asset, it leans into returns-based style analysis to infer sleeve weights and downside metrics like CDaR (Conditional Drawdown at Risk).
*   **Gemini**: Emphasizes isolating "pure alpha." It uses multi-factor regressions to separate manager skill from raw market-cap exposure, heavily weighting the Omega Ratio, downside resistance, and cycle agility.
*   **GPT**: Blends NAV-derived SIP alignment (simulating real-world retail investor outcomes via rolling SIP XIRR) with benchmark capture metrics. It focuses on recovery strength, hold-phase drawdown control, and multi-factor alpha versus broad indices.
*   **Grok**: Relies on robust cross-sectional statistical scoring. It uses theory-weighted composites and z-scores (median/MAD) mapped through a normal CDF. It stresses rebound capture, downside resilience, and timing purity.

## Normalization and Compilation

Because each model scores funds differently (some output scores from 0-100, others from -2 to +2), we run a compilation step (`src/compile_ranks.py`).
1.  **Z-Score Normalization**: Each model's raw scores are normalized across the peer group.
2.  **Composite Score**: The normalized scores from all four models are aggregated to create a single, unified "Final Score."
3.  **Ranking**: Funds are ranked strictly based on this composite score.

## Focus on Verifiability: SIP XIRR

A major differentiator in our methodology is the emphasis on **SIP (Systematic Investment Plan) returns**. Traditional screeners often rely solely on point-to-point CAGR, which can be highly sensitive to the start and end dates and doesn't reflect how most retail investors actually invest.

The AI strategies are specifically prompted to calculate and prioritize **Historical SIP XIRR**. This ensures that the highly-ranked funds have demonstrated consistent, verifiable performance for periodic investors through various market cycles.

## Disclaimers & Bias

*   **Data Quality**: Results are bound by the accuracy of the historical NAV data fetched via TickerTape.
*   **Look-Ahead Bias**: While models are prompted to avoid look-ahead bias (especially in SIP panel construction), some implicit bias may remain in the AI's feature selection.
*   **Not Financial Advice**: This methodology is experimental and educational. Always cross-verify metrics and consult a financial advisor before investing.
