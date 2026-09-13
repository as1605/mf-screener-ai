# Retrospective and Algorithmic Improvements

{model}={Grok}
Categories: `Mid Cap`, `Small Cap`, `Multi Asset`, `Total Market`

This task guides a disciplined retrospective analysis, financial re-grounding, and iterative improvement of our Mutual Fund scoring algorithms.

Your goal is to evaluate your previous algorithm (`src/algorithms/{SECTOR}_{model}.py`) and generated rankings (`results/{SECTOR}_{model}.csv`), diagnose shortcomings, and elevate the scoring system into an institutional-grade screener. You can check previous git branches (`date/YYYY-MM-DD`) or commit history for earlier baseline results.

> **Rule on Model Isolation:**
> You must inspect and modify **only your own script** (`src/algorithms/*_{model}.py`). Do **not** inspect or copy implementation code from competing model scripts (`Claude`, `GPT`, `Gemini`). You may only inspect competitor *rankings and recommendations* in `results/{SECTOR}_{other_model}.csv` to benchmark and critique their picks.

---

## 1. The Investment Scenario & Horizon Rationale

Review the investment scenario established in `tasks/2_algorithm.md`:
- **Scenario:** Invest in a monthly SIP for 1 year (12 installments), followed by a 1-year hold (12 months with zero additional capital infusions), then exiting at Month 24.
- **Economic & Tax Rationale for this Horizon:**
  - **LTCG Tax Optimization:** Under Indian tax law, equity mutual fund units require a minimum holding period of 1 year to qualify for Long Term Capital Gains (LTCG) tax treatment (avoiding higher Short Term Capital Gains tax). A 1-year hold following a 1-year SIP guarantees that even the final SIP installment (bought at Month 12) completes at least 1 full year of seasoning by Month 24 (1 year minimum, 2 years preferred for optimal compounding and tax efficiency).
  - **Disciplined Rotation Horizon:** A 2-year total cycle provides a realistic, practical timeframe for tactical portfolio review and capital rotation without excessive turnover, exit load penalties, or transaction drag.
  - **Accumulation vs. Hold Dynamics:** During the second year (months 13–24), the investor loses the dollar-cost averaging cushion of ongoing SIPs. A severe market downturn during this hold phase strikes the full accumulated corpus.
- **Optimization Metric & Indicators (Model's Discretion):**
  - **The model decides its own optimization metric and indicators.** Rather than blindly optimizing a rigid predefined 24-month calculation function, you should select specific financial indicators using the NAV data (e.g., downside-penalized return, volatility contraction, custom risk-adjusted metrics) that ensure the fund will perform well consistently for this scenario.

---

## 2. Trend & Performance Tracking: `results/ranks/xirr.csv`

For easier analysis and a quick look at realized trends across dates and market regimes, use `results/ranks/xirr.csv`:
- **What it tracks:** Weekly snapshots of realized 24-month hybrid XIRRs across:
  - **Benchmark Indices:** `NIFTY 50`, `NIFTY 500`, `NIFTY Midcap 150`, `NIFTY Smallcap 250`, `Gold`
  - **Category Medians:** `Mid Cap`, `Multi Asset`, `Small Cap`, `Total Market`
  - **Model Picks:** Realized weekly returns for picks by each agent (`{model}`, `Claude`, `GPT`, `Gemini`).
- **How to leverage it:**
  - **Quick Regime Diagnostics:** Identify weekly periods where markets rallied or sharply corrected to observe how each sector and index behaved under stress.
  - **Trend Analysis:** Quickly assess whether `{model}`'s picks outperformed or lagged benchmark indices and category medians over time.
  - **Competitor Benchmarking:** Observe the performance spread between your model and competing models without reading their code.

---

## 3. Data Resolution: Daily NAV (`duration='5y'`) over Weekly (`'max'`)

A critical data upgrade for this retrospective:
- **Switch from `'max'` to `'5y'`:** When retrieving historical NAV series via `MfDataProvider`, query with `duration='5y'` (e.g., `provider.get_mf_chart(mf_id, duration='5y')` or `provider.fetch_mf_chart(mf_id, duration='5y')`).
- **Why this matters:**
  - `duration='max'` aggregates historical NAV to **weekly** samples.
  - `duration='5y'` provides granular **daily** NAV data over a 5-year window.
- **Financial Benefit:** Daily series enable high-fidelity analysis of rolling drawdowns, volatility clustering, day-to-day recovery dynamics, accurate downside risk profiling, and precise cashflow matching on exact SIP dates.

---

## 4. Core Philosophy: Financial Grounding vs. Data-Mining Biases

> ### ⚠️ Crucial Principle: Learn from Past Performance Without Overfitting to Data
> **Past performance is not indicative of future returns.** A naive algorithm that merely curve-fits historical NAV curves or tunes weights to maximize trailing backtest numbers will inevitably fail out-of-sample. Historical results are an input to study, not an objective function to blindly overfit.

You must ground your algorithmic design in actual **finance knowledge, market cycle awareness, and risk management principles**:

### Learn from Past Performance Without Biasing to Data
- Historical data should be analyzed to understand *how and why* funds behaved across different market conditions—not to cherry-pick parameters that flatter past numbers.
- Distinguish between **luck (beta/liquidity tailwinds)** and **skill (genuine alpha and risk control)**. Did a fund outperform because of superior portfolio construction, or simply by holding high-beta momentum during an aggressive bull run?
- Avoid recency bias and survivorship bias. Strong returns during a one-way liquidity expansion do not guarantee resilience when liquidity tightens or markets enter a corrective/sideways regime.

### Ground Decisions on Financial Knowledge (Model's Discretion)
- **Do not blindly follow rigid templates.** As the quantitative researcher, **you decide** which financial concepts, metrics, and risk factors make economic sense for each sector.
- You must research and justify your methodology from first principles:
  - How do you measure true risk-adjusted performance across full market cycles?
  - How do you assess downside resilience, drawdown duration, and recovery characteristics?
  - What structural fund factors matter (e.g., AUM scale and liquidity constraints in smaller-cap spaces, expense ratio drag, portfolio concentration)?
- Every metric, factor weight, or rule you introduce must have a coherent financial rationale that you can clearly articulate and defend.

### Explicitly Consider Underlying Index Risk
Funds do not operate in a vacuum; their performance and risk are deeply tied to the regime and volatility of their underlying benchmark index:
- **Mid Cap Benchmark:** `NIFTY Midcap 150` (`.NIMI150`)
- **Small Cap Benchmark:** `NIFTY Smallcap 250` (`.NISM250`)
- **Total Market Benchmark:** `NIFTY 500` (`.NIFTY500`)
- **Multi Asset Benchmark:** Cross-asset blend (Equity + Debt + Commodities like Gold `GBES`)

**Incorporate index risk into your research:**
- Examine how funds perform relative to systemic swings, corrections, and volatility in their respective index.
- Consider the risks of the underlying index itself (e.g., small-cap liquidity freeze, mid-cap valuation stretch, broad-market consolidation).
- Determine how the fund navigates index drawdowns: Does the fund amplify index losses, hug the index passively while charging active fees, or demonstrate independent downside resilience?

---

## 5. Execution Workflow

You will analyze and enhance the algorithms for all four categories:
1. `Mid Cap`
2. `Small Cap`
3. `Multi Asset`
4. `Total Market`

Follow this 4-step workflow:

### Step 1: Retrospective & Code Post-Mortem
- **First, conduct a comprehensive review** of your existing script (`src/algorithms/{SECTOR}_{model}.py`), previous outputs in `results/{SECTOR}_{model}.csv`, and historical performance trends in `results/ranks/xirr.csv`.
- **Compare and Contrast:** Understand the current situation by comparing your performance against benchmark movements, category medians, and competitor picks.
- **Identify Wins and Losses:** Explicitly point out the current *great decisions* (factors/logic that genuinely worked) and *big mistakes* (flawed assumptions, data-mining biases, or missing risk controls) made by your previous algorithm.
- Upgrade data ingestion to `duration='5y'` for daily NAV data.
- Diagnose mathematical errors, calculation bugs, lookahead biases, or flawed assumptions (e.g., misaligned dates with index charts, improper annualization).
- Critically evaluate which historical signals had genuine predictive merit versus which were spurious noise.

### Step 2: Financial Research & Hypothesis Formulation
- Research the macroeconomic environment, sector dynamics, and index risk characteristics in Indian markets.
- Formulate clear, testable, financially grounded hypotheses for each category before coding.
- Define what financial mechanisms you want to test and why they are expected to improve forward 24-month outcomes.

### Step 3: Running Unstaged Iterative Experiments
- Run **at least 4 iterative cycles of experiments** per category to test and refine your hypotheses.
- **Keep experimental iterations unstaged:** Save intermediate test outputs, backtest logs, or comparison files under `results/experiments/` without staging them in git prematurely.
- Stress-test your logic across sub-periods (e.g., index corrections, rallies, consolidation phases) to verify that the scoring behavior is robust across different market regimes.

### Step 4: Self-Review, Code Implementation & Advisor Thesis
- **Implement Code Changes:** Update `src/algorithms/{SECTOR}_{model}.py` with your finalized, production-ready scoring algorithm and output final rankings to `results/{SECTOR}_{model}.csv`.
- **Self-Review:** Rigorously evaluate your final model:
  - Did the changes genuinely address past shortcomings?
  - Are top funds selected due to sound financial characteristics or data quirks?
- **Investment Advisor Thesis for Top 3 Picks:** For each category, provide an advisor-level rationale explaining why the top 3 funds are suited for the 24-month SIP-to-hold horizon, accounting for their underlying index risk.
- **Competitor Critique:** Review competitor rankings in `results/{SECTOR}_{other_model}.csv` and trend performance in `results/ranks/xirr.csv`. Critique their selections from a financial standpoint and explain why your grounded approach provides a more defensible forward outlook.

---

## 6. Deliverables Checklist

- [ ] **Data Upgrade:** Daily NAV data (`duration='5y'`) integrated across all algorithms.
- [ ] **Unstaged Experiments:** Iterative experiment logs and test runs documented in `results/experiments/` (kept unstaged).
- [ ] **Code Updates:** Clean, production-ready algorithm implementations in `src/algorithms/{SECTOR}_{model}.py` for all 4 sectors.
- [ ] **Final Results:** Generated ranking sheets in `results/{SECTOR}_{model}.csv`.
- [ ] **Retrospective & Self-Review Report:** *(Note: Do not stage your report, it is for self-review only. You may keep it in `results/experiments/` or as an unstaged artifact.)*
  1. Post-mortem of previous script and mistakes identified (leveraging `results/ranks/xirr.csv` trend analysis).
  2. Financial research, economic hypotheses, and index risk analysis.
  3. Summary of experimental iterations and findings.
  4. Self-review of final algorithmic logic.
  5. Investment advisor thesis for Top 3 funds per category.
  6. Financial critique of competitor model recommendations.
