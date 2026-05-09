# Multi Asset Allocation Strategies

This section covers **Multi Asset Allocation** funds (hybrid category): portfolios that blend equity, debt, and precious metals (gold/silver) in one mandate. Managers express skill through **asset mix and tactical shifts**, not only stock picking.

Proxies used in the algorithms (per task spec): **Nifty 500** for equity, **SBI Gold ETF Fund of Fund** (`M_SBIGL`) for gold, **ICICI Pru Silver ETF Fund of Fund** (`M_ICPVF`) for silver where history allows.

## Available Algorithms

### 1. [Claude — Returns-based style & outlook fit](./Multi%20Asset_Claude.md)

*Focus: Infer sleeve weights from NAVs (RBSA), strategic vs tactical value-add, downside-aware risk, and a forward-looking mix penalty/reward.*

**Top 5 Funds:**

1. Quant Multi Asset Allocation Fund (Score: 73.55)
2. Aditya Birla SL Multi Asset Allocation Fund (Score: 73.49)
3. SBI Multi Asset Allocation Fund (Score: 71.93)
4. DSP Multi Asset Allocation Fund (Score: 69.20)
5. Nippon India Multi Asset Allocation Fund (Score: 69.00)

**Key metrics:** Inferred equity/debt/gold/silver weights, Brinson-style strategic vs tactical decomposition, Sortino, CDaR, Calmar, pain index, outlook-fit vs a stated target mix, rolling SIP XIRR consistency.

[View Results CSV](../../results/Multi%20Asset_Claude.csv)

### 2. [GPT — Multi-asset feature blend with learned weights](./Multi%20Asset_GPT.md)

*Focus: Rolling SIP behaviour, regime and stress alphas, drawdown controls, exposure stability; feature weights learned from a look-ahead-safe historical panel and shrunk toward priors.*

**Top 5 Funds:**

1. SBI Multi Asset Allocation Fund (Score: 74.78)
2. DSP Multi Asset Allocation Fund (Score: 69.36)
3. WOC Multi Asset Allocation Fund (Score: 68.53)
4. Nippon India Multi Asset Allocation Fund (Score: 67.02)
5. Mirae Asset Multi Asset Allocation Fund (Score: 62.37)

**Key metrics:** SIP percentiles and hit rate, risk-on and mixed alphas, metal capture, stress alpha, Sortino/Calmar/Ulcer/CVaR controls, timing alignment, regime fit, durability.

[View Results CSV](../../results/Multi%20Asset_GPT.csv)

### 3. [Gemini — Cycle agility & metals participation](./Multi%20Asset_Gemini.md)

*Focus: Equity up-beta vs down-beta “agility”, upside capture on gold/silver rallies, Sortino, SIP return stability.*

**Top 5 Funds:**

1. SBI Multi Asset Allocation Fund (Score: 64.30)
2. ICICI Pru Multi-Asset Fund (Score: 53.44)
3. HDFC Multi-Asset Allocation Fund (Score: 48.83)
4. Tata Multi Asset Allocation Fund (Score: 48.72)
5. Nippon India Multi Asset Allocation Fund (Score: 46.55)

**Key metrics:** Agility score, precious-metals capture, 3Y Sortino, SIP stability.

[View Results CSV](../../results/Multi%20Asset_Gemini.csv)

## Composite ranking (cross-model)

Normalized scores from Claude, GPT, and Gemini are combined into a single peer ranking (see publish pipeline). **Illustrative top 5** from the latest composite sheet:

1. **SBI Multi Asset Allocation Fund**
2. **Nippon India Multi Asset Allocation Fund**
3. **DSP Multi Asset Allocation Fund**
4. **ICICI Pru Multi-Asset Fund**
5. **Aditya Birla SL Multi Asset Allocation Fund**

[View composite CSV](../../results/Multi%20Asset.csv) · [All Multi Asset results](../../results/)

## How the models differ

- **Claude** treats multi-asset alpha as **inferable sleeve weights** plus **strategic vs tactical** decomposition, and penalises portfolios that drift far from a **stated forward mix** (e.g. heavy metals after a strong rally).
- **GPT** emphasises **repeatable SIP outcomes** under many start dates, **stress-period behaviour**, and **stable allocation inference**, with weights partly **data-learned** (no peeking at future SIP windows).
- **Gemini** emphasises **how aggressively the fund participates in equity up-markets vs down-markets** and **capture of precious-metal upside**, alongside **downside-focused Sortino** and **SIP stability**.

**SBI Multi Asset Allocation Fund** scores highly across GPT and Gemini and leads the composite, while **Quant** tops Claude’s ranking—showing how different philosophies stress different sleeves and behaviours.
