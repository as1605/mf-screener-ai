# GPT scorer retrospective — 2026-09-13

## Purpose and constraint

The target is a **12-installment monthly SIP followed by a 12-month hold and one exit at Month 24**. This review modifies only the four `*_GPT.py` algorithms. It uses competing ranking files only as an outcome benchmark; no competing implementation was inspected.

This is a screening model, not investment advice. NAV-only data cannot observe holdings, fees, turnover, fund-manager tenure, stated scheme benchmark composition, or small-cap liquidity. Those inputs should be added when reliably available rather than approximated from NAV.

## Post-mortem

The existing GPT concepts had genuine strengths: category benchmark comparison, downside capture, drawdown/recovery, AUM/history awareness, and a cash-flow-aware XIRR objective. `results/ranks/xirr.csv` (17 overlapping snapshots, 2026-05-18 to 2026-09-07) showed mean realized excess XIRR versus the category median of +5.93pp Mid Cap (16/17 wins), +7.16pp Small Cap (16/17), +7.33pp Total Market (17/17), and only +0.09pp Multi Asset (11/17). In broad-equity stress snapshots, GPT beat the category median in all categories; in strong broad-market snapshots, Multi Asset lagged by -2.26pp.

The main mistake was material: Mid Cap, Small Cap, and Multi Asset redeemed after the 12th SIP rather than after the required hold year. They were therefore scoring a 12-month SIP, not the specified Month-24 payoff. The second mistake was allowing feature weights to be learned from short, overlapping rolling outcomes. That creates an attractive in-sample story but weak evidence of a future edge. A third error was saturated confidence: short-lived funds could receive near-full confidence and rank above long-established peers.

The 17 weekly snapshots are overlapping 24-month outcomes, so they are useful monitoring data—not 17 independent backtests. They are explicitly not used to fit production weights.

## Financial basis for the revised design

- **Appropriate benchmarks and risk-adjusted appraisal.** Fund results are assessed versus the specified category benchmark: Nifty Midcap 150, Nifty Smallcap 250, Nifty 500, and a provisional multi-asset blend. Information ratio, downside/upside capture, drawdown depth and time to recovery are complementary measures of active-risk quality, not substitutes for each other. [CFA Institute performance evaluation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/portfolio-performance-evaluation)
- **No performance chasing.** Research finds that much apparent mutual-fund persistence reflects factor exposure, costs, and the persistence of poor performance; later samples show weak persistence. The model therefore emphasizes repeatable benchmark-relative behavior and lower-tail outcomes rather than raw trailing CAGR or the highest rolling XIRR. [Carhart (1997)](https://doi.org/10.1111/j.1540-6261.1997.tb03808.x), [Choi & Zhao](https://www.nber.org/papers/w26707)
- **Skill is possible but capacity matters.** Economic models explain why manager skill and return persistence can differ as flows erode opportunities. NAV alone cannot quantify dollar value added, so AUM is used only as a modest confidence check, not proof of quality. [Berk & Green](https://www.nber.org/papers/w9275), [Berk & van Binsbergen](https://www.nber.org/papers/w18184)
- **Index-risk rationale.** NSE describes greater one-year return/risk for the Midcap 150 than the Nifty 100 and weaker performance in weak markets, supporting explicit correction-defense tests. Smallcap 250 covers the lower-capitalisation portion of the Nifty 500 and warrants the tightest downside/liquidity scrutiny. [Nifty Midcap 150](https://www.niftyindices.com/docs/default-source/indices/nifty-midcap-150/nifty-midcap-150-whitepaper_2021.pdf?sfvrsn=15d66e35_4), [Nifty Smallcap 250](https://www.niftyindices.com/indices/equity/broad-based-indices/niftysmallcap250)
- **Multi-asset caveat.** AMFI’s benchmark policy requires an appropriate benchmark for every meaningful asset class. A generic equity/gold/silver blend is an interim NAV-only proxy and should be replaced by each scheme’s declared composite benchmark. [AMFI policy](https://www.amfiindia.com/uploads/Policyframework_337a96d520.pdf)

## Final architecture

1. Daily five-year fund NAV is requested with `duration="5y"`; daily data is used for drawdowns and exact-date cash-flow matching. Index data is aligned before relative metrics are computed.
2. Every rolling scenario makes 12 monthly purchases, adds 12 zero cash-flow hold months, then sells at Month 24. Scores use distributional evidence (median, lower-quartile, benchmark hit rate) rather than one realized return.
3. Scores remain category-specific but use financially motivated **fixed priors**. The former learned weights are not used in production because the effective sample of non-overlapping 24-month outcomes is small.
4. A fund must have at least three years of history and 12 completed Month-24 windows to be `evidence_qualified`. Unqualified funds are disclosed and heavily haircutted; they cannot outrank qualified funds.
5. Mid/Small/Total reward active consistency only when it is accompanied by downside capture, drawdown/recovery, and benchmark-relative controls. Multi Asset prioritizes realized all-regime resilience but needs declared-composite benchmarks before making strong claims about tactical allocation skill.

## Experiments

Five unstaged cycles are recorded in `experiment_log.csv`: baseline retrospective, horizon correction, fixed-prior comparison, evidence-gating, and the Multi Asset benchmark review. The decision rule was economic plausibility first, then as-of-date diagnostics. No experiment selected a formula by maximizing historical XIRR.

## Final advisor-style reading of the current top three

The rankings are a shortlist for due diligence, not a recommendation to buy. Confirm the direct/regular plan, expense ratio, exit load, portfolio overlap, declared benchmark, and current scheme documents before investing.

- **Mid Cap:** HDFC Mid Cap, Invesco India Midcap, and WOC Mid Cap lead because their completed Month-24 vintages, benchmark-relative behavior and downside/recovery profiles score well. WOC has only about four years of data and should receive more scrutiny than the other two despite passing the minimum evidence gate. Midcap-150 risk remains substantial in a valuation/liquidity correction.
- **Small Cap:** ITI Small Cap, Bank of India Small Cap, and Union Small Cap are selected from the evidence-qualified set rather than the short-history high-return cohort. Smallcap-250 liquidity and tail-risk can dominate a two-year result; the ranking does not observe each portfolio’s stressed-liquidation capacity. AMFI’s published stress-test disclosures are an essential next diligence input. [AMFI risk parameters](https://www.amfiindia.com/risk-parameters)
- **Multi Asset:** Nippon India Multi Asset Allocation, SBI Multi Asset Allocation, and WOC Multi Asset Allocation rise on balanced rolling scenario outcomes and stress/drawdown evidence. The first two have nearly five years of daily history; WOC has about 3.3 years and deserves a higher uncertainty discount in a real allocation decision. GPT’s weaker historic Multi Asset result and a generic proxy benchmark make these especially provisional.
- **Total Market:** Axis Value, ICICI Pru Focused, and Mahindra Manulife Multi Cap score well on Nifty-500-relative scenario outcomes, recovery/resilience, and factor-adjusted consistency. Axis Value is just under five years of history, so the evidence gate prevents any sub-three-year name from outranking it but does not claim a complete market-cycle record. The Nifty 500 is the task comparator, although a fund’s stated benchmark is preferable for final due diligence. [Nifty 500](https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500)

## Competitor output critique

The comparison is restricted to rankings and `xirr.csv`. GPT outperformed all peer picks in Small Cap over the tracked snapshots, but trailed Claude in Mid and Multi and trailed Gemini and Grok in Total Market. That is evidence to be humble, not to copy picks or tune GPT toward their historical winners. The lack of an independent sample means the more defensible response is the horizon correction, benchmark-relative risk gates, and uncertainty control implemented here.

## Remaining work before calling this institutional-grade

1. Add point-in-time scheme benchmark, expense, turnover, holdings, manager-change and AMFI liquidity/stress-test data. Do not infer them from NAV.
2. Use non-overlapping quarterly or annual formation dates and lock rules before a later holdout period.
3. Add portfolio-level AMC and underlying-holding concentration constraints when turning a ranking into a multi-fund allocation.
4. Replace Multi Asset’s generic proxy with each scheme’s disclosed composite benchmark and minimum sleeve weights.
