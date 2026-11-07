# Experiments

This is for ad-hoc analysis of data

## SIF

```bash
python src/experiments/sif.py
```

This populates the results:

- [NAV](../results/experiments/sif_nav.csv)
- [3 Month Returns](../results/experiments/sif_3m.csv)
- [6 Month Returns](../results/experiments/sif_6m.csv)

## Point-to-Point Returns

```bash
python src/experiments/returns.py
python src/experiments/returns.py --start 2026-10-23 --end 2026-11-06
```

Computes point-to-point returns for mutual funds sorted descending by performance over a specified period. Parallelized with 16 workers using `ThreadPoolExecutor`.

- Enriches each fund with metadata: category, sector, AUM.
- Joins `mf-screener-ai_rank` from the last available weekly ranking on or before `--start` across universes (`Mid Cap`, `Small Cap`, `Multi Asset`, `Total Market`).
- If `--start` is too far back (prior to available rankings), `mf-screener-ai_rank` remains empty.
- Automatically saves results to `results/experiments/returns_<start>_to_<end>.csv`.
