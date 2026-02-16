"""
Compile results from multiple models for a sector into a single CSV.
Reads results/{SECTOR}_{model}.csv, joins by mfId, computes final_score and total_rank.
"""
from pathlib import Path
import pandas as pd
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def discover_sectors() -> list[str]:
    """Find all sectors that have at least one model result file."""
    sectors = set()
    for f in RESULTS_DIR.glob("*_*.csv"):
        name = f.stem  # e.g. "Small Cap_Claude"
        if "_" in name:
            sector = name.rsplit("_", 1)[0]
            sectors.add(sector)
    return sorted(sectors)


def get_model_files(sector: str) -> list[tuple[str, Path]]:
    """Return list of (model_name, path) for results/{sector}_{model}.csv."""
    out = []
    prefix = f"{sector}_"
    for f in RESULTS_DIR.glob(f"{prefix}*.csv"):
        model = f.stem[len(prefix) :]
        out.append((model, f))
    return sorted(out, key=lambda x: x[0])


def compile_sector(sector: str, results_dir: Path | None = None) -> pd.DataFrame:
    """
    Read all model CSVs for sector, join by mfId, compute final_score (stddev-normalized
    per model then averaged), total_rank, avg_data_days, avg_cagr_3y, and score_{model}.
    """
    results_dir = results_dir or RESULTS_DIR
    model_files = get_model_files(sector)
    if not model_files:
        raise FileNotFoundError(f"No result files found for sector {sector!r} in {results_dir}")

    models = [m for m, _ in model_files]
    dfs = []
    for model, path in model_files:
        df = pd.read_csv(path)
        required = {"mfId", "name", "score", "data_days", "cagr_3y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        # Keep one name column (from first model); we'll take name from first merge
        sub = df[["mfId", "score", "data_days", "cagr_3y"]].copy()
        sub = sub.rename(columns={
            "score": f"score_{model}",
            "data_days": f"data_days_{model}",
            "cagr_3y": f"cagr_3y_{model}",
        })
        # Coerce numeric
        for c in [f"score_{model}", f"data_days_{model}", f"cagr_3y_{model}"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        dfs.append((model, sub, df[["mfId", "name"]].drop_duplicates("mfId")))

    # Outer join on mfId
    base = dfs[0][1]
    names = dfs[0][2]
    for model, sub, _ in dfs[1:]:
        base = base.merge(sub, on="mfId", how="outer")
    merged = base
    merged = names.merge(merged, on="mfId", how="right")
    # Fill name from other model tables if missing
    for _, _, name_df in dfs[1:]:
        merged = merged.merge(
            name_df.rename(columns={"name": "_name_alt"}),
            on="mfId",
            how="left",
        )
        merged["name"] = merged["name"].fillna(merged["_name_alt"])
        merged = merged.drop(columns=["_name_alt"])

    # Normalize each model's score to 0-1 using mean and std (then min-max of z-scores)
    # Missing (not ranked) is treated as 0 so that final_score is averaged over ALL models.
    score_cols = [f"score_{m}" for m in models]
    n_models = len(models)
    norm_scores = []
    for m in models:
        col = f"score_{m}"
        s = merged[col].astype(float)
        valid = s.notna()
        mu = s[valid].mean() if valid.any() else np.nan
        std = s[valid].std() if valid.sum() > 1 else 0.0
        if std == 0 or np.isnan(std) or (valid.sum() == 0):
            z = np.zeros_like(s, dtype=float)
            z[:] = 0.5
        else:
            z = (s - mu) / std
        z_valid = z[valid]
        if len(z_valid) == 0:
            zmin = zmax = 0.0
        else:
            zmin, zmax = z_valid.min(), z_valid.max()
        if zmax - zmin == 0 or (np.isnan(zmin) or np.isnan(zmax)):
            n = np.zeros_like(z, dtype=float) + 0.5
        else:
            n = (z - zmin) / (zmax - zmin)
        # Not ranked => 0 in 0-1 scale (worst), so combined score is penalized
        n = np.where(valid, n, 0.0)
        norm_scores.append(n)
    merged["final_score"] = np.round(np.array(norm_scores).mean(axis=0), 3)

    # total_rank: 1 = best (highest final_score)
    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged["total_rank"] = np.arange(1, len(merged) + 1, dtype=int)

    # Averages (only over numeric values); round data_days to whole days, cagr to 2 decimals
    merged["avg_data_days"] = merged[[f"data_days_{m}" for m in models]].mean(axis=1).round(0)
    merged["avg_cagr_3y"] = merged[[f"cagr_3y_{m}" for m in models]].mean(axis=1).round(2)

    # Output columns: mfId name total_rank final_score avg_data_days avg_cagr_3y score_{m} ...
    out_cols = [
        "mfId",
        "name",
        "total_rank",
        "final_score",
        "avg_data_days",
        "avg_cagr_3y",
    ] + score_cols
    out = merged[out_cols].copy()

    # Fill blanks with reasons so cells are never empty
    out["avg_cagr_3y"] = out["avg_cagr_3y"].apply(
        lambda x: "N/A (insufficient history)" if pd.isna(x) else x
    )
    for c in score_cols:
        out[c] = out[c].apply(lambda x: "N/A (not ranked)" if pd.isna(x) else x)
    # avg_data_days NaN is rare; fill with reason if it happens
    out["avg_data_days"] = out["avg_data_days"].apply(
        lambda x: "N/A" if pd.isna(x) else x
    )
    return out


def write_compiled(sector: str, df: pd.DataFrame, results_dir: Path | None = None) -> Path:
    """Write compiled DataFrame to results/{SECTOR}.csv. Returns path."""
    results_dir = results_dir or RESULTS_DIR
    out_path = results_dir / f"{sector}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def compile_and_write(sector: str, results_dir: Path | None = None) -> Path:
    """Compile sector and write results/{SECTOR}.csv. Returns path to written file."""
    df = compile_sector(sector, results_dir=results_dir)
    return write_compiled(sector, df, results_dir=results_dir)


if __name__ == "__main__":
    import sys
    sectors = discover_sectors()
    if not sectors:
        print("No sector result files found.", file=sys.stderr)
        sys.exit(1)
    for sector in sectors:
        path = compile_and_write(sector)
        print(f"Compiled {sector} -> {path}")
