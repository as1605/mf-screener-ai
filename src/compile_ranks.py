import os
import pandas as pd
from io import StringIO
from compile_xirr import write_xirr_scores

RESULTS_DIR = "results/"
BRANCH_DATE_PREFIX = "date/"

def get_dates():
    """Get all branches from all remote repositories"""
    os.system("git fetch --all")
    branches = os.popen("git branch -r").read().splitlines()
    # remove origin/ from branches and remove ->
    dates = []
    for branch in branches:
        if BRANCH_DATE_PREFIX in branch:
            dates.append(branch.replace("origin/"+BRANCH_DATE_PREFIX, "").strip())
    return sorted(list(set(dates)))

def get_categories():
    categories = []
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".csv") and "_" not in file:
            categories.append(file.split(".")[0])
    return categories

def process_category(category: str, dates: list[str]):
    file_path = RESULTS_DIR + category + ".csv"
    df = None
    for date in dates:
        branch = f"origin/{BRANCH_DATE_PREFIX}{date}"
        cmd = f"git show '{branch}':'{file_path}'"
        print(cmd)
        content = os.popen(cmd).read()
        if not content.strip():
            continue
        try:
            df_d = pd.read_csv(StringIO(content))
        except Exception:
            continue
        if "mfId" not in df_d.columns or "total_rank" not in df_d.columns:
            continue
        df_d = df_d[["mfId", "name", "total_rank"]].set_index("mfId")

        rank_series = df_d["total_rank"].rename(date)
        if df is None:
            df = df_d[["name"]].copy()
            df[date] = rank_series
        else:
            df = df.join(rank_series, how="outer")
            df["name"] = df_d["name"].combine_first(df["name"])

    if df is not None:
        date_cols = [c for c in df.columns if c != "name"]
        rev_date_cols = sorted(date_cols, reverse=True)
        latest_date = rev_date_cols[0]
        df = df[["name"] + rev_date_cols]
        df = df.sort_values(by=latest_date, na_position="last")
        for c in rev_date_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        os.makedirs("results/ranks", exist_ok=True)
        df.to_csv(f"results/ranks/{category}.csv")

def main():
    dates = get_dates()
    print(dates)
    categories = get_categories()
    print(categories)
    for category in categories:
        process_category(category, dates)
    print(f"Wrote XIRR scores to {write_xirr_scores()}")
    
if __name__ == "__main__":
    main()
