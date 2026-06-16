import os
import pandas as pd
from io import StringIO

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
    return dates

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
        df_d = pd.read_csv(StringIO(content))
        df_d.set_index("mfId")
        df_d = df_d[["mfId", "name", "total_rank"]]
        df_d = df_d.set_index("mfId")

        if df is None:
            df = df_d[["name"]].copy()

        df = df.join(
            df_d[["total_rank"]].rename(
                columns={"total_rank": date}
            ),
            how="left"
        )
    df.to_csv(f"results/ranks/{category}.csv")

def main():
    dates = get_dates()
    print(dates)
    categories = get_categories()
    print(categories)
    for category in categories:
        process_category(category, dates)
    
if __name__ == "__main__":
    main()
