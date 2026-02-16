"""
Publish compiled results to a Google Sheet.
Uses GOOGLE_SERVICE_ACCOUNT_KEY and GOOGLE_SHEET_URL from .env.
"""
import os
import re
from pathlib import Path

import pandas as pd
import gspread
from gspread_formatting import (
    ConditionalFormatRule,
    GradientRule,
    InterpolationPoint,
    CellFormat,
    Color,
    GridRange,
    get_conditional_format_rules,
    set_column_widths,
    format_cell_range,
    cellFormat,
    textFormat,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_env():
    """Load .env from project root if present."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(env_path)
    except ImportError:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)


def get_spreadsheet_id_from_url(url: str) -> str:
    """Extract spreadsheet ID from GOOGLE_SHEET_URL."""
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError(f"Cannot extract SPREADSHEET_ID from URL: {url}")
    return m.group(1)


def get_client():
    """Build gspread client from GOOGLE_SERVICE_ACCOUNT_KEY path."""
    load_env()
    key_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not key_path:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY not set in .env")
    path = Path(key_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / key_path
    if not path.exists():
        raise FileNotFoundError(f"Service account key not found: {path}")
    return gspread.service_account(filename=str(path))


def publish_sector(sector: str, results_dir: Path | None = None) -> None:
    """
    Create or replace worksheet named {sector}, upload results/{sector}.csv,
    resize columns, add conditional formatting on score columns, bold name and final_score.
    """
    results_dir = results_dir or RESULTS_DIR
    csv_path = results_dir / f"{sector}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Compiled file not found: {csv_path}")

    load_env()
    sheet_url = os.environ.get("GOOGLE_SHEET_URL")
    if not sheet_url:
        raise ValueError("GOOGLE_SHEET_URL not set in .env")
    spreadsheet_id = get_spreadsheet_id_from_url(sheet_url)

    gc = get_client()
    sh = gc.open_by_key(spreadsheet_id)

    # Sheet title: same as sector (Google Sheets allows most chars; avoid : \ / ? * [ ])
    sheet_title = sector[:100]

    try:
        ws = sh.worksheet(sheet_title)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_title, rows=1000, cols=30)

    df = pd.read_csv(csv_path)
    # Ensure we write strings for display; keep numbers as numbers
    data = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()
    ws.update(data, value_input_option="USER_ENTERED")

    nrows, ncols = len(data), len(df.columns)
    if nrows == 0 or ncols == 0:
        return

    # Resize columns to fit (reasonable widths: first col narrow, name wide, rest medium)
    col_config = []
    for j, col in enumerate(df.columns):
        letter = _col_index_to_a1(j)
        if col == "mfId":
            col_config.append((letter, 90))
        elif col == "name":
            col_config.append((letter, 280))
        else:
            col_config.append((letter, 100))
    set_column_widths(ws, col_config)

    # Score columns: final_score and score_*
    score_cols = [c for c in df.columns if c == "final_score" or c.startswith("score_")]
    # Conditional format: lowest = white, highest = green (per column)
    rules = get_conditional_format_rules(ws)
    rules.clear()
    white = Color(1, 1, 1)
    green = Color(0.4, 0.8, 0.4)
    for col_name in score_cols:
        j = df.columns.get_loc(col_name)
        col_letter = _col_index_to_a1(j)
        a1_range = f"{col_letter}2:{col_letter}{nrows}"
        rule = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range(a1_range, ws)],
            gradientRule=GradientRule(
                minpoint=InterpolationPoint(
                    type="MIN",
                    color=white,
                ),
                maxpoint=InterpolationPoint(
                    type="MAX",
                    color=green,
                ),
            ),
        )
        rules.append(rule)
    rules.save()

    # Bold header row and columns "name" and "final_score"
    name_col = df.columns.get_loc("name") + 1
    final_score_col = df.columns.get_loc("final_score") + 1
    name_letter = _col_index_to_a1(name_col - 1)
    final_letter = _col_index_to_a1(final_score_col - 1)
    fmt_bold = cellFormat(textFormat=textFormat(bold=True))
    format_cell_range(ws, f"A1:{_col_index_to_a1(ncols - 1)}1", fmt_bold)
    format_cell_range(ws, f"{name_letter}1:{name_letter}{nrows}", fmt_bold)
    format_cell_range(ws, f"{final_letter}1:{final_letter}{nrows}", fmt_bold)

    # Create filter on top row (header + all data)
    sh.batch_update({
        "requests": [{
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": 0,
                        "endRowIndex": nrows,
                        "startColumnIndex": 0,
                        "endColumnIndex": ncols,
                    }
                }
            }
        }]
    })


def _col_index_to_a1(j: int) -> str:
    """Convert 0-based column index to A1 letter(s)."""
    out = []
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        out.append(chr(65 + r))
    return "".join(reversed(out))


if __name__ == "__main__":
    import sys
    from compile_results import discover_sectors, compile_and_write

    load_env()
    sectors = discover_sectors()
    if not sectors:
        print("No sector result files found.", file=sys.stderr)
        sys.exit(1)
    for sector in sectors:
        compile_and_write(sector)
        publish_sector(sector)
        print(f"Published {sector}")
