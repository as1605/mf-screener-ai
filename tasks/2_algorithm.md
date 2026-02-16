# MF Scoring Algorithm

{SECTOR}={Small Cap}
{model}={Codex}

Your task is to create an algorithm to score Indian Mutual Funds from {SECTOR} sector. A higher score should predict higher returns in the next 1 year. You are given NAV history of all funds in that sector from MfDataProvider


## Task
- Understand how MfDataProvider can be used (given below).
- Understand requirements to gain best returns in {SECTOR}
- Research indicators of funds and strategies which can give great returns in {SECTOR} in next 1 year
- Research advanced metrics used by intelligent Portfolio Managers to analyze MFs
- Research long term strategies which can be implemented with this NAV data. Thouroughly think about how each strategy or pattern can be used. Do not limit to simple ratios, make full use of 
- Research how to identify market swings and check fund performance against it. The fund should be able to take advantage of swings to maximise total return in long term
- See how we can create rolling metrics and backtest performance for tuning metrics with the given data
- Finally, create a script `src/algorithms/{SECTOR}_{model}.py` which does an intelligent analysis from the given data for each {SECTOR} fund, and assigns it a score.
- It should output a sheet `results/{SECTOR}_{model}.tsv` with columns: mfId, name, rank, score, data_days, cagr_5y, other metrics. Rank should be 1 for best, CAGR should be a percentage. data_days should tell how many days data was available for that fund

## Data Statistics

Sector Name: Equity
Sub Sector Name: Small Cap Fund
Total AUM: 356,098.73
Total Funds: 34
Funds with data > 1Y: 30
Funds with data > 5Y: 22

## MfDataProvider - Quick API Reference

### Core Functions

#### `list_all_mf() → pd.DataFrame`
Get all mutual funds with metadata.
- **Arguments:** None
- **Returns:** DataFrame with columns: `[mfId, name, aum, sector, subsector]`

#### `list_mf_by_sector() → Dict[str, Dict[str, List[str]]]`
Get funds organized by sector and subsector.
- **Arguments:** None
- **Returns:** `{"sector": {"subsector": ["mfId1", "mfId2", ...]}}`

#### `get_mf_chart(mf_id: str) → pd.DataFrame`
Get historical NAV data for a mutual fund.
- **Arguments:** `mf_id` (str) - Fund ID (e.g., 'M_PARO')
- **Returns:** DataFrame with columns: `[timestamp, nav]`

#### `list_indices() → Dict[str, str]`
Get available indices.
- **Arguments:** None
- **Returns:** `{"Large Cap": ".NSEI", "Mid Cap": ".NIMI150", ...}`

#### `get_index_chart(index_id: str) → pd.DataFrame`
Get historical price data for an index.
- **Arguments:** `index_id` (str) - Index ID (e.g., '.NSEI') or name (e.g., 'Large Cap')
- **Returns:** DataFrame with columns: `[timestamp, nav]`

---

### Data Fetching Functions (with force_refresh)

#### `fetch_all_mf_list(force_refresh: bool = False) → pd.DataFrame`
Fetch/refresh list of all mutual funds.
- **Arguments:** `force_refresh` (bool) - Force API call if True
- **Returns:** DataFrame with columns: `[mfId, name, aum, sector, subsector]`

#### `fetch_mf_chart(mf_id: str, force_refresh: bool = False) → pd.DataFrame`
Fetch/refresh chart data for a fund.
- **Arguments:** 
  - `mf_id` (str) - Fund ID
  - `force_refresh` (bool) - Force API call if True
- **Returns:** DataFrame with columns: `[timestamp, nav]`

#### `fetch_index_chart(index_id: str, force_refresh: bool = False) → pd.DataFrame`
Fetch/refresh chart data for an index.
- **Arguments:**
  - `index_id` (str) - Index ID
  - `force_refresh` (bool) - Force API call if True
- **Returns:** DataFrame with columns: `[timestamp, nav]`

#### `fetch_all_data() → None`
Fetch everything: all MF list, all MF charts, all index charts.
- **Arguments:** None
- **Returns:** None (saves to disk)

---

### Quick Example

```python
from mf_data_provider import MfDataProvider

provider = MfDataProvider()

## Get all funds
df_all = provider.list_all_mf()                    ## → DataFrame

## Get funds by sector
sectors = provider.list_mf_by_sector()             ## → Dict

## Get fund chart
chart = provider.get_mf_chart('M_PARO')            ## → DataFrame

## Get indices
indices = provider.list_indices()                   ## → Dict

## Get index chart
idx_chart = provider.get_index_chart('Large Cap')  ## → DataFrame
```
