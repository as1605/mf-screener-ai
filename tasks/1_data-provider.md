# Mutual Fund Data Provider

Create a data provider util class in python which fetches and provides data. Write production ready code with proper code structure and logging, and failure handling.

## Given data

### List MFs
You have an API which fetches a complete list of MFs
```python
requests.post(
    "https://api.tickertape.in/mf-screener/query",
    headers={"accept-version": "8.14.0"},
    json={
        "match": {"option": ["Growth"]},
        "sortBy": "aum",
        "sortOrder": -1,
        "project": ["subsector", "option", "aum", "ret3y", "expRatio"],
        "offset": 0,
        "count": 2000,
        "mfIds": [],
    },
)
```
with following result format
```json
{
    "success": true,
    "data": {
        "result": [
            {
                "mfId": "M_PARO",
                "slug": "/mutualfunds/parag-parikh-flexi-cap-fund-M_PARO",
                "name": "Parag Parikh Flexi Cap Fund",
                "values": [
                    {
                        "filter": "aum",
                        "doubleVal": 133969.8052
                    },
                    {
                        "filter": "ret3y",
                        "doubleVal": 21.3531307347264
                    },
                    {
                        "filter": "expRatio",
                        "doubleVal": 0.63
                    },
                    {
                        "filter": "subsector",
                        "strVal": "Flexi Cap Fund"
                    },
                    {
                        "filter": "option",
                        "strVal": "Growth"
                    }
                ],
                "sector": "Equity"
            },
            // ... and other MF data
        ]
    }
}
```

### MF Data

You are given an API to fetch the chart data
`https://api.tickertape.in/mutualfunds/{mfId}/charts/inter?duration=max`
Which returns a format 
```json
{
    "success": true,
    "data": [
        {
            "h": 13.3304,
            "l": 9.8829,
            "r": 32.36000000000001,
            "points": [
                {
                    "ts": "2023-09-25T00:00:00Z",
                    "lp": 10
                },
                // ... and other dates NAV
            ]
        }
    ]
}
```

### Indices
To get the chart data for any index, we can use 
`https://api.tickertape.in/stocks/charts/inter/{indexId}?duration=max`

Where the indexId are mapped as follows
- Large Cap: `.NSEI`
- Mid Cap: `.NIMI150`
- Small Cap: `.NISM250`
- Total Market: `.NIFTY500`
- Gold: `GBES`

## Task

1. Create a python script which keeps `DATA_DIR = './data/yyyy-mm-dd/'` and fetches
- list of all MFs along with metadata. Columns: mfId, name, aum, sector, subsector: `{DATA_DIR}/ALL.csv`
- individual MF chart data. Columns: timestamp, nav: `{DATA_DIR}/mf/{mfId}.csv`
- index chart data. Columns: timestamp, nav: `{DATA_DIR}/index/{indexId}.csv`

On running `__main__`, fetch all MFs and all Indices

2. Provide a python class MfDataProvider to expose this data and provide following functions
- list_all_mf() -> return ALL.csv as a pd df
- list_mf_by_sector() -> return a dict of dict of list. `{"sector": {"subsector": ["mfId"]}}`
- get_mf_chart(mfId: str) -> return pd df
- list_indices() -> return a dict of `{"name": "indexId"}`
- get_index_chart(indexId: str) -> return pd df
- fetch_all_data() -> this should be called in `__main__` . Optimize with multithreading to get 5-10x speed at each stage.

3. Provide a crisp documentation/API definition for other LLM/dev to generate python codes to import and use this class properly. 
Create a function to output a statistic/integration metadata in markdown with
* For each subsector:
    - Sector
    - Subsector Name
    - Total AUM
    - No of funds with data > 1y
    - No of funds with data > 5y
* For each index:
    - Name
    - Index Id
    - 5y rolling return
