# Compile and Publish results

Your task is to compile results from multiple models for {SECTOR} and publish results to a Google Sheet

## Compiling results

- First create a script which reads files from `results/{SECTOR}_{model}.csv` format with columns `mfId name rank score data_days cagr_3y ...`. It should take {SECTOR} as input and should infer the model name from the file paths.
- Then join the results by mfId, and output a sheet `results/{SECTOR}.csv` with columns `mfId name total_rank final_score avg_data_days avg_cagr_3y score_{model1} score_{model2} score_{model3}...`
    - `final_score` should normalise the scores with **stddev** by each model to 0-1 range, then take average for final score.
    - `total_rank` should be 1 to best fund on `final_score`. Sort the sheet from `final_score` high to low.
    - `data_days` and `cagr_3y` should be average from each each.
    - `score_{model[]}` should be the score given by that model

## Publishing

- Use a google service account with credentials in `GOOGLE_SERVICE_ACCOUNT_KEY` path defined in `.env`
- Extract `SPREADSHEET_ID` from `GOOGLE_SHEET_URL` defined in `.env`
- Create a sheet with name same as `{SECTOR}`, upload only `results/{SECTOR}.csv` to it. Replace if already present
- Resize columns to fit the data appropriately
- Add conditional formatting on each score column. Lowest should be white and Highest should be suitable shade of green in each score column.
- Bold the name and final score
- Add filter to the top row

## Final Script
Final script should run both steps for each {SECTOR} for which we have results
