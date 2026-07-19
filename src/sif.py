from mf_data_provider import MfDataProvider
import pandas as pd

PATH_NAV = 'results/experiments/sif_nav.csv'
PATH_3M = 'results/experiments/sif_3m.csv'
PATH_6M = 'results/experiments/sif_6m.csv'

TRADING_DAYS_PER_MONTH = 21

def main():
    mf_data_provider = MfDataProvider()
    df = mf_data_provider.list_all_mf()
    sifs = df[df['name'].str.contains('Long-Short') | df['name'].str.contains('Long Short')]
    sifs = sifs.sort_values(by='name')

    charts = []
    for sif, name in zip(sifs['mfId'], sifs['name']):
        chart = mf_data_provider.get_mf_chart(sif, duration='5y')
        chart["timestamp"] = pd.to_datetime(chart["timestamp"])
        chart = chart.set_index('timestamp').rename(columns={'nav': name})
        charts.append(chart)
    
    charts = pd.concat(charts, axis=1)
    charts.sort_index(inplace=True)
    charts.to_csv(PATH_NAV)
    ret_3mo = charts.pct_change(3 * TRADING_DAYS_PER_MONTH)
    ret_3mo.to_csv(PATH_3M)
    ret_6mo = charts.pct_change(6 * TRADING_DAYS_PER_MONTH)
    ret_6mo.to_csv(PATH_6M)

if __name__ == "__main__":
    main()