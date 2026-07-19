from mf_data_provider import MfDataProvider
import pandas as pd

PATH = 'results/experiments/sifs.csv'

def main():
    mf_data_provider = MfDataProvider()
    df = mf_data_provider.list_all_mf()
    sifs = df[df['name'].str.contains('Long-Short') | df['name'].str.contains('Long Short')]

    charts = []
    for sif, name in zip(sifs['mfId'], sifs['name']):
        chart = mf_data_provider.get_mf_chart(sif, duration='5y')
        chart = chart.set_index('timestamp').rename(columns={'nav': name})
        charts.append(chart)
    
    charts = pd.concat(charts, axis=1)
    charts.sort_index(inplace=True)
    charts.to_csv(PATH)

if __name__ == "__main__":
    main()