import pandas as pd
import numpy as np
import gc, time, sqlite3
from polygon import RESTClient

def cleanPoly(aggs):
    """
    Clean API Response
    :param aggs: polygon.Aggs - ohlc bars
    :return: pd.DataFrame - clean ohlc records
    """
    records = list()
    for a in aggs:
        rec = [a.timestamp,
               a.open,
               a.high,
               a.low,
               a.close,
               a.volume,
               a.transactions]
        records.append(rec)
    records = pd.DataFrame(records,
                           columns=['date',
                                    'open',
                                    'high',
                                    'low',
                                    'close',
                                    'volume',
                                    'transactions'])
    gc.collect()
    records['date'] = pd.to_datetime(records['date'], unit='ms')
    return records

def main():
    final = list()
    client = RESTClient(api_key=KEY) # instantiate Polygon client
    for i in range(4):
        print(f'--- Downloading {START + i} {TICKER.split(":")[1]} data ---')
        start_date = f'{START + i}-01-01'
        end_date = f'{START + i}-12-31'
        aggs = client.list_aggs(ticker=TICKER,
                                multiplier=MULT,
                                timespan=INTERVAL,
                                from_=start_date,
                                to=end_date,
                                limit=50000)
        clean = cleanPoly(aggs=aggs) # clean response
        final.append(clean)
        time.sleep(0.5)
    final = pd.concat(final, axis=0)
    gc.collect()
    final['log'] = np.log(final['close'])
    final['logRet'] = final.log.diff()
    final.sort_values(by='date', ascending=False, inplace=True)
    final.dropna(axis=0, inplace=True)
    final.to_sql(TICKER.split(':')[1], engine, if_exists='replace', index=False)
    return

if __name__ == '__main__':
    KEY = 'SwYjDFUR1C2brsZPpO2ZWDdLafh2nCq0'
    START = 2020
    TICKER = 'X:BTCUSD'
    MULT = 15
    INTERVAL = 'minute'
    engine = sqlite3.connect('./btc-data.db')
    main()
