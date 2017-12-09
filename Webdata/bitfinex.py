import urllib.parse
import urllib.request
import pprint
import json
import numpy as np
import pandas as pd
import time, datetime
import sys,os

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Bitfinex/')

def bitfinex(until='1451606400', resolution='1h'):
    '''
    Gets candle data from Bitfinex
    '''
    pairs = {
        'Bitcoin': 'tBTCUSD',
        'Litecoin': 'tLTCBTC',
        'Ethereum': 'tETHBTC',
        'Monero': 'tXMRBTC'
    }
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    for key, currency in pairs.items():
        print('Fetching %s data...' % key)
        url = 'https://api.bitfinex.com/v2/candles/trade:%s:%s/hist' % (resolution, currency)
        values = {'limit' : 1000}
        prices = []
        last = 0
        while True:
            if prices:
                # Begin at previous last timestamp
                last = prices[-1][0]
                values['end'] = last
                prices = prices[:-1]
                # print(int(last)/1000, int(until))
            data = urllib.parse.urlencode(values)
            full_url = url + '?' + data
            for i in range(5): # Max retries
                try:
                    with urllib.request.urlopen(full_url) as response:
                        json_obj = json.loads(response.read().decode('utf-8'))
                        prices += json_obj
                        break
                except Exception as e: #except urllib.error.HTTPError as e:
                    print(e,'... retrying')
            
            time.sleep(2) # Wait to not reach API limit.
            print(datetime.datetime.fromtimestamp(prices[-1][0]/1000))
            if int(last) == int(prices[-1][0]): 
                break # End fetch if dates repeat.
            
        array = np.array(prices)
        np.savetxt(os.path.join(data_path,key + ".csv"), array, delimiter=",")
        print('Saved %s Data from %s to %s' % (key, datetime.datetime.fromtimestamp(array[-1][0]/1000),datetime.datetime.fromtimestamp(array[0][0]/1000)))

    
def find_gap(gap, filenames):
    '''
    Returns date of first gap occurence
    '''
    first_errors = []
    for f in filenames:
        print('Checking ',f)
        array = np.loadtxt(os.path.join(data_path, '%s.csv' % f), delimiter=',')
        for count, line in enumerate(array[1:]):
            n = (array[count][0]-line[0])/(1000*60*60) # TODO: Make other timeframes possible
            if  n >= gap:
                    first_errors.append(array[count][0])
                    # print(n,line, array[count][0])
                    break
            first_errors.append(array[-1][0])
    return datetime.datetime.fromtimestamp(max(first_errors)/1000)


def clean_data(filenames, max_gap=10):
    '''
    Cleans Bitfinex data: 
        - Removes duplicates
        - Fills empty rows
        - Truncates at max acceptable gap
        - Combines Dataframes
    '''
    
    end = find_gap(max_gap, filenames)
    
    def dateparse(time_in_secs):    
        return datetime.datetime.fromtimestamp(float(float(time_in_secs)/1000))
    
    def clean(filename):
        print('Cleaning %s data' % filename)

        # Load pandas Dataframe with DateTimeIndex
        array = pd.read_csv(os.path.join(data_path, '%s.csv' % filename), index_col=0, names=['DateTime','Open','Close','High','Low','Volume'], header=0, parse_dates=True, date_parser=dateparse)
        
        # Remove duplicates
        array = array[~array.index.duplicated()]
        
        # Drop all columns but Close and Volume
        array = array.drop(['Open','High','Low'], axis=1)
        array.columns = [filename+column for column in array.columns]
        
        # Fill missing dates
        dates = pd.date_range(array.index[-1],array.index[0],freq='1H')
        array = array.reindex(dates,fill_value=0)
        array.index.name = 'DateTime'
        
        # Fill empty rows
        # TODO: Problem if DF ends with empty row.
        last = array.iloc[0]
        for index, row in array[1:].iterrows():
            if row[1] == 0:
                row[1] = last[1]
            last = row
        array = array[::-1]
        
        # Truncate at max acceptable gap
        array = array.truncate(after=end)
            
        return array

    dframes = []
    for f in filenames:
        dframes.append(clean(f))
        
    combined = pd.concat(dframes, axis=1)
    combined.to_csv(path_or_buf=os.path.join(data_path,'Cleaned_data.csv'))
    return combined

if __name__ == "__main__":
    # bitfinex()
    
    filenames = [
        'Bitcoin',
        'Litecoin',
        'Monero',
        'Ethereum'
    ]
    
    array = clean_data(filenames=filenames)
    print(array)
    
    
    
    
