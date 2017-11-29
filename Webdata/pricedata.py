import urllib.parse
import urllib.request
import pprint
import json
import numpy as np
import pandas as pd
import time, datetime
import sys,os

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Bitfinex/')

def bitfinex(until='1451606400'):
    '''
    Gets candle data from Bitfinex
    '''
    pairs = {
        'Bitcoin': 'tBTCUSD',
        'Litecoin': 'tLTCBTC',
        'Ethereum': 'tETHBTC',
        'Monero': 'tXMRBTC'
    }
    for key, currency in pairs.items():
        url = 'https://api.bitfinex.com/v2/candles/trade:%s:%s/hist' % ('1h', currency)
        values = {'limit' : 1000}
        prices = []
        # finished = False
        # n = 0
        #for n in range(2):
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
            for i in range(5):
                try:
                    with urllib.request.urlopen(full_url) as response:
                        json_obj = json.loads(response.read().decode('utf-8'))
                        prices += json_obj
                        break
                except urllib.error.HTTPError as e:
                    print(e,'... retrying')
            
            # n += 1
            # print(n)
            # In case DDOS protection acivates
            # sys.stdout.write('.')
            # sys.stdout.flush()
            time.sleep(2)
            print(datetime.datetime.fromtimestamp(prices[-1][0]/1000))
            if int(last) == int(prices[-1][0]):
                break
        array = np.array(prices)
        # print(array)
        np.savetxt(os.path.join(data_path,key + ".csv"), array, delimiter=",")
        print('Saved %s Data from %s to %s' % (key, datetime.datetime.fromtimestamp(array[-1][0]/1000),datetime.datetime.fromtimestamp(array[0][0]/1000)))
    

def fixSmallErrors():
    '''
    Tool for checking integrity of previously downloaded files from Bitfinex and filling missing data with average prices.
    '''
    first_errors = []
    for root, dirs, filenames in os.walk(data_path):
        for f in filenames:
            print('Checking ',f)
            # pd.read_csv(os.path.join(data_path, f))
            array = np.loadtxt(os.path.join(data_path, f), delimiter=',')
            wrong = 0
            first = 0
            for count, line in enumerate(array[1:]):
                n = (array[count][0]-line[0])/(1000*60*60)
                if n != 1:
                    wrong += n
                    if first == 0 and n >= 10:
                        first = count
                        first_errors.append(count)
            print('%d hours missing.' % wrong)
            print('First occurence line %d of %d' % (first,len(array)))
    # print('Merging valid lines')
    # print(min(float(s) for s in first_errors))
    # TODO: fit in new rows with avg. price data
    
if __name__ == "__main__":
    # bitfinex()
    checkIntegrity()
    
