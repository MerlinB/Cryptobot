from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pprint, time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from bs4 import BeautifulSoup

# BTC Node Source
# https://github.com/jhoenicke/mempool

def dataframe(unsorted):
    noindex = unsorted[:,:,1]
    index = pd.Series(unsorted[0,:,0])
    index = index.map(lambda x:datetime.fromtimestamp(x/1000))
    # index = index.map(lambda x: datetime.strptime(x[:-3], '%Y-%m-%d %X'))
    frame = pd.DataFrame(noindex, index=np.arange(len(noindex)), columns=index)
    sorted = frame.T
    sorted.index.name = 'Timestamp'
    sorted['SUM'] = sorted[:].sum(axis=1)
    # sorted.index = sorted.index.map(lambda x:datetime.fromtimestamp(x/1000))
    return(sorted)

def memGen():
    print('Generating Mempool CSVs.')
    url = "https://core.jochen-hoenicke.de/queue/#30d"

    browser = webdriver.Firefox()
    browser.get(url)
    time.sleep(5)
    data = browser.execute_script("return data")
    browser.quit()

    graphs = {
        "unconfirmed_tx": np.array(data[0][:-1], dtype=np.float32),
        "pending_tx_fee": np.array(data[1][:-1], dtype=np.float32),
        "mempool_size": np.array(data[2][:-1], dtype=np.float32),
        }

    for key, g in graphs.items():
        graphs[key] = dataframe(g)
        graphs[key].to_csv(path_or_buf=(key+".csv"))
        #g.plot()
    
    print('CSVs created.')
    #plt.show()
    return graphs
        
    
def priceGen(timespan=10):
    print('Generating prices CSV.')
    days=10
    interval = [[
        date.today(),
        date.today() - timedelta(days=days),
    ]]
    
    for n in range(int(round(timespan/days))): # Month
        interval.append([
            interval[-1][1] - timedelta(days=1, hours=1),
            interval[-1][1] - timedelta(days=days+1, hours=1),
        ])
    
    frames = []
    
    for n, times in enumerate(reversed(interval)): 
        url = "https://bitcoincharts.com/charts/bitstampUSD#rg10zig5-minzczsg"+str(times[1])+"zeg"+str(times[0])+"ztgSzm1g10zm2g25zv" #Minutes
        #url = "https://bitcoincharts.com/charts/bitstampUSD#rg150zigHourlyzczsg"+str(interval[1])+"zeg"+str(interval[0])+"ztgSzm1g10zm2g25zv" # 1h
        
        browser = webdriver.Firefox()
        browser.get(url)

        elem = browser.find_element_by_xpath('//*[@id="content_chart"]/div/div[2]/a')
        elem.click()
    
        print("loading data...",str(n+1),"/",str(len(interval)))
        time.sleep(3)
    
        soup = BeautifulSoup(browser.find_element_by_xpath('//*[@id="chart_table"]').get_attribute("outerHTML"), 'html.parser')

        head = pd.Series(soup.find_all('th'))
        head = head.map(lambda x: x.string)
            
        data = soup.find_all('tr')[2:]
        frame = pd.DataFrame()
        
        for n, tr in enumerate(data):
            data[n] = tr.find_all('td')
            for m, td in enumerate(data[n]):
                #print(n,m,td.string)
                frame.loc[n,m] = td.string
        
        frame = frame.rename(columns=head).set_index(['Timestamp'])
        frame.index = frame.index.map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %X'))
        frames.append(frame)
        browser.quit() 
        
    dataframe = pd.concat(frames)
    dataframe.to_csv(path_or_buf=('prices.csv'))
    print('CSV created.')
    #pprint.pprint(dataframe)
    return dataframe
    
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
    
def fcl(df, dtObj):
    return df.iloc[np.argmin(np.abs(df.index.to_pydatetime() - dtObj))]

def combine(prices, df):
    for index, price in prices[:].iterrows():
        df2 = df.loc[df.index < (index.to_pydatetime()+timedelta(days=1)).strftime("%Y-%m-%d")]
        df_small = df2.loc[df2.index > index.to_pydatetime().strftime("%Y-%m-%d")]
        
        # i = min(np.abs(df_small.index.to_pydatetime() - index.to_pydatetime()))
        # print(index," -> ",)
        print(index," -> ")
        pprint.pprint(df_small)
        break
        
#print(nearest([1,2,3,4,5,6.5], 3.6))
mempool = memGen()
combine(priceGen(), mempool['unconfirmed_tx'])
# print(mempool['unconfirmed_tx'].loc['20170813':'20170814'])
# mempool['unconfirmed_tx'][mempool['unconfirmed_tx'].index.map(lambda x: ((x>'2017-08-13') and (x<'2017-08-15')))]
# print(mempool['unconfirmed_tx'].filter(like='2017-08-13', axis=0))


    # print(mempool['unconfirmed_tx'].index.tolist())
    # print(index.to_pydatetime().strftime("%Y-%m-%d") in x for x in mempool['unconfirmed_tx'].index.tolist())
    # print(mempool['unconfirmed_tx'].iloc[index.to_pydatetime().date()].index)
    # print(mempool['unconfirmed_tx'][[index.to_pydatetime().strftime("%Y-%m-%d") in x for x in mempool['unconfirmed_tx'].index.strftime("%Y-%m-%d %X")]].index)
    # 
    # print(mempool['unconfirmed_tx'].loc[index.to_pydatetime().strftime("%Y%m%d"):(index.to_pydatetime()-timedelta(days=1)).strftime("%Y%m%d")])
    # print(mempool['unconfirmed_tx'].loc[index.to_pydatetime().strftime("%Y%m%d")])
    # print(mempool['unconfirmed_tx'].loc[index.to_pydatetime()])
    # i = np.abs(mempool['unconfirmed_tx'].index.to_pydatetime() - index.to_pydatetime())
    # print(i)
    
    # alle timedeltas zu index
    # print(min(i))   
    # print(mempool.iloc[i])
