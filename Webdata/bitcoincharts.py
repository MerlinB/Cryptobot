#from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import time, csv
from datetime import date, timedelta


span = 10 # Days
interval = [[
    date.today(),
    date.today() - timedelta(days=span),
]]

for n in range(2): # Month
    interval.append([
        interval[-1][1] - timedelta(hours=1),
        interval[-1][1] - timedelta(days=span+1, hours=1),
    ])

data = []

print('=====================================START')

for times in reversed(interval): 
    url = "https://bitcoincharts.com/charts/bitstampUSD#rg10zig5-minzczsg"+str(times[1])+"zeg"+str(times[0])+"ztgSzm1g10zm2g25zv" #Minutes
    #url = "https://bitcoincharts.com/charts/bitstampUSD#rg150zigHourlyzczsg"+str(interval[1])+"zeg"+str(interval[0])+"ztgSzm1g10zm2g25zv" # 1h

    browser = webdriver.Firefox()
    browser.get(url)
    elem = browser.find_element_by_xpath('//*[@id="content_chart"]/div/div[2]/a')
    elem.click()

    print("loading...")
    time.sleep(3)

    soup = BeautifulSoup(browser.find_element_by_xpath('//*[@id="chart_table"]').get_attribute("outerHTML"), 'html.parser')
    browser.quit()

    head = soup.find_all('th')
    for n, th in enumerate(head):
        head[n] = th.string
        
    for tr in soup.find_all('tr')[2:]:
        tds = tr.find_all('td')
        for n, td in enumerate(tds):
            tds[n] = td.string
            
        data.append(tds)

with open('prices.csv', 'w', newline="") as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(head)
    for row in data:
        #row = row.split()
        wr.writerow(row)

print('=====================================END')
print("=====================================COMBINE")
