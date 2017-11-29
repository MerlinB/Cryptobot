import urllib.parse
import urllib.request

def bitfinex():
    url = 'https://api.bitfinex.com/v2/candles/trade:%s:%s/hist' % ('1h','tBTCUSD')
    values = {'limit' : 10}

    data = urllib.parse.urlencode(values)
    data = data.encode('ascii') # data should be bytes
    req = urllib.request.Request(url, data)
    print(req.full_url)
    with urllib.request.urlopen(req.full_url) as response:
       the_page = response.read()
    print('Result: ',the_page)
    # for n in range(3):
    #     data[end]
    
if __name__ == "__main__":
    bitfinex()
