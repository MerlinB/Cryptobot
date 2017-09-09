import pandas as pd
from datetime import datetime
from Kraken.public import getOHLC

# df = pd.DataFrame.from_csv('Webdata/krakenEUR.csv', header=None)
# df.index = df.index.map(lambda x:datetime.fromtimestamp(x))
# df.columns = ['Price','Volume']
# df.columns.name = 'Timestamp'

## Kraken API reicht nicht weit genug zurück.
# df2 = pd.DataFrame(getOHLC('XBTEUR', interval=5)['XXBTZEUR'])
# df2 = df2.set_index(0)
# df2.index = df2.index.map(lambda x:datetime.fromtimestamp(x))
# df2.columns.name = 'Timestamp'
# print(df2)

df2 = pd.DataFrame.from_csv('Webdata/mempool.log', header=None)
print(df2)
print(df2.iloc[-1])
