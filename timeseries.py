import pandas as pd
import numpy as np
import sys, os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from progress.bar import Bar
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
import matplotlib.pyplot as plt


def genPriceData(csv='Webdata/krakenEUR.csv', output='Webdata/XBTEUR.csv'):
    df = pd.DataFrame.from_csv(csv, header=None)
    df.index = df.index.map(lambda x:datetime.fromtimestamp(x))
    df.columns = ['Price','Volume']
    df.columns.name = 'Timestamp'

    print('rounding...')
    df['round'] = df.index.map(lambda x: x - timedelta(minutes=x.minute % 10, seconds=x.second, microseconds=x.microsecond))

    print('removing duplicates...')
    df = df.drop_duplicates(subset='round', keep='first')
    df = df.drop('round', 1)

    print("normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[:] = scaler.fit_transform(df[:])

    print(df[0])
    print(df[-1])
    df.to_csv(path_or_buf=output)


def genTrainData(df, steps = 10, exp=4, function=1, average=False):
    '''
    TODO: Implement function
    Collects the [steps] last entries in one row, then every [steps]^1 entry from [steps]^2, and so on, until every [steps]^([exp]-1) step from [steps]^[exp].
    Example with steps=3, exp=4, function=1, average=False:
    ((1 2 3) 6 9) 18 27) 54 81
    Example with steps=10, exp=4, function=0.5, average=False:
    ((1 2 3 4 5 6 7 8 9 10) 20 40 60 80 100) 333 666 1000) 5000 10000)
    The last [steps]^[exp] entries get cut off. If average is True, every [exp] cycle Collects all [steps] in an average value to append instead of appending all values.
    '''
    datadict = {}
    bar = Bar('Preparing Data', max=df.shape[0]-steps)
    for index, row in df.iloc[steps**exp:].iterrows():
        l = []
        for x in range(exp):
            last = df.loc[:index].tail(steps**(x+1))
            last = last.iloc[::(steps**x), :].iloc[::-1]
            last = last.ix[1:]
            l = l+[last.loc[x] for x in last.index]
        
        if average:
            av = sum(l)/float(len(l)) #Average instead of large list
            new = row.append(av)
        else:
            new = row.append(l)
            
        new.index = range(new.shape[0])
        datadict[index] = new
        bar.next()

    bar.finish()
    df2 = pd.DataFrame.from_dict(datadict, orient='index')
    return df2


def fitData():
    df = pd.read_csv('Webdata/train.csv', header=0, index_col=0)

    train, test = np.split(df,[int(0.8*len(df))])
    train = np.array(train.sample(frac=1))
    test = np.array(test)
    train_labels, volume, train_data = np.split(train,[1,2], axis=1)
    test_labels, volume, test_data = np.split(test,[1,2], axis=1)

    # data = np.array(df.ix[:,2:])
    # labels = np.array(df.ix[:,0])
    # train_data, test_data = np.split(data,[int(0.8*len(data))])
    # train_labels, test_labels = np.split(labels,[int(0.8*len(labels))])

    print(test_labels.shape, test_data.shape)
    print(train_labels.shape, train_data.shape)

    model = Sequential([
        Dense(40, input_shape=(train_data.shape[1],)),
        Activation('tanh'),
        Dense(100),
        Activation('tanh'),
        Dense(90),
        Activation('tanh'),
        Dense(1)
    ])

    # model.add(SimpleRNN(100, activation='tanh'))

    model.compile(optimizer='adam', loss='mse')

    model.fit(train_data, train_labels, epochs=100, batch_size=32)

    test_results = model.predict(test_data, batch_size=32, verbose=0)
    # plt.plot(test_results, label="results")
    # plt.plot(test_labels, label="price")
    # plt.show()
    r = 0
    a = 0
    w = 0

    for n, result in enumerate(test_results[1:]):
        real = test_labels[n-1]/test_labels[n]
        ai = test_labels[n-1]/result
        if real < 1:
            r += 1
            if ai < 1:
                w += 1
        if ai < 1:
            a += 1
    print("total neg:",r,"ai neg:",a,"accurate:",w)

    return test_results, test_labels

# 
# for f in os.listdir("Webdata"):
#     if f.endswith(".csv"):
#         genPriceData(csv=f, output='data_'+f)

if __name__ == "__main__":
    df = genTrainData(pd.DataFrame(np.arange(100000)))
    print(df)
# genPriceData()
# genTrainData()
# fitData()

# dataset = read_csv('Webdata/XBTEUR.csv', header=0, index_col=0)
# values = dataset.values
# values = values.astype('float32')
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
#
# print(scaled)
