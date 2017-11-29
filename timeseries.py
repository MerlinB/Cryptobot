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


def genTrainData(steps = 10, exp=4, csv='Webdata/XBTEUR.csv', output="Webdata/train.csv"):
    df = pd.read_csv(csv, header=0, index_col=0)
    datadict = {}
    bar = Bar('Preparing Data', max=df.shape[0]-steps)
    for index, row in df.iloc[steps**exp:].iterrows():
        l = []
        for x in range(exp):
            last = df.loc[:index].tail(steps**(x+1))
            last = last.iloc[::(steps**x), :].iloc[::-1]
            last = last.ix[1:]
            l = l+[last.loc[x] for x in last.index]

        av = sum(l)/float(len(l)) #Average instead of large list
        new = row.append(av)
        new.index = range(new.shape[0])
        datadict[index] = new
        bar.next()

    bar.finish()
    df2 = pd.DataFrame.from_dict(datadict, orient='index')
    df2.to_csv(path_or_buf=output)
    print(df2)


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


for f in os.listdir("Webdata"):
    if f.endswith(".csv"):
        genPriceData(csv=f, output='data_'+f)

# genPriceData()
# genTrainData()
# fitData()

# def botSim(test_results, test_labels)
#     #Start balance
#     btc = 10.0
#     eur = 10000.0
#
#     lastprice = test_labels[-1]
#
#     #Strategies
#     allin = lastprice*(btc+(eur/test_labels[0]))
#     tenhav = 0
#     dnn = 0
#
#     for t in test_labels:

    #     change = (result/test_labels[n-1]-1)
    #     print(change*100,"%")
    #     if eur >= btc*change*test_labels[n-1]:
    #         btc = btc + btc*change
    #         eur = eur - btc*change*test_labels[n-1]
    #     elif change >= 0 :
    #         btc = btc + eur/test_labels[n-1]
    #         eur = 0
    #     print("BTC:",btc,"EUR:",eur)
    #     if eur<0 or btc<0:
    #         print("ERROR")
    #         break
    #
    # print("All in: ",(10000/test_labels[0] + 10)*test_labels[-1])
    # print("AI: ",btc*test_labels[-1]+eur)


# dataset = read_csv('Webdata/XBTEUR.csv', header=0, index_col=0)
# for row in dataset:
#     print row
# values = dataset.values
# values = values.astype('float32')
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
#
# print(scaled)
