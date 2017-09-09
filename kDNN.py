from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
import pandas as pd
import numpy as np
import os, pprint, csv

PATH = os.path.dirname(os.path.realpath(__file__))

test = pd.DataFrame(pd.read_csv(PATH + '/Data/test.csv', header=None).values.astype('float32'))
train = pd.DataFrame(pd.read_csv(PATH + '/Data/train.csv', header=None).values.astype('float32')).sample(frac=1).reset_index(drop=True)

x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1:]
x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# print(x_train)
# print(y_train)
# print(x_test)
print(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential([
    Dense(100, input_shape=(90,)),
    Activation('tanh'),
    Dense(30),
    Activation('tanh'),
    Dense(1)
])

# model.add(SimpleRNN(100, activation='tanh'))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10000, batch_size=32)

test_results = model.predict(x_test, batch_size=32, verbose=0)
print(test_results,y_test)
#print(pd.concat([test_results, y_test], axis=1,join_axes=[test_results.index]))
values = []
for n,row in enumerate(test_results):
    print(row[0]-y_test[n][0])
    values.append((row[0]<0) and (y_test[n][0]<0) or (row[0]>0) and (y_test[n][0]>0))

print(values)
