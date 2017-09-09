from urllib import request
import os
import csv
import pprint
import re
import pandas as pd
import glob

class CSVData:

    PATH = os.path.dirname(os.path.realpath(__file__))

    csv_urls = {
        "market-cap" : "https://api.blockchain.info/charts/market-cap?timespan=180days&format=csv",
        "trade-volume" : "https://api.blockchain.info/charts/trade-volume?timespan=180days&format=csv",
        "avg-block-size" : "https://api.blockchain.info/charts/avg-block-size?timespan=180days&format=csv",
        "hash-rate" : "https://api.blockchain.info/charts/hash-rate?timespan=180days&format=csv",
        "median-confirmation-time" : "https://api.blockchain.info/charts/median-confirmation-time?timespan=180days&format=csv",
        "mempool-count" : "https://api.blockchain.info/charts/mempool-count?timespan=180days&format=csv",
        "mempool-size" : "https://api.blockchain.info/charts/mempool-size?timespan=180days&format=csv",
        "miners-revenue" : "https://api.blockchain.info/charts/miners-revenue?timespan=180days&format=csv",
        "n-transactions-excluding-popular" : "https://api.blockchain.info/charts/n-transactions-excluding-popular?timespan=180days&format=csv",
        "market-price" : "https://api.blockchain.info/charts/market-price?timespan=180days&format=csv",
    }

    def __init__(self):
        self.data_dict = {}



    def updateData(self):
        for graph in self.csv_urls:
            request.urlretrieve(self.csv_urls[graph], self.PATH+"/"+graph+'.csv')

    def generateCSV(self):
        self._generateDict()
        self._dictToCSV()
        self._gatherDays()
        self._splitCSV(file="features")
        self._splitCSV(file="labels")
        self._combineCSVs("labels_test", "features_test", "test")
        self._combineCSVs("labels_train", "features_train", "train")

    def _generateDict(self):
        newdata = {}

        with open(self.PATH + '/mempool-size.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                newdata[row[0]] = {'mempool-size': row[1][9:]}

        for graph in self.csv_urls:
            with open(self.PATH + '/' + graph + ".csv", newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in spamreader:
                    newdata[row[0]][graph] =  row[1][9:]

        self.data_dict = newdata


    def _dictToCSV(self, label="market-price"):
        with open(self.PATH + '/feature_data.csv', 'w', newline='') as feature_csv, open(self.PATH + '/label_data.csv', 'w', newline='') as label_csv:
            featurewriter = csv.writer(feature_csv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labelwriter = csv.writer(label_csv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for date in self.data_dict:
                rowstr = ""
                for n, row in enumerate(self.data_dict[date]):
                    if row != label:
                        if n is not 0:
                            rowstr = rowstr + ","
                        rowstr = rowstr + self.data_dict[date][row]
                if len(re.findall(',', rowstr)) is len(self.csv_urls)-2:
                    featurewriter.writerow([rowstr])
                    labelwriter.writerow([self.data_dict[date][label]])
                else:
                    print("not enough data: "+str(len(re.findall(',', rowstr))) + " != " +str(len(self.csv_urls)-2))


    def _gatherDays(self, days=10):
        with open(self.PATH + '/feature_data.csv', 'r', newline='') as features_old, open(self.PATH + '/features.csv', 'w', newline='') as features_new:
            spamreader = csv.reader(features_old, delimiter=' ', quotechar='|')
            spamwriter = csv.writer(features_new, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            rowlist = []
            newlist = []
            for n,row in enumerate(spamreader):
                rowlist.append(row[0])
                if n >= days-1: # and n < sum(1 for item in spamreader)-2: # and n < len(list(spamreader))-1:
                    rowstr = ','.join(rowlist)
                    newlist.append([rowstr])
                    rowlist.pop(0)
            newlist = newlist[:-1]
            for row in newlist:
                spamwriter.writerow(row)



        with open(self.PATH + '/label_data.csv', 'r', newline='') as labels_old, open(self.PATH + '/labels.csv', 'w', newline='') as labels_new:
            spamreader = csv.reader(labels_old, delimiter=' ', quotechar='|')
            spamwriter = csv.writer(labels_new, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n,row in enumerate(spamreader):
                if n >= days: # +1 to predict next day
                    spamwriter.writerow([(float(row[0])-float(lastday[0]))/float(lastday[0])*100]) #([100*(float(row[0])/float(lastday[0])-1)]) 
                lastday = row

    def _splitCSV(self, line=34, file="features"):
        with open(self.PATH + '/'+file+'.csv', 'r', newline='') as oldfile, open(self.PATH + '/'+file+'_test.csv', 'w', newline='') as test, open(self.PATH + '/'+file+'_train.csv', 'w', newline='') as train:
            spamreader = csv.reader(oldfile, delimiter=' ', quotechar='|')
            testwriter = csv.writer(test, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            trainwriter = csv.writer(train, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n,row in enumerate(spamreader):
                if n<=33:
                    testwriter.writerow(row)
                else:
                    trainwriter.writerow(row)


    def _combineCSVs(self, file1, file2, output):
        dfs = {}
        df1 = pd.read_csv(self.PATH + '/'+file1+'.csv', header=None) #, names=["trade-volume","avg-block-size","hash-rate","median-confirmation-time","mempool-count","mempool-size","miners-revenue","n-transactions-excluding-popular"])
        df2 = pd.read_csv(self.PATH + '/'+file2+'.csv', header=None) #, names=['market-price'])
        bigdf = pd.concat([df2, df1], axis=1,join_axes=[df2.index])
        bigdf.to_csv(self.PATH + '/'+output + '.csv', header=False, index=False)
