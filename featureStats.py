from logisticCommon import Data, evaluate, evaluate_bulk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html

#This file uses a vectorizer, which allows for lots of simplification with the one-hot encoding
#Instead of imputing on the missing fields, they are set up as their own category
#Seems to run faster, although evaluation still takes forever, with similar success
vec = DictVectorizer()

def convertBidArr(bid):
    arr = vec.transform(convertBidDict(bid))[0]
    #arr = kbest.transform(arr)
    return arr

def convertBidArrBulk(bids):
    arr = vec.transform([convertBidDict(bid) for bid in bids])
    #arr = kbest.transform(arr)
    return arr

def removeFeatures():

def convertBidDict(bid):
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}

def loadData(trainFileName):
    print('Loading Data...')
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidDict(bid))
        ydata.append(int(bid.click))
    return xdata, ydata


xaxis = []
values = []
#kbest = SelectKBest(chi2, k=20)
xdata, ydata = loadData('dataset/train.csv')
#baseBid = 6.5 
for baseBid in np.arange(0.01,10.1,0.01):
    model, avgCTR, score = learnModel(xdata,ydata)
    print('Evaluating (%f)...' % baseBid)
    values.append(evaluate_bulk('dataset/validation.csv', calculateBids, baseBid, model, avgCTR))
    print('Score: ', score)
print(xaxis)
print(values)
plt.plot(xaxis,values,'ro')
plt.axis([0.01,10.1,0,227])
plt.show()
