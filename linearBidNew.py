from common import Data, evaluate
from sklearn import linear_model
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html


#This file uses a vectorizer, which allows for lots of simplification with the one-hot encoding
#Instead of imputing on the missing fields, they are set up as their own category
#Seems to run faster, although evaluation still takes forever, with similar success
vec = DictVectorizer()

def convertBidArr(bid):
    return vec.transform(convertBidDict(bid))[0]

def convertBidDict(bid):
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}

def learnModel(trainFileName):
    print('Loading Data...')
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidDict(bid))
        ydata.append(int(bid.click))
    print('Preprocessing...')
    xarr = vec.fit_transform(xdata)
    yarr = np.array(ydata)
    avgCTR = np.average(yarr)
    print('Learning...')
    lr = linear_model.LogisticRegression(C=1, n_jobs=-1)
    lr.fit(xarr, yarr)
    return lr, avgCTR

def calculateBid(bid, baseBid, model, averageCTR):
    arr = convertBidArr(bid).toarray().reshape(1,-1)
    pCTR = model.predict_proba(arr)[0][1]
    return baseBid * pCTR / averageCTR


lr, avgCTR = learnModel('dataset/train.csv')
for baseBid in range(50, 400, 50):
    print('Evaluating (%d)...' % baseBid)
    evaluate('dataset/validation.csv', calculateBid, baseBid, lr, avgCTR)
