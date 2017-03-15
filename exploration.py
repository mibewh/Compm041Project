import csv
from collections import namedtuple
from logisticCommon import Data
import numpy as np
from scipy.stats import pearsonr
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

def convertBidDict(bid):
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}

def undersample(xdata, yarr):
    numOnes = np.sum(yarr)
    inds = np.arange(0, len(xdata)).reshape(-1,1)
    rus = RandomUnderSampler() # #1/#0
    inds, yarr = rus.fit_sample(inds, yarr)
    xarr = np.array(xdata)
    xdata = xarr[inds].reshape(-1).tolist()
    return xdata, yarr

def loadData(trainFileName):
    global vec, kbest
    print('Loading Data...')
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidDict(bid))
        ydata.append(int(bid.click))
    return xdata, ydata



'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

clicks = 0
total = 0
for bid in Data('dataset/validation.csv'):
    if(bid.click=='1'): clicks+=1
    total += 1
print(clicks)
print(total)
np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))
print "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size))