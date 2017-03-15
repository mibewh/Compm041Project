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
    arr = kbest.transform(arr)
    return arr

def convertBidArrBulk(bids):
    arr = vec.transform([convertBidDict(bid) for bid in bids])
    arr = kbest.transform(arr)
    return arr

def convertBidDict(bid):
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice), 'bidprice':int(bid.bidprice)}

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

def learnModel(xdata, ydata):
    print('Preprocessing...')
    yarr = np.array(ydata)
    xdata, yarr = undersample(xdata, yarr)
    xarr = vec.fit_transform(xdata)
    xarr = kbest.fit_transform(xarr, yarr) # Limit to k best features
    avgCTR = np.average(yarr)
    print('Learning...')
    model = linear_model.LogisticRegression(penalty = 'l2', class_weight='balanced', n_jobs = -1)
    model.fit(xarr, yarr)
    score = model.score(xarr,yarr)
    return model, avgCTR, score

def calculateBids(bids, model, averageCTR):
    arr = convertBidArrBulk(bids)
    probs = model.predict_proba(arr)
    pCTR = probs[:, 1]
    return pCTR / averageCTR

xaxis = []
values = []
baseBid = 50
#for baseBid in np.arange(3,7,0.1):
kbest = SelectKBest(chi2, k=20)
xdata, ydata = loadData('dataset/train.csv')
model, avgCTR, score = learnModel(xdata,ydata)
xaxis.append(baseBid)
print('Evaluating (%f)...' % baseBid)
values.append(evaluate_bulk('dataset/validation.csv', baseBid, calculateBids, model, avgCTR))
print('Score: ', score)
#print(xaxis)
#print(values)
#plt.plot(xaxis,values,'ro')
#plt.axis([1.5,10.5,0,227])
#plt.show()
