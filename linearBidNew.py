from common import Data, evaluate, evaluate_bulk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import RandomUnderSampler

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html

cacheFile = 'dataset/cacheFile'
#This file uses a vectorizer, which allows for lots of simplification with the one-hot encoding
#Instead of imputing on the missing fields, they are set up as their own category
#Seems to run faster, although evaluation still takes forever, with similar success
vec = DictVectorizer()
kbest = SelectKBest(chi2, k=10)

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
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}

def undersample(xdata, yarr):
    numOnes = np.sum(yarr)
    inds = np.arange(0, len(xdata)).reshape(-1,1)
    rus = RandomUnderSampler()
    inds, yarr = rus.fit_sample(inds, yarr)
    xarr = np.array(xdata)
    xdata = xarr[inds].reshape(-1).tolist()
    return xdata, yarr

def learnModel(trainFileName, fromCache=False):
    global vec, kbest
    if(fromCache):
        print('Loading from file...')
        #model, vec, avgCTR = pickle.load(open(cacheFile, 'rb')) WILL NOT LOAD KBEST
        return model, avgCTR
    print('Loading Data...')
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidDict(bid))
        ydata.append(int(bid.click))
    print('Preprocessing...')
    yarr = np.array(ydata)
    xdata, yarr = undersample(xdata, yarr)
    xarr = vec.fit_transform(xdata)
    xarr = kbest.fit_transform(xarr, yarr) # Limit to k best features
    avgCTR = np.average(yarr)
    print('Learning...')
    # model = linear_model.LogisticRegression()
    model = RandomForestClassifier(n_estimators=10, n_jobs=-1, class_weight='balanced')
    model.fit(xarr, yarr)
    # pickle.dump((model,vec,avgCTR), open(cacheFile, 'wb'))
    return model, avgCTR

def calculateBid(bid, baseBid, model, averageCTR):
    arr = convertBidArr(bid).toarray().reshape(1,-1)
    pCTR = model.predict_proba(arr)[0][1]
    return baseBid * pCTR / averageCTR

def calculateBids(bids, baseBid, model, averageCTR):
    arr = convertBidArrBulk(bids)
    probs = model.predict_proba(arr)
    pCTR = probs[:, 1]
    print(pCTR)
    return baseBid * pCTR / averageCTR

model, avgCTR = learnModel('dataset/train.csv', fromCache=False)
baseBid = 150
print('Evaluating (%f)...' % baseBid)
evaluate_bulk('dataset/validation.csv', calculateBids, baseBid, model, avgCTR)
# for baseBid in range(50, 400, 50):
#     print('Evaluating (%d)...' % baseBid)
#     evaluate('dataset/validation.csv', calculateBid, baseBid, model, avgCTR)
