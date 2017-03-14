from common import Data, evaluate
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
#from imblearn.under_sampling import RandomUnderSampler

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html

cacheFile = 'dataset/cacheFile'
#This file uses a vectorizer, which allows for lots of simplification with the one-hot encoding
#Instead of imputing on the missing fields, they are set up as their own category
#Seems to run faster, although evaluation still takes forever, with similar success
vec = DictVectorizer()
kbest = SelectKBest(chi2, k=3)

def convertBidArr(bid):
    arr = vec.transform(convertBidDict(bid))[0]
    arr = kbest.transform(arr)
    return arr

def convertBidDict(bid):
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}


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
    xarr = vec.fit_transform(xdata)
    yarr = np.array(ydata)
    #rus = RandomUnderSampler()
    #xarr, yarr = rus.fit_sample(xarr, yarr)
    xarr = kbest.fit_transform(xarr, yarr) # Limit to k best features
    avgCTR = np.average(yarr)
    print('Learning...')
    # model = linear_model.LogisticRegression(C=1, n_jobs=-1)
    model = RandomForestClassifier(n_estimators=10, oob_score=True, n_jobs=-1,class_weight='balanced')
    model.fit(xarr, yarr)
    #print (model.decision_path(xarr))
    print ('oob function:')
    print(model.oob_decision_function_)
    print ('oob score:')
    print(model.oob_score_)
    print('feature imporances:')
    print (model.feature_importances_)
    
    pickle.dump((model,vec,avgCTR), open(cacheFile, 'wb'))
    return model, avgCTR

def calculateBid(bid, baseBid, model, averageCTR):
    arr = convertBidArr(bid).toarray().reshape(1,-1)
    pCTR = model.predict_proba(arr)[0][1]
    return baseBid * pCTR / averageCTR


model, avgCTR = learnModel('dataset/train.csv', fromCache=False)
baseBid = 150
print('Evaluating (%d)...' % baseBid)
evaluate('dataset/validation.csv', calculateBid, baseBid, model, avgCTR)
# for baseBid in range(50, 400, 50):
#     print('Evaluating (%d)...' % baseBid)
#     evaluate('dataset/validation.csv', calculateBid, baseBid, model, avgCTR)
