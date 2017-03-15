from common import Data, evaluate, evaluate_bulk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html

cacheFile = 'dataset/cacheFile'
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
    rus = RandomUnderSampler()
    inds, yarr = rus.fit_sample(inds, yarr)
    xarr = np.array(xdata)
    xdata = xarr[inds].reshape(-1).tolist()
    return xdata, yarr

def loadData(trainFileName, fromCache=False):
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
    return xdata,ydata

def prepareData(xdata, ydata):
    print('Preprocessing...')
    yarr = np.array(ydata)
    avgCTR = np.average(yarr)
    xdata, yarr = undersample(xdata, yarr)
    xarr = vec.fit_transform(xdata)
    baseModel = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight='balanced')
    baseModel.fit(xarr, yarr)
    importances = baseModel.feature_importances_
    xnew = SelectFromModel(baseModel, prefit=True).transform(xarr)
    return xnew, yarr, avgCTR
    #xarr = kbest.fit_transform(xarr, yarr) # Limit to k best features

def learnModel(xdata, ydata):
    print('Learning...')
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight='balanced')
    model.fit(xdata, ydata)
    #print (X_new.shape)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(xdata.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances 2")
    plt.bar(range(xdata.shape[1]), importances[indices],color="r", align="center")
    #plt.xticks(range(xarr.shape[1]), indices)
    plt.xlim([-1, xdata.shape[1]])
    plt.show()
    
    return model

def calculateBids(bids, baseBid, model, averageCTR):
    arr = convertBidArrBulk(bids)
    probs = model.predict_proba(arr)
    pCTR = probs[:, 1]
    #print('pCTR:')
    #print(pCTR)
    return baseBid * pCTR / averageCTR, pCTR

pCTRs = []
xs = []
baseBid = 100
xdata, ydata = loadData('dataset/train.csv', fromCache=False)
xtrans, ytrans, avgCTR = prepareData(xdata,ydata)
#for x in range (10,20):
#xs.append(x)

#kbest = SelectKBest(chi2, k='all')
model = learnModel(xtrans,ytrans)
print('Evaluating (%f)...' % baseBid)
evaluate_bulk('dataset/validation.csv', calculateBids, baseBid, model, avgCTR)

#plt.plot(xs, pCTRs, 'ro' )
#plt.axis([0,110, 0, 1])
#plt.show()
# for baseBid in range(50, 400, 50):
#     print('Evaluating (%d)...' % baseBid)
#     evaluate('dataset/validation.csv', calculateBid, baseBid, model, avgCTR)
