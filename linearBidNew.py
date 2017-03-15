import pickle
import numpy as np
from common import Data, evaluate, evaluate_bulk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler

SAMPLE_RATIO = 1.0 / 1.0 # lesser / greater
BALANCED = 'balanced' # 'balanced' or None
C = 1e10 # Inverse regularization constant for logistic regression
NUM_TREES = 20 # Number of estimator trees for forest classifier
K_FEATS = 50 # Number of best features to use
BASE_BID = 150
CUTOFF_RATIO = 0.5 # If pCTR / avgCTR is less than this, do not bid

vec = DictVectorizer()
kbest = SelectKBest(chi2, k=K_FEATS)

def convertBidArr(bid):
    arr = vec.transform(convertBidDict(bid))[0]
    arr = kbest.transform(arr)
    return arr

def convertBidArrBulk(bids):
    arr = vec.transform([convertBidDict(bid) for bid in bids])
    arr = kbest.transform(arr)
    return arr

def convertBidDict(bid):
    #'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)
    return {'weekday':bid.weekday, 'hour':bid.hour, 'region':bid.region, 'city':bid.city,\
            'useragent':bid.useragent, 'advertiser':bid.advertiser, 'slotformat':bid.slotformat,\
            'adexchange':bid.adexchange, 'slotvisibility':bid.slotvisibility,\
            'slotwidth':int(bid.slotwidth), 'slotheight':int(bid.slotheight), 'slotprice':int(bid.slotprice)}

def undersample(xdata, yarr):
    inds = np.arange(len(xdata)).reshape(-1,1)
    rus = RandomUnderSampler(ratio=SAMPLE_RATIO)
    inds, yarr = rus.fit_sample(inds, yarr)
    xarr = np.array(xdata)
    xdata = xarr[inds].reshape(-1).tolist()
    return xdata, yarr

def learnModel(trainFileName):
    global vec, kbest
    print('Loading Data...')
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidDict(bid))
        ydata.append(bool(int(bid.click)))
    print('Preprocessing...')
    yarr = np.array(ydata, dtype=np.bool)
    # xdata, yarr = undersample(xdata, yarr)
    xarr = vec.fit_transform(xdata)
    xarr = kbest.fit_transform(xarr, yarr) # Limit to k best features
    avgCTR = np.average(yarr)
    print('Learning...')
    model = linear_model.LogisticRegression(n_jobs=-1, class_weight=BALANCED, C=C)
    # model = RandomForestClassifier(n_estimators=NUM_TREES, n_jobs=-1, class_weight=BALANCED)
    model.fit(xarr, yarr)
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
    a = pCTR / 0.5 #AverageCTR for when not balanced?
    # a[a < CUTOFF_RATIO] = 0
    return baseBid * a

def diagnostic(validateFileName, model, avgCTR):
    bids = []
    actual = []
    for bid in Data(validateFileName):
        bids.append(bid)
        actual.append(bool(int(bid.click)))
    X = convertBidArrBulk(bids)
    Y = np.array(actual)
    pCTR = model.predict_proba(X)[:,1]
    inds = np.argwhere(Y.flatten()==True).flatten()
    print(np.average(pCTR.flatten()[inds]/avgCTR))
    print('Accuracy: %f' % model.score(X, Y))
    print('Avg. Precision: %f' % average_precision_score(Y, pCTR))

model, avgCTR = learnModel('dataset/train.csv')

print('Evaluating (%f)...' % BASE_BID)
diagnostic('dataset/validation.csv', model, avgCTR)
evaluate_bulk('dataset/validation.csv', calculateBids, BASE_BID, model, avgCTR)

# for baseBid in range(50, 400, 50):
#     print('Evaluating (%d)...' % baseBid)
#     evaluate('dataset/validation.csv', calculateBid, baseBid, model, avgCTR)
