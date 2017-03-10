from common import Data, evaluate
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder

# Useful for dealing with categorical features (OneHot) and impressions with missing values
# http://scikit-learn.org/stable/modules/preprocessing.html

imp = Imputer(strategy='most_frequent')
ohe = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7,8], sparse=True)

#Maybe reduce this to just OS, as browser is less relevant?
agents = {'linux_chrome':0, 'linux_firefox':1, 'other_firefox':2, 'windows_safari':3, 'other_other':4, 'windows_other':5,\
        'ios_other':6, 'mac_ie':7, 'ios_safari':8, 'mac_firefox':9, 'linux_safari':10, 'mac_safari':11, 'other_ie':12,\
        'android_firefox':13, 'windows_chrome':14, 'windows_theworld':15, 'android_maxthon':16, 'other_opera':17, 'linux_other':18,\
        'linux_opera':19, 'mac_maxthon':20, 'android_chrome':21, 'mac_other':22, 'android_opera':23, 'android_ie':24, 'windows_ie':25,\
        'windows_maxthon':26, 'linux_ie':27, 'mac_sogou':28, 'android_other':29, 'mac_chrome':30, 'mac_opera':31, 'other_chrome':32,\
        'windows_firefox':33, 'other_safari':34, 'windows_sogou':35, 'android_safari':36, 'android_sogou':37, 'windows_opera':38}
visibility = {'0':0, '1':1, '2':2, '255':3, 'FirstView':4, 'SecondView':5, 'ThirdView':6, 'FourthView':7, 'FifthView':8,\
            'OtherView':9, 'Na':np.nan}

def convertBidArr(bid):
    #Add back adexchange and slotformat? These have some bad data... useragent?
    if bid.slotformat == 'Na': bid = bid._replace(slotformat=np.nan)
    if bid.adexchange == 'null': bid = bid._replace(adexchange=np.nan)
    bid = bid._replace(useragent=agents[bid.useragent], slotvisibility=visibility[bid.slotvisibility])
    # Add back useragent to the return statement AND ohe if you want it in
    return np.array([bid.weekday,bid.hour,bid.region,bid.city,bid.useragent\
                ,bid.advertiser, bid.slotformat, bid.adexchange, bid.slotvisibility\
                ,bid.slotwidth,bid.slotheight,bid.slotprice], dtype=np.float32)

def preprocess(xarr):
    xarr = imp.transform(xarr)
    xarr = ohe.transform(xarr)
    return xarr

def learnModel(trainFileName):
    xdata = []
    ydata = []
    for bid in Data(trainFileName):
        xdata.append(convertBidArr(bid))
        ydata.append(int(bid.click))
    xarr = np.array(xdata)
    print('Preprocessing...')
    xarr = imp.fit_transform(xarr) # Can't just use preprocess here because the imputer needs to be fitted first (as well as transformed)
    ohe.fit(xarr)
    xarr = ohe.transform(xarr)
    yarr = np.array(ydata)
    avgCTR = np.average(yarr)
    print('Learning...')
    lr = linear_model.LogisticRegression(C=1, n_jobs=-1)
    lr.fit(xarr, yarr)
    return lr, avgCTR

def calculateBid(bid, baseBid, model, averageCTR):
    arr = preprocess(convertBidArr(bid).reshape(1,-1)).toarray()
    pCTR = model.predict_proba(arr)[0][1]
    return baseBid * pCTR / averageCTR


baseBid = 200
print('Begin Learning...')
lr, avgCTR = learnModel('dataset/train.csv')
print('Evaluating...')
evaluate('dataset/validation.csv', calculateBid, baseBid, lr, avgCTR)
