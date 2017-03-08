from common import Data, evaluate
from sklearn import datasets, linear_model
import numpy as np

def convertBidArr(bid):
    #Add back adexchange and slotformat? These have some bad data... useragent?
    return [int(bid.weekday),int(bid.hour),int(bid.region),int(bid.city)\
                ,int(bid.slotwidth),int(bid.slotheight),int(bid.slotprice)\
                ,int(bid.advertiser)]

def learnModel(trainFileName):
    xdata = []
    ydata = []
    clicks = []
    for bid in Data(trainFileName):
        xdata.append(convertBidArr(bid))
        ydata.append(int(bid.click))
        clicks.append(int(bid.click))
    avgCTR = np.average(np.array(clicks))

    xarr = np.array(xdata)
    yarr = np.array(ydata)
    regr = linear_model.LinearRegression()
    regr.fit(xarr, yarr)
    return regr.coef_, avgCTR

def calculateBid(bid, baseBid, coeffs, averageCTR):
    pCTR = np.sum(np.array(convertBidArr(bid), dtype='float64') * coeffs)
    return baseBid * pCTR / averageCTR


baseBid = 300
coeffs, avgCTR = learnModel('dataset/train.csv')
evaluate('dataset/validation.csv', calculateBid, baseBid, coeffs, avgCTR)
