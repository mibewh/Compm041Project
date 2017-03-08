from common import evaluate, Data
import numpy as np
import random, math

def calculateUpperBound(trainFileName):
    bids = []
    for bid in Data(trainFileName):
        bids.append(int(bid.payprice))
    arr = np.array(bids)
    return np.average(arr) + np.std(arr) # Upper bound set at 1 standard deviation above mean (?)

upper = calculateUpperBound('dataset/train.csv')

def getBidPrice(bid):
    return upper * random.random()

print('Random Upper Bound: %d' % upper)
evaluate('dataset/validation.csv', getBidPrice)
