import math
from common import evaluate, Data


def calculateBidPrice(trainFileName):
    avg = 0
    bids = 0
    stuff = []
    for bid in Data(trainFileName):
        #Base calculatuion on bidprice or payprice? Payprice probably?, but might want to add 1 or so to make sure its higher?
        #if(row.click == '1'):
        avg += int(bid.payprice)
        bids += 1
    avg /= bids
    return math.ceil(avg) + 1

bidAmount = calculateBidPrice('dataset/train.csv')

def getBidPrice():
    return bidAmount


print('Constant Bid: %d' % bidAmount)
evaluate('dataset/validation.csv', getBidPrice)
