import math
from common import evaluate, Data

bidAmount = 110

def getBidPrice(bid):
    return bidAmount


print('Constant Bid: %d' % bidAmount)
evaluate('dataset/validation.csv', getBidPrice)
