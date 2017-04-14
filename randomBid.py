from common import evaluate, Data
import numpy as np
import random, math


upper = 200

def getBidPrice(bid):
    return upper * random.random()

print('Random Upper Bound: %d' % upper)
evaluate('dataset/validation.csv', getBidPrice)
