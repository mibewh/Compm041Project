import math
from Common import evaluate, Data


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


print(bidAmount)
evaluate('dataset/validation.csv', getBidPrice, 1, 2)


#FOR LATER: How do I calculate pCTR? Is it found on a bid-by-bid basis, or for the whole set, like the average?
#Is there a better way to find the optimal constant bid? Also, do I just bid until I run out of money / is the budget in the doc right?
#One standard deviation above mean to find the random upper parameter?
#CVR??? CPM???
