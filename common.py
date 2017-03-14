import csv
from collections import namedtuple

class Data:
    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
        with open(self.fileName, 'rt') as f:
            reader = csv.reader(f)
            Row = namedtuple('Row', next(reader)) #Create tuple from header row
            for row in map(Row._make, reader):
                yield row

class OutFile:
    def __init__(self, fileName):
        self.fileName = fileName

    def __enter__(self):
        self.f = open(self.fileName, 'wt')
        self.writer = csv.writer(self.f)
        self.writer.writerow(('bidid','bidrow'))

    def __exit__(self):
        self.f.close()

    def addRow(self, bidid, bidprice):
        self.writer.writerow((bidid,bidrow))

def evaluate(validateFileName, getBidPrice, *args):
    bidsPlaced = 0
    numWins = 0
    clicks = 0
    spent = 0
    for bid in Data(validateFileName):
        bidAmt = getBidPrice(bid, *args)
        if (spent + bidAmt) <= 25000: #Would not place bid if the bid amount surpasses the budget
            bidsPlaced += 1
            if bidAmt > int(bid.payprice):
                numWins += 1
                clicks += int(bid.click)
                spent += int(bid.payprice)

    print('Win proportion: %f' % (numWins / bidsPlaced))
    print('Wins: %d' % numWins)
    print('CTR: %f' % (clicks / numWins)) #Only need to consider the ads we paid for
    print('Conversions: %d' % (clicks))
    print('Spend: %d' % spent)
    print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
    print('Average CPC: %f' % (clicks / spent))

def evaluate_bulk(validateFileName, getBidPrices, *args):
    bids = []
    actual = []
    for bid in Data(validateFileName):
        bids.append(bid)
        actual.append(int(bid.payprice))
    guesses = getBidPrices(bids, *args)
    print(guesses)
    bidsPlaced = 0
    numWins = 0
    clicks = 0
    spent = 0
    for i in range(len(actual)):
        bidAmt = guesses[i]
        payprice = actual[i]
        bid = bids[i]
        if (spent + bidAmt) <= 25000: #Would not place bid if the bid amount surpasses the budget
            bidsPlaced += 1
            if bidAmt > payprice:
                numWins += 1
                clicks += int(bid.click)
                spent += int(bid.payprice)

    print('Win proportion: %f' % (numWins / bidsPlaced))
    print('Wins: %d' % numWins)
    print('CTR: %f' % (clicks / numWins)) #Only need to consider the ads we paid for
    print('Conversions: %d' % (clicks))
    print('Spend: %d' % spent)
    print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
    print('Average CPC: %f' % (clicks / spent))

#Is there a better way to find the optimal constant bid? Also, do I just bid until I run out of money / is the budget in the doc right?
#One standard deviation above mean to find the random upper parameter?
