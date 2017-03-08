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

def evaluate(validateFileName, getBidPrice):
    numBids = 0
    wins = 0
    clicks = 0
    cost = 0
    for bid in Data(validateFileName):
        numBids += 1
        if getBidPrice() > int(bid.payprice): # and (cost + int(row.payprice)) <= 25000:
            wins += 1
            clicks += int(bid.click)
            cost += int(bid.payprice)

    print('Win percentage: %f' % (wins / numBids))
    print('CTR: %f' % (clicks / numBids))
    print('Conversions: %d' % (clicks))
    print('Whats the difference between CTR and CVR???')
    print('Spend: %d' % cost)
    print('Average CPM???')
    print('Average CPC: %f' % (clicks / cost))




#FOR LATER: How do I calculate pCTR? Is it found on a bid-by-bid basis, or for the whole set, like the average?
#Is there a better way to find the optimal constant bid? Also, do I just bid until I run out of money / is the budget in the doc right?
#One standard deviation above mean to find the random upper parameter?
#CVR??? CPM???
