import csv
from collections import namedtuple

class Data():
    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
        with open(self.fileName, 'rt') as f:
            reader = csv.reader(f)
            Row = namedtuple('Row', next(reader)) #Create tuple from header row
            for row in map(Row._make, reader):
                yield row

def evaluate(validateFileName, getBidPrice, *args):
    numBids = 0
    wins = 0
    clicks = 0
    cost = 0
    for bid in Data(validateFileName):
        numBids += 1
        if getBidPrice(args) > int(bid.payprice): # and (cost + int(row.payprice)) <= 25000:
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
