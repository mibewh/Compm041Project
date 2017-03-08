import csv
from collections import namedtuple
from common import Data



'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

trainAdvertisers = set()
for bid in Data('dataset/train.csv'):
    trainAdvertisers.add(bid.advertiser)
testAdvertisers = set()
for bid in Data('dataset/test.csv'):
    testAdvertisers.add(bid.advertiser)

for a in trainAdvertisers:
    print(a)
print()
for a in testAdvertisers:
    print(a)
