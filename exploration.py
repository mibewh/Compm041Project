import csv
from collections import namedtuple
from common import Data



'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

vis = set()
vis2 = set()
vis3 = set()
for bid in Data('dataset/train.csv'):
    vis.add(bid.slotvisibility)
for bid in Data('dataset/test.csv'):
    vis2.add(bid.slotvisibility)
for bid in Data('dataset/validation.csv'):
    vis3.add(bid.slotvisibility)
print(vis2.issubset(vis))
print(vis3.issubset(vis))
