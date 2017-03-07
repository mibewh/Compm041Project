import csv
from collections import namedtuple
#from sets import Set


'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

fileName = 'dataset/test.csv'
with open(fileName, 'rt') as f:
    reader = csv.reader(f)
    Row = namedtuple('Row', next(reader)) #Create tuple from header row
    print(Row._fields)
    for row in map(Row._make, reader):
        pass
