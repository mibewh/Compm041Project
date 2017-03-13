import csv
from collections import namedtuple
from common import Data



'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

clicks = 0
total = 0
for bid in Data('dataset/validation.csv'):
    if(bid.click=='1'): clicks+=1
    total += 1
print(clicks)
print(total)
