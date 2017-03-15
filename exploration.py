import csv
from collections import namedtuple
from common import Data



'''
'click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain',
'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
'payprice', 'keypage', 'advertiser', 'usertag'
'''

'''
{'ios': 2, 'android': 342, 'windows': 1514, 'other': 2, 'linux': 3, 'mac': 171}
{'firefox': 15, 'other': 55, 'chrome': 419, 'opera': 2, 'maxthon': 1, 'safari': 469, 'theworld': 6, 'ie': 1067}
'''

# os = {'ios':0, 'android':0, 'mac':0, 'windows':0, 'linux':0, 'other':0}
# browser = {'theworld':0, 'maxthon':0, 'chrome':0, 'firefox':0, 'safari':0, 'opera':0, 'ie':0, 'other':0}
# for bid in Data('dataset/validation.csv'):
#     if bid.click=='1':
#         a = bid.useragent.split('_')
#         os[a[0]] += 1
#         browser[a[1]] += 1
#
# print(os)
# print(browser)

clicks = 0
for bid in Data('dataset/validation.csv'):
    if bid.click=='1':
        if int(bid.payprice) < 10: clicks += 1
print(clicks)
