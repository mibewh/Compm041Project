import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mets


trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
testFile = 'dataset/test.csv'
out_file = 'dataset/testing_bidding_price.csv'

ZERO_MULT = 30
ONE_WEIGHT = 20
MODEL_CONST = 200
L = 5e-6

dv = DictVectorizer()

columns = ['click','payprice','bidprice','bidid','weekday','hour','region','useragent',\
            'slotformat','adexchange','slotvisibility',\
            'slotwidth','slotheight','slotprice','creative','keypage', 'advertiser', 'city']
test_columns = columns[3:]
def undersample(data, zeroMult=1):
    click_ind = data[data.click == 1].index
    nonclick_ind = data[data.click == 0].index
    numSamples = zeroMult * len(click_ind)
    sample_ind = np.random.choice(nonclick_ind, numSamples, replace=False)
    nonclick_samples = data.loc[sample_ind]
    click_samples = data.loc[click_ind]
    return pd.concat([nonclick_samples,click_samples], ignore_index=True)

def getFeatures(data, fit=False, test=False):
    global dv
    data = data.drop('bidid', axis=1) # Don't need bidid to find features
    metrics = ['click','payprice','bidprice'] # Bid price must be considered a metric because it is not present in the test data, and therefore cannot be used as a feature in the model
    if not test: data = data.drop(metrics, axis=1)
    agent_info = pd.DataFrame(np.array([[item for item in agent.split('_')] for agent in data.useragent]), columns=['os','browser'])
    data = data.drop('useragent',axis=1).join(agent_info)
    weekend_info = pd.DataFrame(np.array([1 if 2<=int(weekday)<=4 else 0 for weekday in data.weekday]))
    peak_info = pd.DataFrame(np.array([1 if 16<=int(hour)<=19 else 0 for hour in data.hour]))
    area_info = pd.DataFrame(np.array([int(dim['slotwidth'])*int(dim['slotheight']) for i,dim in data.iterrows()]))
    mobile_info = pd.DataFrame(np.array([1 if os in ['android','ios'] else 0 for os in data.os]))
    data = data.drop(['weekday','hour','slotwidth','slotheight'],axis=1)
    data['weekend'] = weekend_info
    data['peak'] = peak_info
    data['area'] = area_info
    data['mobile'] = mobile_info
    if fit:
        vec = dv.fit_transform(data.to_dict(orient='records'))
    else:
        vec = dv.transform(data.to_dict(orient='records'))
    features = normalize(vec, axis=1)
    return features

def loadData(fileName, train=False, test=False):
    print('Loading data (%s)...' % fileName)
    if not test: df = pd.read_csv(fileName, usecols=columns, dtype={'weekday':object,'hour':object,'region':object,'city':object,'advertiser':object})
    else: df = pd.read_csv(fileName, usecols=test_columns, dtype={'weekday':object,'hour':object,'region':object,'city':object,'advertiser':object})
    print('Preprocessing (%s)...' % fileName)
    if train: df = undersample(df, zeroMult=ZERO_MULT)
    features = getFeatures(df, fit=train, test=test)
    return df, features

def getPredictionORTB1(pCTR):
    return np.sqrt(MODEL_CONST / L * pCTR + MODEL_CONST**2) - MODEL_CONST #* mult

def outputTestResults(model):
    print('Outputting test results (%s)...' % out_file)
    test_df, test_features = loadData(testFile, train=False, test=True)
    pCTRs = model.predict_proba(test_features)[:, 1]
    guesses = []
    for pCTR in pCTRs:
        guesses.append(getPredictionORTB1(pCTR))
    guesses = np.array(guesses)
    out_df = pd.DataFrame({'bidid': test_df['bidid'], 'bidprice': guesses})
    out_df.to_csv(out_file, index=False)



train_df, train_features = loadData(trainFile, train=True)
avgCTR = train_df['click'].mean()
print('Learning...')
model = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', class_weight={0:1,1:ONE_WEIGHT}, n_jobs=-1) #increasing the 1 weight increases both true and false positives
model.fit(train_features, train_df.click)
outputTestResults(model)
# sys.exit()

print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)
pCTR = model.predict_proba(val_features)[:, 1]

bidsPlaced = 0
numWins = 0
clicks = 0
spent = 0
clicksMissed = 0
for i in range(len(pCTR)):
    bidAmt = getPredictionORTB1(pCTR[i])
    bid = val_df.iloc[i]
    if (spent + bidAmt) <= 25000000:
        bidsPlaced += 1
        if bidAmt > bid.payprice:
            numWins += 1
            clicks += bid.click
            spent += bid.payprice
        else:
            if bid.click == 1:
                clicksMissed += 1

print('Bids Placed: %d' % bidsPlaced)
print('Wins: %d' % numWins)
print('Clicks Missed: %d' % clicksMissed)
print('CTR: %f' % (clicks / numWins)) #Only need to consider the ads we paid for
print('Conversions: %d' % (clicks))
print('Spend: %d' % (spent/1000))
print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
print('Average CPC: %f' % (spent / clicks / 1000))
