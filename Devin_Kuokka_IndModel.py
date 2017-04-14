import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as mets
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import SelectFpr
from sklearn.linear_model import LogisticRegressionCV, Lasso

trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
testFile = 'dataset/test.csv'
out_file = 'dataset/testing_bidding_price.csv'

ZERO_MULT = 5
STRATEGY_CONST = 80.25
L = 0.000312

dv = DictVectorizer()
fpr = SelectFpr(alpha=0.01)

columns = ['click','payprice','bidprice','bidid', \
           'weekday','hour',\
           'region','city',\
           'useragent','adexchange', 'domain',\
           'slotid', 'slotformat','slotvisibility','slotwidth','slotheight', 'slotprice', 'creative','keypage']

test_columns = columns[3:]

def undersample(data, zeroMult=1):
    click_ind = data[data.click == 1].index
    nonclick_ind = data[data.click == 0].index
    numSamples = zeroMult * len(click_ind)
    sample_ind = np.random.choice(nonclick_ind, numSamples, replace=False)
    nonclick_samples = data.loc[sample_ind]
    click_samples = data.loc[click_ind]
    return pd.concat([nonclick_samples,click_samples], ignore_index=True)

def getFeatures(data, fit=False):
    global dv,fpr
    data = data.drop('bidid', axis=1) # Don't need bidid to find features
    metrics = ['click','payprice','bidprice'] # Bid price must be considered a metric because it is not present in the test data, and therefore cannot be used as a feature in the model
    # Stratify useragent into os and browser
    agent_info = pd.DataFrame(np.array([[item for item in agent.split('_')] for agent in data.useragent]), columns=['os','browser'])
    data = data.drop('useragent',axis=1).join(agent_info)
    # Stratify hours into peak hours (5-12) or not
    peak_info = pd.DataFrame(np.array([1 if 18<=int(hour)<=24 else 0 for hour in data.hour]))
    data = data.drop('hour',axis=1)
    # Combine slotwidth and slotheight into area--maybe more indicative of something?
    area_info = pd.DataFrame(np.array([int(dim['slotwidth'])*int(dim['slotheight']) for i,dim in data.iterrows()]))
    data = data.drop(['slotwidth','slotheight'],axis=1)
    # Stratify os into mobile or not
    mobile_info = pd.DataFrame(np.array([1 if os in ['android','ios'] else 0 for os in data.os]))
    data = data.drop('os',axis=1)
    # Add new features to dataset
    data['peak'] = peak_info
    data['area'] = area_info
    data['mobile'] = mobile_info
    if fit:
        vec = dv.fit_transform(data.drop(metrics, axis=1).to_dict(orient='records'))
    else:
        vec = dv.transform(data.drop(metrics, axis=1).to_dict(orient='records'))
    if fit:
        fpr.fit(vec, data['click'])
    features = fpr.transform(vec)
    return features

def loadData(fileName, train=False, test=False):
    print('Loading data (%s)...' % fileName)
    if not test: df = pd.read_csv(fileName, usecols=columns, dtype={'weekday':object,'hour':object,'region':object, 'city':object, 'domain':object, 'slotid':object})
    else: df = pd.read_csv(fileName, usecols=test_columns, dtype={'weekday':object,'hour':object,'region':object, 'city':object, 'domain':object, 'slotid':object})
    print('Preprocessing (%s)...' % fileName)
    if train: df = undersample(df, zeroMult=ZERO_MULT)
    features = getFeatures(df, fit=train)
    return df, features

def getpredictionORTB1(pCTR, budgetRemaining):
    return (np.sqrt(STRATEGY_CONST / L * pCTR + STRATEGY_CONST**2) - STRATEGY_CONST)

def outputTestResults(model):
    test_df, test_features = loadData(testFile, train=False, test=True)
    guesses = getPredictionsORTB1(model, test_features)
    # Prepare output dataframe and send it to csv file
    out_df = pd.DataFrame({'bidid': test_df['bidid'], 'bidprice': guesses})
    out_df.to_csv(out_file, index=False)

# Generate Models from Training Data
train_df, train_features = loadData(trainFile, train=True)
print('Learning...')
scorer = mets.make_scorer(score_func=mets.f1_score, average='weighted') # creates a callable scorer object
model = LogisticRegressionCV(Cs=1, class_weight='balanced', cv = 15, penalty='l1', scoring=scorer, dual=False, solver='liblinear', refit=True)
model.fit(train_features, train_df.click)
    
#Evalute Model on validation set
print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)
pCTR = model.predict_proba(val_features)[:, 1]

count = 0
pCTR_c = list()
pCTR_nc = list()
for e in val_df['click']:
     if e == 1:
         pCTR_c.append(pCTR[count])
     else:
         pCTR_nc.append(pCTR[count])
     count+=1
pCTR_click = np.asarray(pCTR_c)
pCTR_nonclick = np.asarray(pCTR_nc)
print('Average Click pCTR: %f' % np.average(pCTR_click))
print('Average Nonclick pCTR: %f' %np.average(pCTR_nonclick))
print('Median Click pCTR: %f' %np.median(pCTR_click))
print('Median Nonclick pCTR: %f' % np.median(pCTR_nonclick))

bidsPlaced = 0
numWins = 0
clicks = 0
spent = 0
clicksMissed = 0
for i in range(len(pCTR)):
    bidAmt = getpredictionORTB1(pCTR[i], 25000000-spent)
    bid = val_df.iloc[i]
    if (spent + bidAmt) <= 25000000: #Would not place bid if the bid amount surpasses the budget
        bidsPlaced += 1
        if bidAmt > bid.payprice:
            numWins += 1
            clicks += bid.click
            spent += bid.payprice
        else:
            if bid.click == 1:
                clicksMissed += 1
print('CTR: %f' % (clicks/numWins))
print('Clicks: %d' % (clicks))
print('Spend: %f' % (spent/1000))
print('Average CPM: %f' % (spent / numWins))
print('Average CPC: %f' %(spent/clicks/1000))