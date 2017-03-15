import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import Imputer, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn_pandas import DataFrameMapper

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
BALANCED = None #'balanced'
K_FEATS = 100
ZERO_MULT = 15
C = 1
BASE_BID = 5
MODEL_CONST = 1e4

dv = DictVectorizer()
kbest = SelectKBest(chi2, k=K_FEATS)
columns = ['click','bidprice','payprice','weekday','hour','region','city','useragent',\
            'advertiser','slotformat','adexchange','slotvisibility',\
            'slotwidth','slotheight','slotprice']
def undersample(data, zeroMult=1):
    click_ind = data[data.click == 1].index
    nonclick_ind = data[data.click == 0].index
    numSamples = zeroMult * len(click_ind)
    sample_ind = np.random.choice(nonclick_ind, numSamples, replace=False)
    nonclick_samples = data.loc[sample_ind]
    click_samples = data.loc[click_ind]
    return pd.concat([nonclick_samples,click_samples], ignore_index=True)

def getFeatures(data, fit=False):
    global dv, kbest
    metrics = ['click','payprice']
    # Stratify useragent into os and browser
    agent_info = pd.DataFrame(np.array([[item for item in agent.split('_')] for agent in data.useragent]), columns=['os','browser'])
    data = data.drop('useragent',axis=1).join(agent_info)
    if fit:
        vec_df = pd.DataFrame(dv.fit_transform(data.to_dict(orient='records')).toarray())
    else:
        vec_df = pd.DataFrame(dv.transform(data.to_dict(orient='records')).toarray())
    vec_df.columns = dv.get_feature_names()
    vec_df.index = data.index
    #Reduce the feature space manually? As in get reduce to a bit for each browser, device type
    # Do one hot encoding for weekday?, hour?, region, city*,
    # Hour, weekday don't have to be categorical, does they?
    kbest_mapper = DataFrameMapper([(vec_df.drop(metrics,axis=1).columns, kbest)])
    if fit:
        features = kbest_mapper.fit_transform(vec_df.drop(metrics,axis=1), data['click'])
    else:
        features = kbest_mapper.transform(vec_df.drop(metrics,axis=1))
    features_df = pd.DataFrame(features, columns=vec_df.columns[kbest.get_support(indices=True)])
    print(features_df.columns)
    return features

def loadData(fileName, train=False):
    print('Loading data (%s)...' % fileName)
    df = pd.read_csv(fileName, usecols=columns, dtype={'weekday':object,'hour':object,'region':object,'city':object,'advertiser':object})
    print('Preprocessing (%s)...' % fileName)
    if train: df = undersample(df, zeroMult=ZERO_MULT)
    features = getFeatures(df, fit=train)
    return df, features

def getPredictions(model, features, avgCTR):
    pCTR = model.predict_proba(features)[:, 1]
    print(pCTR)
    return BASE_BID * pCTR / avgCTR

def getPredictionsORTB1(model, features, avgCTR):
    pCTR = model.predict_proba(features)[:, 1]
    return np.sqrt(MODEL_CONST / 0.007 * pCTR + MODEL_CONST**2) - MODEL_CONST

train_df, train_features = loadData(trainFile, train=True)
avgCTR = train_df['click'].mean() # Average ctr of reduced (if performed) data set
print('Learning...')
model = linear_model.LogisticRegression(n_jobs=-1, C=C, class_weight=BALANCED)
model.fit(train_features, train_df.click)

#Display model accuracy metrics

#Evalute Model on validation set
print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)
# guesses = getPredictions(model, val_features, avgCTR)
guesses = getPredictionsORTB1(model, val_features, avgCTR)
print(guesses)
bidsPlaced = 0
numWins = 0
clicks = 0
spent = 0
clicksMissed = 0
for i in range(len(guesses)):
    bidAmt = guesses[i]
    bid = val_df.iloc[i]
    if (spent + bidAmt) <= 25000: #Would not place bid if the bid amount surpasses the budget
        bidsPlaced += 1
        if bidAmt > bid.payprice:
            numWins += 1
            clicks += bid.click
            spent += bid.payprice
            if bid.click==1: print(bid)
        else:
            if bid.click == 1:
                clicksMissed += 1

print('Bids Placed: %d' % bidsPlaced)
print('Wins: %d' % numWins)
print('Clicks Missed: %d' % clicksMissed)
print('CTR: %f' % (clicks / numWins)) #Only need to consider the ads we paid for
print('Conversions: %d' % (clicks))
print('Spend: %d' % spent)
print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
print('Average CPC: %f' % (clicks / spent))
