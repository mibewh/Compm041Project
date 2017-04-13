import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, svm
from sklearn.preprocessing import Imputer, OneHotEncoder, FunctionTransformer, normalize, PolynomialFeatures
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, mutual_info_classif
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as mets


trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
testFile = 'dataset/test.csv'
out_file = 'dataset/testing_bidding_price.csv'
BALANCED = 'balanced' #'balanced'
K_FEATS = 'all' #300 makes really good predicts too
ZERO_MULT = 1
C = 1e6
BASE_BID = 50

# MODEL_CONST = 45
# L = (25000/300000)*1e-3
MODEL_CONST = 120
L = 2e-3

dv = DictVectorizer()
#For Kbest feature selection
kbest = SelectKBest(chi2, k=K_FEATS)

stratifier = RandomForestClassifier(n_estimators=10)

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

def getFeatures(data, fit=False):
    # suggests that the bid price has the biggest influence on the optimal bid, all other categorical data
    # don't necessarily have a clear influence (but is likely still useful)
    # "the bid price is the key factor influencing the campaign's winning rate in its auctions"
    global dv, kbest, lr
    data = data.drop('bidid', axis=1) # Don't need bidid to find features
    metrics = ['click','payprice','bidprice'] # Bid price must be considered a metric because it is not present in the test data, and therefore cannot be used as a feature in the model
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
    # Keep it sparse!
    if fit:
        vec = dv.fit_transform(data.drop(metrics, axis=1).to_dict(orient='records'))
    else:
        vec = dv.transform(data.drop(metrics, axis=1).to_dict(orient='records'))

    vec = normalize(vec, axis=1)
    if fit:
        kbest.fit(vec, data['click'])


    features = kbest.transform(vec)
    # features = vec

    return features


def loadData(fileName, train=False, test=False):
    print('Loading data (%s)...' % fileName)
    if not test: df = pd.read_csv(fileName, usecols=columns, dtype={'weekday':object,'hour':object,'region':object,'city':object,'advertiser':object})
    else: df = pd.read_csv(fileName, usecols=test_columns, dtype={'weekday':object,'hour':object,'region':object,'city':object,'advertiser':object})
    print('Preprocessing (%s)...' % fileName)
    if train: df = undersample(df, zeroMult=ZERO_MULT)
    features = getFeatures(df, fit=train)
    return df, features


def getpredictionORTB1(pCTR, budgetRemaining, avgCTR):
    # return 50 * pCTR / avgCTR
    # if pCTR < .47: return 0
    return np.sqrt(MODEL_CONST / L * pCTR + MODEL_CONST**2) - MODEL_CONST #* mult
    # return np.sqrt(pCTR) * 70#* budgetRemaining/25000 #(pCTR * (25000/budgetRemaining))**2 * (400 * budgetRemaining/25000)
    # return np.sqrt(pCTR/avgCTR) * 150


def outputTestResults(model):
    test_df, test_features = loadData(testFile, train=False, test=True)
    guesses = getPredictionsORTB1(model, test_features)
    # Prepare output dataframe and send it to csv file
    out_df = pd.DataFrame({'bidid': test_df['bidid'], 'bidprice': guesses})
    out_df.to_csv(out_file, index=False)



train_df, train_features = loadData(trainFile, train=True)
avgCTR = train_df['click'].mean() # Average ctr of reduced (if performed) data set
print('Learning...')
# model = linear_model.LogisticRegression(n_jobs=-1, C=C, class_weight=None)
model = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', class_weight={0:1,1:2}) #increasing the 1 weight increases both true and false positives
# model = DecisionTreeClassifier(criterion='entropy', max_features=100, class_weight={0:1,1:100})
model.fit(train_features, train_df.click)
# outputTestResults(model)

#Evalute Model on validation set
print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)

pCTR = model.predict_proba(val_features)[:, 1]
pCTR_click = pCTR[val_df['click'] == 1]
pCTR_nonclick = pCTR[val_df['click'] == 0]
print('Average Click pCTR: %f' % np.average(pCTR_click))
print('Median Click pCTR: %f' %np.median(pCTR_click))
print('Average Nonclick pCTR: %f' %np.average(pCTR_nonclick))
print('Median Nonclick pCTR: %f' % np.median(pCTR_nonclick))
print('ROC AUC Score: %f' % mets.roc_auc_score(val_df['click'], pCTR, None))
curve = mets.precision_recall_curve(val_df['click'], pCTR, 1)
print('AUC Score: %f' % mets.auc(curve[1], curve[0], reorder=True))
predicts = np.array([0 if p <= 0.5 else 1 for p in pCTR])
print(mets.confusion_matrix(val_df['click'], predicts)) #model.predict(val_features)

# plt.hist(pCTR_click, 50)
# plt.show()
# plt.hist(pCTR_nonclick, 50)
# plt.show()
# sys.exit()

# results = []
# for c in range(40,51):
# MODEL_CONST = c
bidsPlaced = 0
numWins = 0
clicks = 0
spent = 0
clicksMissed = 0
for i in range(len(pCTR)):
    bidAmt = getpredictionORTB1(pCTR[i], 25000000-spent, avgCTR)
    bid = val_df.iloc[i]
    if (spent + bidAmt) <= 25000000: #Would not place bid if the bid amount surpasses the budget
        bidsPlaced += 1
        if bidAmt > bid.payprice:
            numWins += 1
            # print('Won bid with price %d and pCTR %f' % (bid.payprice, pCTR[i]))
            clicks += bid.click
            spent += bid.payprice
            if bid.click==1: print('Won click, we bid %d and the payprice was %d with pCTR %f' % (bidAmt, bid.payprice, pCTR[i]))
        else:
            if bid.click == 1:
                clicksMissed += 1
                print('Missed click, we bid %d but the payprice was %d with pCTR %f' % (bidAmt, bid.payprice, pCTR[i]))
print('Bids Placed: %d' % bidsPlaced)
print('Wins: %d' % numWins)
print('Clicks Missed: %d' % clicksMissed)
print('CTR: %f' % (clicks / numWins)) #Only need to consider the ads we paid for
print('Conversions: %d' % (clicks))
print('Spend: %d' % (spent/1000))
print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
print('Average CPC: %f' % (spent / clicks / 1000))
# results.append(clicks)
# print(results)
