import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import Imputer, OneHotEncoder, FunctionTransformer, normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#improve pCTR
#write non-linear model
#machine learning for c and langrangian multiplier


trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
testFile = 'dataset/test.csv'
out_file = 'dataset/testing_bidding_price.csv'
BALANCED = None #'balanced'
K_FEATS = 50
ZERO_MULT = 100
C = 1e6
BASE_BID = 50

MODEL_CONST = 1e6
L = .0007

dv = DictVectorizer()
#For Kbest feature selection
kbest = SelectKBest(chi2, k=K_FEATS)


lr = linear_model.LogisticRegression(C=2, penalty='l1', dual=False) # For select from model feature selection


stratifier = RandomForestClassifier(n_estimators=10)

columns = ['click','payprice','bidprice','bidid','weekday','hour','region','city','useragent',\
            'slotformat','adexchange','slotvisibility',\
            'slotwidth','slotheight','slotprice','creative','keypage']
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
    # Stratify useragent into os and browser
    agent_info = pd.DataFrame(np.array([[item for item in agent.split('_')] for agent in data.useragent]), columns=['os','browser'])
    data = data.drop('useragent',axis=1).join(agent_info)
    weekend_info = pd.DataFrame(np.array([1 if 5<=int(weekday)<=7 else 0 for weekday in data.weekday]))
    peak_info = pd.DataFrame(np.array([1 if 17<=int(hour)<=24 else 0 for hour in data.hour]))
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
        # lr.fit(vec, data['click'])
        kbest.fit(vec, data['click'])
    else:
        vec = dv.transform(data.drop(metrics, axis=1).to_dict(orient='records'))
    # selector = SelectFromModel(lr, prefit=True)
    # features = selector.transform(vec)
    vec = normalize(vec, axis=0) #normalize(vec)
    features = kbest.transform(vec)
    if fit:
        lr.fit(features, data['click'])
    selector = SelectFromModel(lr, prefit=True)
    # features = selector.transform(features)

    # if fit:
    #     vec_df = pd.DataFrame(dv.fit_transform(data.to_dict(orient='records')).toarray())
    # else:
    #     vec_df = pd.DataFrame(dv.transform(data.to_dict(orient='records')).toarray())
    # vec_df.columns = dv.get_feature_names()
    # vec_df.index = data.index
    # #Reduce the feature space manually? As in get reduce to a bit for each browser, device type
    # # Do one hot encoding for weekday?, hour?, region, city*,
    # # Hour, weekday don't have to be categorical, does they?
    #
    # # feat_select_mapper = DataFrameMapper([(vec_df.drop(metrics,axis=1).columns, kbest)])
    # if fit: lr.fit(vec_df.drop(metrics,axis=1), data['click'])
    # selector = SelectFromModel(lr, prefit=True)
    # feat_select_mapper = DataFrameMapper([(vec_df.drop(metrics,axis=1).columns, selector)])
    # features = feat_select_mapper.transform(vec_df.drop(metrics,axis=1))
    # print(features.shape[1])

    # if fit:
    #     features = feat_select_mapper.fit_transform(vec_df.drop(metrics,axis=1), data['click'])
    # else:
    #     features = feat_select_mapper.transform(vec_df.drop(metrics,axis=1))
    # features_df = pd.DataFrame(features, columns=vec_df.columns[kbest.get_support(indices=True)])
    # print(features_df.columns)
    return features.toarray()

def appendPriceCategories(features, cats):
    cats = cats.reshape(-1,1)
    features = np.append(features,cats, axis=1)
    return normalize(features, axis=1)

def getCats(data):
    cats = []
    for price in data['payprice']:
        if price <= 60:
            cats.append(0)
        elif price <= 160:
            cats.append(1)
        else:
            cats.append(2)
    return np.array(cats)

def learnPriceStrata(data, features):
    cats = getCats(data)
    stratifier.fit(features, cats)

def loadData(fileName, train=False, test=False):
    print('Loading data (%s)...' % fileName)
    if not test: df = pd.read_csv(fileName, usecols=columns, dtype={'weekday':object,'hour':object,'region':object,'city':object})
    else: df = pd.read_csv(fileName, usecols=test_columns, dtype={'weekday':object,'hour':object,'region':object,'city':object})
    print('Preprocessing (%s)...' % fileName)
    if train: df = undersample(df, zeroMult=ZERO_MULT)
    features = getFeatures(df, fit=train)
    return df, features

# def getPredictions(model, features, avgCTR):
#     pCTR = model.predict_proba(features)[:, 1]
#     print(pCTR)
#     return BASE_BID * pCTR / avgCTR


# def getPredictionsORTB1(model, features):
#     pCTR = model.predict_proba(features)[:, 1]
#     pCats = stratifier.predict(features)
#     multipliers = {0: 1, 1: 1, 2: 1}
#     mults = np.vectorize(lambda x: multipliers[x])(pCats)
#     # L = 25000 / 299749
#     # L = 0.006
#     return (np.sqrt(MODEL_CONST / L * pCTR + MODEL_CONST**2) - MODEL_CONST) * mults

def getpredictionORTB1(pCTR, pCat, budgetRemaining, avgCTR):
    multipliers = {0: 1, 1: 1.5, 2: 1}
    mult = multipliers[pCat]
    # return 50 * pCTR / avgCTR
    return (np.sqrt(MODEL_CONST / L * pCTR + MODEL_CONST**2) - MODEL_CONST) #* mult

    # return ((pCTR + (MODEL_CONST**2 * L**2 + pCTR**2)**(1/2)) / (MODEL_CONST * L))**(1/3) - ((MODEL_CONST * L) / (pCTR + (MODEL_CONST**2 * L**2 + pCTR**2)**(1/2)))**(1/3)

    # bidding appears to be a logarithmic function (postive and steep in the beginning, flattens out
    # later on.  Thus, allocating more budget on the lower-cost bids is more beneficial than
    # increasing budget on more expensive ones later on (see paper for images/clairification)

    # bidding strategy for normal target
    # ((c/L)T) + c^2 )^(1/2) - c
    # where: L = langrangian multiplier; T = Theta (pCTR I think); c = constant
    # suggested initial L = 5.2x10^-7

    #bidding strategy for competitive targets:
    # ((T + (c^2 L^2 + T^2)^(1/2))/ cL] ^ (1/3)) - (cL / (T + (c^2 L^2 + T^2 )^(1/2)) ^(1/3))
    # where variables and suggested L are the same as above

    # L = B/N where B is our overall budget, and N is the estimated number of bid requests over life of B

    #suggests using machine learning to get all tuning parameters (L, c, T),
    # where T is the pCTR for the current bid

def outputTestResults(model):
    test_df, test_features = loadData(testFile, train=False, test=True)
    guesses = getPredictionsORTB1(model, test_features)
    # Prepare output dataframe and send it to csv file
    out_df = pd.DataFrame({'bidid': test_df['bidid'], 'bidprice': guesses})
    out_df.to_csv(out_file, index=False)



train_df, train_features = loadData(trainFile, train=True)
cats_train = getCats(train_df)
avgCTR = train_df['click'].mean() # Average ctr of reduced (if performed) data set
print('Learning...')
model = linear_model.LogisticRegression(n_jobs=-1, C=C, class_weight=BALANCED)
# model = linear_model.LogisticRegressionCV(n_jobs=-1, class_weight=BALANCED, Cs=, cv=10, penalty='l2', dual=False, refit=True, multi_class='ovr', solver='liblinear')
model.fit(appendPriceCategories(train_features, cats_train), train_df.click)
learnPriceStrata(train_df, train_features)

# outputTestResults(model)

#Evalute Model on validation set
print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)

pCats = stratifier.predict(val_features)
# pCTR = model.predict_proba(train_features)[:, 1]
# pCTR_click = pCTR[train_df['click'] == 1]
# pCTR_nonclick = pCTR[train_df['click'] == 0]
pCTR = model.predict_proba(appendPriceCategories(val_features, pCats))[:, 1]
# plt.hist(pCTR)
# plt.axis([0,.02,0,300000])
# plt.show()
pCTR_click = pCTR[val_df['click'] == 1]
pCTR_nonclick = pCTR[val_df['click'] == 0]
print('Average Click pCTR: %f' % np.average(pCTR_click))
print('Median Click pCTR: %f' %np.median(pCTR_click))
print('Average Nonclick pCTR: %f' %np.average(pCTR_nonclick))
print('Median Nonclick pCTR: %f' % np.median(pCTR_nonclick))
# sys.exit()

bidsPlaced = 0
numWins = 0
clicks = 0
spent = 0
clicksMissed = 0
for i in range(len(pCTR)):
    bidAmt = getpredictionORTB1(pCTR[i], pCats[i], 25000-spent, avgCTR)
    bid = val_df.iloc[i]
    if (spent + bidAmt) <= 25000: #Would not place bid if the bid amount surpasses the budget
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
print('Spend: %d' % spent)
print('Average CPM: %f' % (spent / numWins)) # Average bid price / pay price?
print('Average CPC: %f' % (spent / clicks))
