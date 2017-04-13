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

#from polylearn import FactorizationMachineClassifier
#improve pCTR
#write non-linear model
#machine learning for c and langrangian multiplier


trainFile = 'dataset/train.csv'
validateFile = 'dataset/validation.csv'
testFile = 'dataset/test.csv'
out_file = 'dataset/testing_bidding_price.csv'
<<<<<<< HEAD
BALANCED = 'balanced'
K_FEATS = 30
ZERO_MULT = 100
=======
BALANCED = None #'balanced'
K_FEATS = 100
ZERO_MULT = 10
>>>>>>> 0425af7cc6fd258f06d5645dbe3f909353a4c216
C = 1e6
BASE_BID = 50

MODEL_CONST = 45
L = 6*10**(-4.5)

dv = DictVectorizer()
#For Kbest feature selection
kbest = SelectKBest(f_classif, k=K_FEATS)

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
    else:
        vec = dv.transform(data.drop(metrics, axis=1).to_dict(orient='records'))

    vec = normalize(vec, axis=1)
    if fit:
        kbest.fit(vec, data['click'])

    # selector = SelectFromModel(lr, prefit=True)
    # features = selector.transform(vec)

    # features = kbest.transform(vec)
    features = vec
    # if fit:
    #     lr.fit(features, data['click'])
    # selector = SelectFromModel(lr, prefit=True)

    return features

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


def getpredictionORTB1(pCTR, pCat, budgetRemaining, avgCTR):
    multipliers = {0: 1, 1: 1.5, 2: 1}
    mult = multipliers[pCat]
    # return 50 * pCTR / avgCTR
    if pCTR < .47: return 0
    # return (np.sqrt(MODEL_CONST / L * pCTR + MODEL_CONST**2) - MODEL_CONST) * mult
    return pCTR**2 * 300 #(pCTR * (25000/budgetRemaining))**2 * (400 * budgetRemaining/25000)


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
# model = linear_model.LogisticRegression(n_jobs=-1, C=C, class_weight=BALANCED)
# model = FactorizationMachineClassifier(loss='logistic', n_components=1, fit_linear=False)
model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=200, class_weight={0:1,1:1000})
# model = DecisionTreeClassifier(criterion='entropy', max_features=100, class_weight={0:1,1:100})
model.fit(train_features, train_df.click)
learnPriceStrata(train_df, train_features)

# outputTestResults(model)

#Evalute Model on validation set
print('Evaluating...')
val_df, val_features = loadData(validateFile, train=False)

pCats = stratifier.predict(val_features)
# pCTR = model.predict_proba(val_features)
pCTR = model.predict_proba(val_features)[:, 1]
pCTR_click = pCTR[val_df['click'] == 1]
pCTR_nonclick = pCTR[val_df['click'] == 0]
print(pCTR_click)
print (pCTR_nonclick)
print('Average Click pCTR: %f' % np.average(pCTR_click))
print('Median Click pCTR: %f' %np.median(pCTR_click))
print('Average Nonclick pCTR: %f' %np.average(pCTR_nonclick))
print('Median Nonclick pCTR: %f' % np.median(pCTR_nonclick))
print('ROC AUC Score: %f' % mets.roc_auc_score(val_df['click'], pCTR, None))
curve = mets.precision_recall_curve(val_df['click'], pCTR, 1)
print('AUC Score: %f' % mets.auc(curve[1], curve[0], reorder=True))
predicts = np.array([0 if p <= 0.3 else 1 for p in pCTR])
print(mets.confusion_matrix(val_df['click'], predicts)) #model.predict(val_features)

# plt.plot(curve[1], curve[0])
# plt.show()

plt.hist(pCTR_click, 50)
plt.show()
plt.hist(pCTR_nonclick, 50)
plt.show()
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
