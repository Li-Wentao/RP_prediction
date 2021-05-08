# import packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import cvxopt
from bayesian_bootstrap.bootstrap import mean, highest_density_interval
import xgboost as xgb
from sklearn.metrics import accuracy_score

##################### Loading data #####################
df = pd.read_csv('feature_train.csv', index_col=0) # Features of training dataset (unique ID)
out = pd.read_csv('label_train.csv', index_col=0) # Labels of training dataset

##################### Data preprocessing #####################
def scale(y, c=True, sc=True):
    x = y.copy()

    if c:
        x -= x.mean()
    if sc and c:
        x /= x.std()
    elif sc:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
    return x
X = pd.get_dummies(df, drop_first=True)
# Find the ratio of null value in the dataset and drop features with null ratio greater than 20%
thres = 0.2
vars_to_drop = (X.isnull().sum()/X.shape[0] >= thres).index[(X.isnull().sum()/X.shape[0] > thres)].tolist()
X = X.drop(vars_to_drop, axis = 1)
# Forward fill to fill the nans
X = X.fillna(method = 'ffill')
# Find the highly realted features with threshold 0.7
corr = X.corr()
np.fill_diagonal(corr.values, 0) 
# X.interpolate(method='polynomial')
vars_to_drop = corr.index[((corr > 0.7).sum() > 0)].tolist()
X = X.drop(vars_to_drop, axis = 1)
X = scale(X)
X = X.dropna(axis = 1)
X.insert(loc=0, column='Intercept', value=1)
X = X.drop('APOEGN_E2/E3', axis = 1)
y = out.rapid_progressor.values

# Bootstrapping
data_positive = X[out.rapid_progressor == 1]
data_negative = X[out.rapid_progressor == 0]
boostrapped = []
for i in data_negative.columns[1:]:
    boostrapped += [mean(data_negative[i], 1330)]
bs = pd.DataFrame(np.transpose(boostrapped), columns=data_negative.columns[1:])
bs.insert(loc=0, column='Intercept', value=1)
X_bs = pd.concat([X, bs])
y_bs = np.append(y, np.repeat(1, 1330))

##################### Prediction #####################
df_test = pd.read_csv('feature_test.csv', index_col=0)
df_test = pd.get_dummies(df_test, drop_first=True)
test = df_test[X.columns[1:]]
test = test.fillna(method = 'ffill')
test.insert(loc=0, column='Intercept', value=1)
logit_model=sm.Logit(y_bs, X_bs)
result=logit_model.fit_regularized(method = 'l1_cvxopt_cp')
def sigmoid(x, beta):
    return 1 / (1 + np.exp(- x @ beta.T))
pre = sigmoid(test, result.params)
pre = pd.DataFrame(pre)
pre.columns = ['rapid_progressor']
pre.to_csv('submission.csv', index=True)

