import xgboost as xgb
import numpy as np
import pandas as pd
import wfdb
import ast
import math

from sklearn import metrics
from clean_dataset import extractCsvDataToArray

test_data, test_data_labels, train_data, train_data_labels = extractCsvDataToArray()

# Subtract 1 from every label so that the labels are 0,1,2 instead of 1,2,3
# This is done because of the way the xgboost library identifies multi-classifications
train_data_labels_xgboost = train_data_labels - 1
test_data_labels_xgboost = test_data_labels - 1

print(train_data_labels_xgboost)
print(len(train_data_labels_xgboost))
# Converts numpy arrays to XGBoost DMatrices
dtrain = xgb.DMatrix(train_data, label=train_data_labels_xgboost)
dtest = xgb.DMatrix(test_data, label=test_data_labels_xgboost)

print("here3")

param = {'booster': '', 'max_depth': 6, 'eta': 0.2, 'objective': 'multi:softmax', 'num_class': 3}  # parameters for XGBoost model
param['eval_metric'] = ['merror', 'mlogloss']  # evaluation techniques used

print("here4")

evallist = [(dtrain, 'train'), (dtest, 'test')]

num_round = 300  # number of rounds for boosting
bst = xgb.train(param, dtrain, num_round, evallist)  # train model with given parameters

prediction = bst.predict(dtest)  # make predictions on the test dataset


# prints the relevant metrics used for evaluation from the prediction
# print("RMSE: ", math.sqrt(metrics.mean_absolute_error(test_data_labels, prediction)))
# print("MSE: ", metrics.mean_squared_error(test_data_labels, prediction))
# print("MAE: ", metrics.mean_absolute_error(test_data_labels, prediction))
# print("R2 score: ", metrics.r2_score(test_data_labels, prediction))

print("Accuracy: ", metrics.accuracy_score(test_data_labels, prediction))
print("Precision: ", metrics.precision_score(test_data_labels, prediction))
print("F1: ", metrics.f1_samples(test_data_labels, prediction))
