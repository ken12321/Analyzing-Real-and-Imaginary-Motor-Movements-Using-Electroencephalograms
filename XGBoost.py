import xgboost as xgb
import numpy as np

from sklearn import metrics
from clean_dataset import extractCsvDataToArray

test_data, test_data_labels, train_data, train_data_labels = extractCsvDataToArray()

# Subtract 1 from every label so that the labels are 0,1,2 instead of 1,2,3
# This is done because of the way the xgboost library identifies multi-classifications
train_data_labels_xgboost = train_data_labels - 1
test_data_labels_xgboost = test_data_labels - 1

# This is the code to check the most frequently occurring class, and the total length of all classes
# Used to calculate the null hypothesis ( num_most_freq / num_total )
print(np.count_nonzero(test_data_labels_xgboost == 0))
print(np.count_nonzero(test_data_labels_xgboost == 1))
print(np.count_nonzero(test_data_labels_xgboost == 2))

print(len(test_data_labels_xgboost))

# Splits data, this is only used when testing and training on the same data type (eg training and testing on real)
split = round(0.8 * len(test_data))  # Defines an 80-20 split
test = test_data[0:split]
test_labels = test_data_labels_xgboost[0:split]

train = train_data[split:len(train_data)]
train_labels = train_data_labels_xgboost[split:len(train_data_labels_xgboost)]

dtrain = xgb.DMatrix(train, label=train_labels)
dtest = xgb.DMatrix(test, label=test_labels)

# Converts numpy arrays to XGBoost DMatrices
# dtrain = xgb.DMatrix(train_data, label=train_data_labels_xgboost)
# dtest = xgb.DMatrix(test_data, label=test_data_labels_xgboost)

# parameters for XGBoost model
param = {'max_depth': 5,
         'eta': 0.1,
         'objective': 'multi:softmax',
         'num_class': 3,
         'subsample': 0.75,
         'gamma': 1}
param['eval_metric'] = ['merror', 'mlogloss']  # evaluation techniques used

evallist = [(dtrain, 'train'), (dtest, 'test')]

num_round = 150  # number of rounds for boosting
bst = xgb.train(param, dtrain, num_round, evallist)  # train model with given parameters

prediction = bst.predict(dtest)  # make predictions on the test dataset
prediction = prediction.astype(int)

print("Accuracy: ", metrics.accuracy_score(test_labels, prediction))

