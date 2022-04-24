import numpy as np
import pandas as pd
import wfdb
import ast
import math

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


from clean_dataset import extractCsvDataToArray




test_data, test_data_labels, train_data, train_data_labels = extractCsvDataToArray()
print("here")

# reshapes test and training numpy arrays into correct shapes
# nsamples, nx, ny = train_data.shape
# train_data = train_data.reshape((nsamples,nx*ny))
#
# nsamples, nx, ny = test_data.shape
# test_data = test_data.reshape((nsamples,nx*ny))

# build the model with given parameters
regressor = RandomForestRegressor(n_estimators=30, random_state=0)

regressor.fit(train_data, train_data_labels)  # train the model

predictions = regressor.predict(test_data)  # make predictions on the test dataset

# prints the relevant metrics used for evaluation from the prediction
print("Accuracy: ", metrics.accuracy_score(test_data_labels, predictions))
# print("RMSE: ", math.sqrt(metrics.mean_absolute_error(test_data_labels, predictions)))
# print("MSE: ", metrics.mean_squared_error(test_data_labels, predictions))
# print("MAE: ", metrics.mean_absolute_error(test_data_labels, predictions))
# print("R2 score: ", metrics.r2_score(test_data_labels, predictions))