import pandas as pd
import numpy as np


def readCsvData(data_files):
    # Reads all the csv files of a legend and returns them in an array
    extracted_data_list = []
    eliminated_data_list = []
    index = 0
    for path in data_files:
        index += 1

        # Try-except used to eliminate any csvs with no data, as they cannot be read.
        try:
            data = pd.read_csv(path)

            # If more than 655, cut to 655
            if(len(data) > 655):
                s = len(data) - 655
                data.drop(data.tail(s).index, inplace=True)
            # If less than 655, cut from dataset
            if(len(data) < 655):
                eliminated_data_list.append(index)
            else:
                extracted_data_list.append(data)
        except:
            eliminated_data_list.append(index)

    return extracted_data_list, eliminated_data_list


def extractCsvDataToArray():
    # Creates an array of the filenames with their corresponding label
    legend_train = pd.read_csv("./data_extracted/LEGEND_REAL.csv")
    legend_test = pd.read_csv("./data_extracted/LEGEND_IMAGINE.csv")

    # Creates arrays of file names of csvs
    test_data_files = legend_test.data
    train_data_files = legend_train.data

    # Creates arrays of the data contained in csv files, and indexes of eliminated data
    test_data, test_elim = readCsvData(test_data_files)
    train_data, train_elim = readCsvData(train_data_files)


    def extractFeatures(data):
        extracted_features = []
        for entity in data:
            feature_array = []
            for feature in entity:
                feature_array.append(np.array(entity[feature], dtype=float))
            extracted_features.append(feature_array)
        return extracted_features

    train = extractFeatures(train_data)
    test = extractFeatures(test_data)

    # Converts training and testing arrays to numpy arrays
    test_data = np.asarray(test)
    train_data = np.asarray(train)

    # Reshapes training and testing data
    nsamples, nx, ny = train_data.shape
    train_data = train_data.reshape((nsamples, nx * ny))

    nsamples, nx, ny = test_data.shape
    test_data = test_data.reshape((nsamples, nx * ny))

    # Creates arrays of the data label subtract any indexes that have been eliminated through cleaning process
    test_labels = np.asarray(legend_test.label)
    test_labels = np.delete(test_labels, test_elim)

    train_labels = np.asarray(legend_train.label)
    train_labels = np.delete(train_labels, train_elim)

    print("Finished reading data...")
    return test_data, test_labels, train_data, train_labels

