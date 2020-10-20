import numpy as np
from sklearn.model_selection import train_test_split

nameToLabel = {"Bob": 0, "Atsuto": 1, "Jörg": 2}
labelToName = {0: "Bob", 1: "Atsuto", 2: "Jörg"}
gradeToFloat = {"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.4, "F": 0.5, "Fx": 0.6}
boolToFloat = {"False": 0.0, "True": 1.0}


def importCsv(filepath):
    data = np.genfromtxt(filepath, dtype=str, delimiter=',')
    return data


def sanitizeTestData(X):
    Ndims = X.shape[1]
    datapoints = []
    for entry in X:
        datavector = []
        for i in range(1, Ndims):
            if i == 5:
                datavector.append(boolToFloat[entry[i]])
                continue
            if i == 6:
                datavector.append(gradeToFloat[entry[i]])
                continue
            try:
                datavector.append(float(entry[i]))
            except:
                print(entry[0])
        datapoints.append(datavector)
    return np.array(datapoints)


def sanitizeTrainingData(X):
    Ndims = X.shape[1]
    labels = []
    datapoints = []
    for entry in X:
        labels.append(nameToLabel[entry[1]])
        datavector = []
        for i in range(2, Ndims):
            if i == 6:
                datavector.append(boolToFloat[entry[i]])
                continue
            if i == 7:
                datavector.append(gradeToFloat[entry[i]])
                continue
            try:
                datavector.append(float(entry[i]))
            except:
                print(entry[0])
        datapoints.append(datavector)
    return np.array(datapoints), np.array(labels)


def splitIntoTrainingAndValidation(X, y, valFraction=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=valFraction)
    return X_train, X_test, y_train, y_test

def countEntriesPerLabel(y):
    classes = np.unique(y)
    for label in classes:
        entries = np.where(y == label)
        print("Number of entries for class: " + str(label))
        print(entries[0].shape)
