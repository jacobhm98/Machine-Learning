import numpy as np
import random

nameToLabel = {"Bob": 0, "Atsuto": 1, "Jörg": 2}
gradeToFloat = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0, "E": 4.0, "F": 5.0}
boolToFloat = {"False": 0.0, "True": 1.0}


def importCsv(filepath):
    data = np.genfromtxt(filepath, dtype=str, delimiter=',')
    return data

def sanitizeData(X):
    Ndims = X.shape[1]
    labels = []
    datapoints = []
    for entry in X:
        labels.append(nameToLabel.get(entry[1]))
        datavector = []
        for i in range(2, Ndims):
            if i == 6:
                datavector.append(boolToFloat.get(entry[i]))
                continue
            if i == 7:
                datavector.append(gradeToFloat.get(entry[i]))
                continue
            try:
                datavector.append(float(entry[i]))
            except:
                print(entry[0])
        datapoints.append(datavector)
    return np.array(datapoints), np.array(labels)

def shuffleData(X, y):
    assert(X.shape[0] == y.shape[0])
    Npts = X.shape[0]
    points = []
    for i in range(Npts):
       points.append((X[i], y[i]))
    random.shuffle(points)
    for i in range(Npts):
        X[i] = points[i][0]
        y[i] = points[i][1]
    return X, y

def splitIntoTrainingAndValidation(X, y, valFraction=0.3):
    assert X.shape[0] == y.shape[0]
    Npts = X.shape[0]
    sliceIndex = int(Npts * valFraction)
    training_data = X[sliceIndex:]
    test_data = X[:sliceIndex]
    training_labels = y[sliceIndex:]
    test_labels = y[:sliceIndex]
    return training_data, training_labels, test_data, test_labels

def main():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeData(X)
    X, y = shuffleData(X, y)
    X, y, validationX, validationY = splitIntoTrainingAndValidation(X, y, 0.3)
    print(X.shape[0])
    print(validationX.shape[0])
main()
