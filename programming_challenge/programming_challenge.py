import numpy as np

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
        labels.append([int(entry[0]), nameToLabel.get(entry[1])])
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


def main():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeData(X)


main()
