import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


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
        labels.append((entry[0], nameToLabel.get(entry[1])))
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
    return np.array(X, dtype=float), np.array(y, dtype=float)

def splitIntoTrainingAndValidation(X, y, valFraction=0.3):
    Npts = X.shape[0]
    sliceIndex = int(Npts * valFraction)
    training_data = X[sliceIndex:]
    training_labels = y[sliceIndex:]
    test_data = X[:sliceIndex]
    test_labels = y[:sliceIndex]

    training_data = torch.Tensor(training_data)
    training_labels = torch.Tensor(training_labels)
    test_data = torch.Tensor(test_data)
    test_labels = torch.Tensor(test_labels)
    training_set = TensorDataset(training_data, training_labels)
    test_set = TensorDataset(test_data, test_labels)
    return training_set, test_set

def main():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeData(X)
    X, y = shuffleData(X, y)
    training_data, test_data = splitIntoTrainingAndValidation(X, y, 0.3)
    training_loader = DataLoader(training_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    net = NeuralNet()
main()
