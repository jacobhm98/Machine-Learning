import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from data_processing import *


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

def trainNetwork(network, dataloader):
    learning_rate = 0.001
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    running_loss = 0.0
    for i, datapoint in enumerate(dataloader, 0):
        X, y = datapoint
        network.zero_grad()
        outputs = network(X)
        loss = F.nll_loss(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss)

def testNetwork(network, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data
            output = network(X)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print("Accuracy: " + str(correct/total))

def createDataLoaders(training_data, training_labels, test_data, test_labels):
    training_data = torch.Tensor(training_data)
    #training_data = F.normalize(training_data)
    training_labels = torch.Tensor(training_labels).long()

    test_data = torch.Tensor(test_data)
    #test_data = F.normalize(test_data)
    test_labels = torch.Tensor(test_labels).long()
    training_set = TensorDataset(training_data, training_labels)
    test_set = TensorDataset(test_data, test_labels)
    training_loader = DataLoader(training_set, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=3, shuffle=False)
    return training_loader, test_loader

def createTrainedNetwork():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeTrainingData(X)
    training_data, test_data, training_labels, test_labels = splitIntoTrainingAndValidation(X, y, 0.3)
    training_loader, test_loader = createDataLoaders(training_data, training_labels, test_data, test_labels)
    net = NeuralNet()
    EPOCHS = 300
    for i in range(EPOCHS):
        trainNetwork(net, training_loader)
        testNetwork(net, training_loader)
        testNetwork(net, test_loader)
createTrainedNetwork()
