import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# CONSTANTS
N = 1000
DIMENSIONS = 2
C = None
training_data = np.zeros((N, DIMENSIONS), dtype=float)
labels = np.zeros(N)


Pmat = np.zeros((N, N), dtype=float)


def linear_kernel(j, i):
    return np.dot(i, j)


kernel_function = linear_kernel


def computePmat():
    for i in range(0, N):
        for j in range(0, N):
            Pmat[i][j] = labels[i] * labels[j] * kernel_function(training_data[i], training_data[j])


def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(alpha, Pmat)) - np.sum(alpha)


def zerofun(alpha):
    return np.dot(alpha, labels)


def callMinimize():
    computePmat()
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = {'type': 'eq', 'fun': 'zerofun'}
    ret = minimize(objective, start, bounds=B, constraints=XC)
    if not ret['success']:
        raise Exception("WAS NOT ABLE TO OPTIMIZE FUNCTION")
    alpha = ret['x']
    supportVectors = []
    for i in range(N):
        if alpha[i] <= 10e-5:
            continue
        supportVectors.append((alpha[i], training_data[i], labels[i]))
    return supportVectors
supportVectors = callMinimize()

def calculateThreshold(supportVectors):
    indexS = 0
    sum = 0
    for i in range(len(supportVectors)):
        if supportVectors[i][0] < C:
            indexS = i
            break
    for i in range(len(supportVectors)):
        sum += supportVectors[i][0] * supportVectors[i][2] * kernel_function(supportVectors[indexS][1], supportVectors[i][1])
    return sum - supportVectors[indexS][2]
b = calculateThreshold(supportVectors)

def indicatorFunction(X):
    sum = 0
    for (alpha, data_point, label) in supportVectors:
        sum += alpha * label * kernel_function(data_point, X)
    return sum - b






