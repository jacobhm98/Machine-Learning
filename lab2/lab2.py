import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


np.random.seed(100)
# Generate training data
classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.5, 0.5], np.random.randn(10, 2) * 0.5 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.5 + [0.0, -1.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate ((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)
permute=list(range(N))
np.random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]



# CONSTANTS
C = 100
Pmat = np.zeros((N, N), dtype=float)
power:int = 6
sigma:float = 20

def linear_kernel(j, i):
    return np.dot(i, j)

def polynomial_kernel(i, j):
    return pow(np.dot(i, j) + 1, power)

def rbf_kernel(i, j):
    return math.exp(-math.pow(np.linalg.norm(np.subtract(i, j)), 2)/(2 * math.pow(sigma,2)))

kernel_function = linear_kernel

def computePmat():
    for i in range(0, N):
        for j in range(0, N):
            Pmat[i][j] = targets[i] * targets[j] * kernel_function(inputs[i], inputs[j])

def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(alpha, Pmat)) - np.sum(alpha)

def zerofun(alpha):
    return np.dot(alpha, targets)

def callMinimize():
    computePmat()
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = {'type': 'eq', 'fun': zerofun}
    ret = minimize(objective, start, bounds=B, constraints=XC)
    if not ret['success']:
        raise Exception("WAS NOT ABLE TO OPTIMIZE FUNCTION")
    alpha = ret['x']
    supportVectors = []
    for i in range(N):
        if alpha[i] <= 10e-5:
            continue
        supportVectors.append((alpha[i], inputs[i], targets[i]))
    return supportVectors
supportVectors = callMinimize()

def extractUsefulSV(suppportVectors):
    indexS = 0
    for i in range(len(supportVectors)):
        if C is None:
            break
        if supportVectors[i][0] < C:
            indexS = i
            break
    return indexS

def calculateThreshold(supportVectors):
    indexS = extractUsefulSV(supportVectors)
    sum = 0
    for i in range(len(supportVectors)):
        sum += supportVectors[i][0] * supportVectors[i][2] * kernel_function(supportVectors[indexS][1], supportVectors[i][1])
    return sum - supportVectors[indexS][2]

b = calculateThreshold(supportVectors)

def indicatorFunction(X):
    sum = 0
    for (alpha, data_point, label) in supportVectors:
        sum += alpha * label * kernel_function(data_point, X)
    return sum - b

# Plot training data
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.plot([p[1][0] for p in supportVectors], [p[1][1] for p in supportVectors], 'yo')
plt.axis('equal')

# Plot decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicatorFunction([X, Y]) for X in xgrid] for Y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.savefig('svmplot.pdf')
plt.show()






