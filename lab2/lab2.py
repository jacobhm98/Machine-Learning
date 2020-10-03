import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#CONSTANTS
N = 1000
DIMENSIONS = 2
training_data = np.zeros((DIMENSIONS, N), dtype=float)
labels = np.zeros(N)

def linear_kernel(j, i):
    return np.dot(i, j)


def objective(alpha):
    sum = 0
     for i in range(0, N):
         for j in range(0, N):
             sum += alpha[i] * alpha[j] *
