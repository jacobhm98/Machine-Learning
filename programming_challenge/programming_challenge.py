import numpy as np

def importCsv(filepath):
    data = np.genfromtxt(filepath, delimiter=',')
    return data


def main():
    X = importCsv("TrainOnMe.csv")
    print(X.shape)
main()