import monkdata as m
import dtree as d
import numpy as np

def printFirstFourAssignments():
    print("Entropy of the datasets:")
    entropy1 = d.entropy(m.monk1)
    print(entropy1)

    entropy2 = d.entropy(m.monk2)
    print(entropy2)

    entropy3 = d.entropy(m.monk3)
    print(entropy3)

    print("Average information gain of the attributes in the datasets")
    datasets = (m.monk1, m.monk2, m.monk3)
    for dataset in datasets:
        print("New dataset")
        for attribute in m.attributes:
            print("Information gain of attribute: " + str(attribute.name))
            print(d.averageGain(dataset, attribute))

def selectAttributeToSplitOn(dataset):
    index = 0
    maxGain = 0
    for i in range(0, len(m.attributes)):
        if d.averageGain(dataset, m.attributes[i]) > maxGain:
            index = i
            maxGain = d.averageGain(dataset, m.attributes[i])
    return m.attributes[index]


def splitDataset(dataset):
    subsets = []
    splitAttribute = selectAttributeToSplitOn(dataset)
    for value in splitAttribute.values:
        subsets.append(d.select(dataset, splitAttribute, value))
    return [splitAttribute, subsets]

def createTreeManually():
    tree = splitDataset(m.monk1)
    for i in range(0, len(tree[1])):
        tree[1][i] = splitDataset(tree[1][i])
    for firstLevelNode in tree[1]:
        for i in range(0, len(firstLevelNode[1])):
            firstLevelNode[1][i] = d.mostCommon(firstLevelNode[1][i])
    return tree
tree = createTreeManually()
id1Tree = d.buildTree(m.monk1, m.attributes, 2)
print("Manually created tree:")
print(tree)
print("ID3 created tree 2 levels deep")
print(id1Tree)