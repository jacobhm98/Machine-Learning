import monkdata as m
import dtree as d


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

def splitOnAttribute():
    dataset = m.monk1
