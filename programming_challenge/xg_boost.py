from data_processing import *
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate


def trainTreeWithValidation(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    eval_metric = ["auc", "error"]
    model.fit(X_train, y_train, eval_metric=eval_metric, verbose=True)
    test_acc = 0.0
    val_acc = 0.0
    n_iter = 100
    for i in range(n_iter):
        test_acc += model.score(X_train, y_train)
        val_acc += (model.score(X_test, y_test))
    model.save_model("lastXGBoost.model")
    print("Training: " + str(test_acc / n_iter))
    print("Validation: " + str(val_acc / n_iter))


def trainTreeWithCV(model, X, y):
    scores = cross_validate(model, X, y, scoring="accuracy", return_train_score=True)
    train_acc = 0.0
    i = 0
    for train_result in scores["train_score"]:
        train_acc += train_result
        i += 1
    train_acc /= i
    i = 0
    test_acc = 0.0
    for test_result in scores["test_score"]:
        test_acc += test_result
        i += 1
    test_acc /= i
    print("Train result: %f.2, Test result: %f.2" % (train_acc, test_acc))

def trainTreeWithWholeTrainSet(model, X, y):
    model.fit(X, y)
    return model

def printPredictions(predictions):
    file = open("predictions.txt", "w")
    for entry in predictions:
        file.write(labelToName[entry] + "\n")
    file.close()


def main():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeTrainingData(X)
    countEntriesPerLabel(y)
    model = xgb.XGBClassifier(learning_rate=0.2, n_estimators=50, max_depth=4, gamma=0.5)
    testSet= importCsv("EvaluateOnMe.csv")
    testSet = sanitizeTestData(testSet)
    model = trainTreeWithWholeTrainSet(model, X, y)
    testPredictions = model.predict(X)
    predictions = model.predict(testSet)
    printPredictions(predictions)
    print("oj")




main()
