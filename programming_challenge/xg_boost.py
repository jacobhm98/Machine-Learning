from data_processing import *
import xgboost as xgb
from sklearn.model_selection import train_test_split

def main():
    X = importCsv("TrainOnMe.csv")
    X, y = sanitizeData(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(y_train.shape[0], X_test.shape[0])
    model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=4, gamma=0.5)
    eval_metric = ["auc", "error"]
    model.fit(X_train, y_train, eval_metric=eval_metric, verbose=True)
    test_acc = 0.0
    val_acc = 0.0
    n_iter = 100
    for i in range(n_iter):
        test_acc += model.score(X_train, y_train)
        val_acc += (model.score(X_test, y_test))
    print("Training: " + str(test_acc/n_iter))
    print("Validation: " + str(val_acc / n_iter))
main()