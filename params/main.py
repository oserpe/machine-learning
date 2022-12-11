from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, train_test_split

from ..utils.estimator import Estimator

from ..dataset.utils import prepare_dataset


def adaboost_grid_search(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [10, 15, 20, 30],
        'learning_rate': [0.1, 0.5, 1, 2, 5],
        'random_state': [0, 1, 2],
    }

    for estimator in [Estimator.PERCEPTRON, Estimator.DECISION_TREE, Estimator.SVC_LINEAR]:
        print("Estimator: {}".format(estimator.value["estimator"]))

        grid_search = GridSearchCV(
            AdaBoostClassifier(algorithm="SAMME", estimator=estimator.value["estimator"]), param_grid, cv=5, return_train_score=True, verbose=0)
        grid_search.fit(X_train, y_train)

        print("Best parameters: {}".format(grid_search.best_params_))
        print(
            "Best cross-validation score: {:.3f}".format(grid_search.best_score_))
        print(
            "Test-set score: {:.3f}".format(grid_search.score(X_test, y_test)))


if __name__ == "__main__":
    random_state = 1

    X, y = prepare_dataset(standardize=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    adaboost_grid_search(X_train, y_train, X_test, y_test)
