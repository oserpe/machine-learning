import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..utils.estimator import Estimator

from ..dataset.utils import prepare_dataset


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def print_grid_search_results(grid_search, filename=None):
    print("Best parameters: {}".format(grid_search.best_params_))
    print(
        "Best cross-validation score: {:.3f}".format(grid_search.best_score_))

    if filename is not None:
        with open(filename, "w") as f:
            f.write("Best parameters: {}\n".format(grid_search.best_params_))
            f.write(
                "Best cross-validation score: {:.3f}\n".format(grid_search.best_score_))


def print_worst_grid_search_results(grid_search, filename=None):
    # find the element with the minimum mean cross-validation score
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    worst_result = grid_search.cv_results_['params'][mean_test_scores.argmin()]
    best_result = grid_search.cv_results_['params'][mean_test_scores.argmax()]

    # print the worst performing combination of parameters
    print("Worst parameters: {}".format(worst_result))
    print("Worst cross-validation score: {:.3f}".format(
        mean_test_scores.min()))
    print("Best parameters: {}".format(best_result))
    print("Best cross-validation score: {:.3f}".format(
        mean_test_scores.max()))

    if filename is not None:
        with open(filename, "w") as f:
            f.write("Worst parameters: {}\n".format(worst_result))
            f.write("Worst cross-validation score: {:.3f}\n".format(
                mean_test_scores.min()))
            f.write("Best parameters: {}\n".format(best_result))
            f.write("Best cross-validation score: {:.3f}\n".format(
                mean_test_scores.max()))


def best_adaboost_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 15, 20, 30],
        'learning_rate': [0.1, 0.5, 1, 2, 5],
        'random_state': [0, 1, 2],
    }

    for estimator in [Estimator.PERCEPTRON, Estimator.DECISION_TREE, Estimator.SVC_LINEAR]:
        print("=================== Adaboost with Estimator: {} ====================================================".format(
            estimator.value["estimator"]))

        grid_search = GridSearchCV(
            AdaBoostClassifier(algorithm="SAMME", estimator=estimator.value["estimator"]), param_grid, cv=5, return_train_score=True, verbose=0)
        grid_search.fit(X_train, y_train)

        print_grid_search_results(
            grid_search, filename="./machine-learning/tmp/adaboost_{}.txt".format(estimator.name))


def worst_linear_svc_grid_search(X_train, y_train):
    print("=================== Linear SVC - Grid Search ====================================================")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
        'max_iter': [100, 200, 300, 400, 500],
        'random_state': [0, 1, 2],
    }

    # run grid search and find worst parameters
    grid_search = GridSearchCV(
        SVC(kernel="linear"), param_grid, cv=5, return_train_score=True, verbose=0)
    grid_search.fit(X_train, y_train)

    print_worst_grid_search_results(
        grid_search, filename="./machine-learning/tmp/linear_svc.txt")


def worst_perceptron_grid_search(X_train, y_train):
    print("=================== Perceptron - Grid Search ====================================================")
    param_grid = {
        'eta0': [0.001, 0.02, 10, 20],
        'max_iter': [2, 4, 6, 8, 10],
        'random_state': [0],
    }

    grid_search = GridSearchCV(
        SGDClassifier(loss="perceptron", learning_rate="constant"), param_grid, cv=5, return_train_score=True, verbose=0)
    grid_search.fit(X_train, y_train)

    print_worst_grid_search_results(
        grid_search, filename="./machine-learning/tmp/perceptron.txt")


def worst_decision_tree_grid_search(X_train, y_train):
    print("=================== Decision Tree - Grid Search ====================================================")
    param_grid = {
        'criterion': ['entropy'],
        'random_state': [0, 1, 2],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(max_depth=1), param_grid, cv=5, return_train_score=True, verbose=0)
    grid_search.fit(X_train, y_train)

    print_worst_grid_search_results(
        grid_search, filename="./machine-learning/tmp/decision_tree.txt")


if __name__ == "__main__":
    random_state = 1

    X, y = prepare_dataset(standardize=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    worst_perceptron_grid_search(X_train, y_train)
    worst_decision_tree_grid_search(X_train, y_train)
    worst_linear_svc_grid_search(X_train, y_train)

    best_adaboost_grid_search(X_train, y_train)
