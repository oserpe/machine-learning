
from sklearn.ensemble import AdaBoostClassifier

from ..utils.estimator import Estimator


def adaboost_classify(estimator: Estimator, X_train, X_test, y_train, random_state, n_estimators=100, learning_rate=0.1, with_y_train_predictions = False):
    subjacent_estimator = estimator.value["estimator"]
    subjacent_estimator.random_state = random_state

    adaboost = AdaBoostClassifier(estimator=subjacent_estimator, random_state=random_state,
                                  algorithm=estimator.value["algorithm"], n_estimators=n_estimators, learning_rate=learning_rate)

    adaboost.fit(X_train, y_train)

    if with_y_train_predictions:
        return adaboost.predict(X_train), adaboost.predict(X_test)

    return adaboost.predict(X_test)
