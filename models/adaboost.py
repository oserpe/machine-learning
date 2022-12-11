
from sklearn.ensemble import AdaBoostClassifier

from ..utils.estimator import Estimator


def adaboost_classify(estimator: Estimator, X_train, X_test, y_train, random_state):
    subjacent_estimator = estimator.value["estimator"]
    subjacent_estimator.random_state = random_state

    adaboost = AdaBoostClassifier(estimator=subjacent_estimator, random_state=random_state,
                                  algorithm=estimator.value["algorithm"], n_estimators=100, learning_rate=0.1)

    adaboost.fit(X_train, y_train)

    return adaboost.predict(X_test)
