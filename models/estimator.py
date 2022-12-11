from ..utils.estimator import Estimator


def estimator_classify(estimator: Estimator, X_train, X_test, y_train, random_state):
    estimator = estimator.value["estimator"]
    estimator.random_state = random_state

    estimator.fit(X_train, y_train)

    return estimator.predict(X_test)
