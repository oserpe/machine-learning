from enum import Enum
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Estimator(Enum):
    SVC_LINEAR = {"estimator": SVC(
        kernel="linear", max_iter=100, C=0.001), "algorithm": "SAMME"}
    DECISION_TREE = {"estimator": DecisionTreeClassifier(
        max_depth=1), "algorithm": "SAMME.R"}
    PERCEPTRON = {"estimator": SGDClassifier(
        loss="perceptron", max_iter=4, learning_rate="constant", eta0=10), "algorithm": "SAMME"}
