from enum import Enum
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Estimator(Enum):
    SVC_LINEAR = {"estimator": SVC(kernel="linear"), "algorithm": "SAMME"}
    DECISION_TREE = {"estimator": DecisionTreeClassifier(
        max_depth=1), "algorithm": "SAMME.R"}
    PERCEPTRON = {"estimator": Perceptron(), "algorithm": "SAMME"}
