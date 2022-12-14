from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

from .plots.classification_results import plot_metrics, n_k_fold_plot

from .models.estimator import estimator_classify

from .models.adaboost import adaboost_classify
from .dataset.utils import prepare_dataset
from .dataset.analysis import dataset_analysis
from .utils.estimator import Estimator
from .metrics.metrics import Metrics

result_column = "diagnosis"
result_column_labels = ["M", "B"]


def main(estimator: Estimator = Estimator.PERCEPTRON, use_adaboost: bool = False, random_state: int = 13):
    X, y = prepare_dataset(standardize=True)

    # standardize
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    # n_k_fold_plot(estimator, X, y, 2)
    # exit(1)

    if use_adaboost:
        y_pred = adaboost_classify(
            estimator, X_train, X_test, y_train, random_state)
    else:
        y_pred = estimator_classify(
            estimator, X_train, X_test, y_train, random_state)

    plot_metrics(y_test, y_pred)

def plot_accuracy_evolution(estimator: Estimator, random_state: int = 1):
    X, y = prepare_dataset(standardize=True)

    # standardize
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    accuracies_train = []
    accuracies_test = []
    estimators_range = [10, 25, 50, 100, 200, 500, 1000, 2500, 5000, 7500, 10000, 15000]
    for i in estimators_range:
        print(f"Estimators: {i}")
        y_train_pred, y_test_pred = adaboost_classify(
            estimator, X_train, X_test, y_train, random_state, n_estimators=i, with_y_train_predictions=True)
        cf_train = Metrics.get_confusion_matrix(
        y_train, y_train_pred, ["M", "B"])
        cf_test = Metrics.get_confusion_matrix(
        y_test, y_test_pred, ["M", "B"])
        accuracies_train.append((Metrics.get_accuracy_for_class(cf_train, "M") + Metrics.get_accuracy_for_class(cf_train, "B")) / 2)
        accuracies_test.append((Metrics.get_accuracy_for_class(cf_test, "M") + Metrics.get_accuracy_for_class(cf_test, "B")) / 2)

    # plot
    import matplotlib.pyplot as plt
    plt.plot(estimators_range, accuracies_train, marker='o', label="Train", color="blue")
    plt.plot(estimators_range, accuracies_test, marker='o', label="Test", color="red")
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/dataset/breast_cancer_wisconsin_data.csv", header=0, sep=',')
    # dataset_analysis(data_df)
    plot_accuracy_evolution(Estimator.DECISION_TREE)
    exit(1)
    estimator = Estimator.DECISION_TREE
    use_adaboost = True
    random_state = 1
    main(estimator=estimator,
         use_adaboost=use_adaboost, random_state=random_state)
