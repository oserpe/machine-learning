from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

from .models.estimator import estimator_classify

from .models.adaboost import adaboost_classify
from .dataset.utils import prepare_dataset

from .utils.estimator import Estimator
from .metrics.metrics import Metrics

result_column = "diagnosis"
result_column_labels = ["M", "B"]


def plot_metrics(y_test, y_pred):
    cf_matrix = Metrics.get_confusion_matrix(
        y_test, y_pred, result_column_labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    metrics, metrics_df = Metrics.get_metrics_per_class(cf_matrix)
    Metrics.plot_metrics_heatmap(metrics)


def n_k_fold_plot(estimator: Estimator, X, y, k):
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(
        X, y, 5, estimator.value["estimator"], k, X.columns.to_list(), result_column_labels)
    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics)


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

    # plot_tree(estimator.value["estimator"])
    plot_metrics(y_test, y_pred)


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/breast_cancer_wisconsin_data.csv", header=0, sep=',')
    # dataset_analysis(data_df)
    estimator = Estimator.DECISION_TREE
    use_adaboost = True
    random_state = 1
    main(estimator=estimator,
         use_adaboost=use_adaboost, random_state=random_state)
