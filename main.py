from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

from .plots.classification_results import plot_metrics, n_k_fold_plot

from .models.estimator import estimator_classify

from .models.adaboost import adaboost_classify
from .dataset.utils import prepare_dataset
from .dataset.analysis import dataset_analysis
from .utils.estimator import Estimator

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


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/dataset/breast_cancer_wisconsin_data.csv", header=0, sep=',')
    dataset_analysis(data_df)
    exit(1)
    estimator = Estimator.DECISION_TREE
    use_adaboost = True
    random_state = 1
    main(estimator=estimator,
         use_adaboost=use_adaboost, random_state=random_state)
