from turtle import title
import pandas as pd

from .models.TreeType import TreeType
from .models.DecisionTree import DecisionTree
from .models.RandomForest import RandomForest
from ..metrics import Metrics


def categorize_data_with_equal_frequency(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, q in columns.items():
        data_df[column] = pd.qcut(
            data_df[column], q=q, labels=False, duplicates='drop')

    return data_df


def categorize_data_with_equal_width(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, bins in columns.items():
        data_df[column] = pd.cut(
            data_df[column], bins, labels=False)

    return data_df

def main_test_and_plot_cf_matrix_random_forest_trees(dataset: pd.DataFrame, n_estimators: int = 10, samples_per_bag_frac: float = 1):
    tree = RandomForest(n_estimators, samples_per_bag_frac)

    # drop continous columns
    continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    dataset = dataset.drop(continuous_columns, axis=1)

    class_column = "Creditability"

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.1)


    tree.train(train_dataset, class_column)

    # test 
    prediction_column = "Classification"
    results_per_tree = tree.test_every_tree(test_dataset, prediction_column)

    labels = [0, 1]

    # print metrics per tree
    for index, results in enumerate(results_per_tree):
        # get the prediction column values
        y_predictions = results[prediction_column].values.tolist()

        # get the class column values
        y = results[class_column].values.tolist()

        cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
        Metrics.plot_confusion_matrix_heatmap(cf_matrix, plot_title=f"Confusion matrix for tree {index+1}")


def main(dataset: pd.DataFrame, tree_type: TreeType):
    tree = tree_type.get_tree()
    # drop continous columns
    continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    dataset = dataset.drop(continuous_columns, axis=1)

    class_column = "Creditability"

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.7)


    tree.train(train_dataset, class_column)

    if(tree_type == TreeType.DECISION_TREE):
        tree.draw()

    # prune
    # tree.prune(test_dataset)

    # test 
    prediction_column = "Classification"
    results = tree.test(test_dataset, prediction_column)

    # print metrics
    # get the prediction column values
    y_predictions = results[prediction_column].values.tolist()

    # get the class column values
    y = results[class_column].values.tolist()

    labels = [0, 1]

    cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    #main_test_and_plot_cf_matrix_random_forest_trees(data_df, n_estimators=3)
    tree_type = TreeType.DECISION_TREE
    main(data_df, tree_type)

    # categorical_columns = {
    #     "Duration of Credit (month)": 12,
    #     "Credit Amount": 10,
    #     "Age (years)": 10
    # }

    # print(data_df)

    # data_df = categorize_data_with_equal_frequency(
    #     data_df, categorical_columns)

    # print(data_df)
    # print(data_df["Duration of Credit (month)"].value_counts())
    # print(data_df["Credit Amount"].value_counts())
    # print(data_df["Age (years)"].value_counts())
