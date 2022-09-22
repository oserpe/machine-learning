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

# def main_random_forest(train_dataset, test_dataset, class_column):
#     # create random forest
#     random_forest = RandomForest()

#     # train random forest
#     random_forest.train(train_dataset, class_column)

#     # test random forest
#     prediction_column = "Classification"
#     results = random_forest.test(test_dataset, prediction_column)

#     # print metrics
#     # get the prediction column values
#     y_predictions = results[prediction_column].values.tolist()

#     # get the class column values
#     y = list(map(str,results[class_column].values.tolist()))

#     labels = ["0", "1"]

#     cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
#     Metrics.plot_confusion_matrix_heatmap(cf_matrix)


def main(dataset: pd.DataFrame, tree_type: TreeType):
    tree = tree_type.get_tree()
    # drop continous columns
    continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    dataset = dataset.drop(continuous_columns, axis=1)

    class_column = "Creditability"

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.2)


    tree.train(train_dataset, class_column)

    if(tree_type == TreeType.DECISION_TREE):
        tree.draw()

    # test 
    prediction_column = "Classification"
    results = tree.test(test_dataset, prediction_column)

    # print metrics
    # get the prediction column values
    y_predictions = results[prediction_column].values.tolist()

    # get the class column values
    y = list(map(str,results[class_column].values.tolist()))

    labels = ["0", "1"]

    cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    tree_type = TreeType.RANDOM_FOREST
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
