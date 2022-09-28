import pandas as pd

from .gender_study import gender_study

from .models.TreeType import TreeType
from .models.DecisionTree import DecisionTree
from .models.RandomForest import RandomForest
from ..metrics import Metrics
import matplotlib.pyplot as plt


def categorize_data_with_equal_frequency(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    categorized_data = data_df.copy()

    for column, q in columns.items():
        categorized_data[column], range = pd.qcut(
            data_df[column], q=q, labels=False, retbins=True)
        # print(range) # print ranges of current discretization, starting from minimum value until the last range that ends on the maximum value
    return categorized_data, range


def categorize_data_with_equal_width(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, bins in columns.items():
        data_df[column] = pd.cut(
            data_df[column], bins, labels=False)

    return data_df


def main_test_and_plot_cf_matrix_random_forest_trees(dataset: pd.DataFrame, n_estimators: int = 10, samples_per_bag_frac: float = 1):
    tree = RandomForest(n_estimators, samples_per_bag_frac)

    # drop continous columns
    continuous_columns = [
        "Duration of Credit (month)", "Credit Amount", "Age (years)"]
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
        Metrics.plot_confusion_matrix_heatmap(
            cf_matrix, plot_title=f"Confusion matrix for tree {index+1}")


def main_k_fold(dataset: pd.DataFrame):
    tree = DecisionTree()

    # drop continous columns
    continuous_columns = [
        "Duration of Credit (month)", "Credit Amount", "Age (years)"]

    dataset = dataset.drop(continuous_columns, axis=1)

    # get dataset without class column
    x = dataset.drop(tree.classes_column_name, axis=1)

    # get dataset dataframe with only class column
    y = dataset[[tree.classes_column_name]]

    # call k-fold cross validation
    results, errors, metrics, avg_metrics, std_metrics = Metrics.k_fold_cross_validation_eval(x.values.tolist(), y.values.tolist(
    ), model=tree, x_column_names=x.columns, y_column_names=y.columns, k=5)

    # print results
    print("Results:")
    print(results)

    print("Metrics:")
    print(metrics)

    print("Average metrics:")
    print(avg_metrics)

    print("Standard deviation metrics:")
    print(std_metrics)

    # save to csv
    Metrics.avg_and_std_metrics_to_csv(
        tree.classes, avg_metrics, std_metrics, path="./machine-learning/ej1/dump/avg_std_metrics.csv")


def s_precision(y, y_predictions):
    positives_count = 0
    for y_pred, y_true in zip(y_predictions, y):
        if y_pred == y_true:
            positives_count += 1

    return positives_count / len(y_predictions)

def main_n_estimators_rf(dataset: pd.DataFrame):

    class_column = "Creditability"

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.2)

    estimators = [2, 5, 10, 20, 30, 40]

    s_test_precisions = []
    s_train_precisions = []

    prediction_column = "Classification"

    for n_estimators in estimators:
        tree = RandomForest(n_estimators)

        tree.train(train_dataset, class_column)

        # test training set
        results = tree.test(train_dataset, prediction_column)

        # get the prediction column values
        y_predictions = results[prediction_column].values.tolist()

        # get the class column values
        y = results[class_column].values.tolist()

        s_train_precisions.append(s_precision(y, y_predictions))

        # test
        results = tree.test(test_dataset, prediction_column)

        # get the prediction column values
        y_predictions = results[prediction_column].values.tolist()

        # get the class column values
        y = results[class_column].values.tolist()

        s_test_precisions.append(s_precision(y, y_predictions))


    # plot precisions
    plt.plot(estimators, s_train_precisions, label="Train",
                linestyle='--', marker='o', color='r')

    plt.plot(estimators, s_test_precisions, label="Test",
                 linestyle='--', marker='o', color='b')
    plt.legend()
    plt.xlabel("Tree count")
    plt.ylabel("Precision")
    plt.title("Precision vs tree count")
    plt.ylim(top=1.1)
    plt.show()


def main(dataset: pd.DataFrame, tree_type: TreeType):
    tree = tree_type.get_tree()

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.3)

    tree.train(train_dataset)

    # prune
    # tree.prune(test_dataset)

    if(tree_type == TreeType.DECISION_TREE):
        tree.draw()

    # test 
    results = tree.test(test_dataset)
    
    # print metrics
    # get the prediction column values
    y_predictions = results[tree.predicted_class_column_name].values.tolist()

    # get the class column values
    y = results[tree.classes_column_name].values.tolist()

    labels = [0, 1]

    cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    # print s-precision plot
    results = tree.s_precision_per_node_count(
        train_dataset, test_dataset, initial_node_count=500, max_node_count=1000)
    print(results)
    print(pd.DataFrame.from_dict(results, orient='index'))
    # save to .csv
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        f'./machine-learning/ej1/dump/{tree.tree_type}_s_precision.csv')
    # tree.plot_precision_per_node_count(results)


def main_n_k_fold(dataset: pd.DataFrame):
    tree = DecisionTree()

    # get dataset without class column
    x = dataset.drop(tree.classes_column_name, axis=1)

    # get dataset dataframe with only class column
    y = dataset[[tree.classes_column_name]]

    # call n-k-fold cross validation
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(x.values.tolist(), y.values.tolist(
    ), model=tree, x_column_names=x.columns, y_column_names=y.columns, n=1, k=5)

    # print results
    print("Average metrics:")
    print(avg_metrics)

    print("Standard deviation metrics:")
    print(std_metrics)

    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics)

    # save to csv
    Metrics.avg_and_std_metrics_to_csv(
        tree.classes, avg_metrics, std_metrics, path=f"./machine-learning/ej1/dump/{tree.tree_type}_n_avg_std_metrics.csv")

def plot_preprune_methods_accuracy(dataset):
    tree = DecisionTree()
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.3)

    method_names = ["Max depth", "Max node count"]
    results_list = [
        tree.s_precision_per_depth(train_dataset, test_dataset),
        tree.s_precision_per_node_count(train_dataset, test_dataset),
    ]

    tree.plot_precision_per_node_count_multiple_results(results_list, method_names)


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    #main_test_and_plot_cf_matrix_random_forest_trees(data_df, n_estimators=3)
    tree_type = TreeType.DECISION_TREE
    categorical_columns = {
        # Quantity picked arbitrarily for this dataset taking into account that it separates the data in categories of the closest amount
        "Duration of Credit (month)": 6,
        "Credit Amount": 10,
        "Age (years)": 7
    }

    data_df, range = categorize_data_with_equal_frequency(
        data_df, categorical_columns)

    # print(data_df["Duration of Credit (month)"].value_counts())
    # print(data_df["Credit Amount"].value_counts())
    # print(data_df["Age (years)"].value_counts())

    # main(data_df, tree_type)
    # main_k_fold(data_df)
    # plot_preprune_methods_accuracy(data_df)
    # main_n_k_fold(data_df)
    # gender_study(data_df)
    main_n_estimators_rf(data_df)
