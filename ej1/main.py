import pandas as pd

from .models.TreeType import TreeType
from .models.DecisionTree import DecisionTree
from .models.RandomForest import RandomForest
from ..metrics import Metrics


def categorize_data_with_equal_frequency(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, q in columns.items():
        data_df[column], range = pd.qcut(
            data_df[column], q=q, labels=False, retbins=True)
        # print(range) # print ranges of current discretization, starting from minimum value until the last range that ends on the maximum value
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


def main_k_fold(dataset: pd.DataFrame):
    tree = DecisionTree()

    # drop continous columns
    continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]

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
    Metrics.avg_and_std_metrics_to_csv(tree.classes, avg_metrics, std_metrics, path="./machine-learning/ej1/dump/avg_std_metrics.csv")
     


def main(dataset: pd.DataFrame, tree_type: TreeType):
    tree = tree_type.get_tree()
    # drop continous columns
    # continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    # dataset = dataset.drop(continuous_columns, axis=1)

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.2)


    # tree.train(train_dataset)

    # if(tree_type == TreeType.DECISION_TREE):
    #     tree.draw()

    # # prune
    # tree.prune(test_dataset)

    # # test 
    # results = tree.test(test_dataset)

    # # print metrics
    # # get the prediction column values
    # y_predictions = results[tree.predicted_class_column_name].values.tolist()

    # # get the class column values
    # y = results[tree.classes_column_name].values.tolist()

    # labels = [0, 1]

    # cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    # Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    # print s-precision plot
    results = tree.s_precision_per_node_count(train_dataset, test_dataset, initial_node_count=500, max_node_count=1000)
    print(results)
    print(pd.DataFrame.from_dict(results, orient='index'))
    # save to .csv
    pd.DataFrame.from_dict(results, orient='index').to_csv(f'./machine-learning/ej1/dump/{tree.tree_type}_s_precision.csv')
    # tree.plot_precision_per_node_count(results)

def gender_study(dataset_df: pd.DataFrame):
    gender_column = "Sex & Marital Status"
    female_data_df = data_df[data_df[gender_column] == 4]
    male_data_df = data_df[data_df[gender_column] != 4]

    print("Porcentaje de hombres que piden un prestamo sobre el total: ",\
        round(len(male_data_df)/len(dataset_df),2))

    occupation_column = "Occupation"
    executive_occupation_id = 4 
    print("Porcentaje de mujeres en puestos ejecutivos sobre el total de ejecutivos: ", \
        round(female_data_df.groupby(occupation_column).size()[executive_occupation_id]/  \
            dataset_df.groupby(occupation_column).size()[executive_occupation_id], 2))
    print("En las mujeres el porcentaje que ocupa puestos ejecutivos es: ", \
        round(female_data_df.groupby(occupation_column).size()[executive_occupation_id]/  \
            len(female_data_df), 2))
    print("En los hombres el porcentaje que ocupa puestos ejecutivos es: ", \
        round(male_data_df.groupby(occupation_column).size()[executive_occupation_id]/  \
            len(male_data_df), 2))
    
    amount_column = "Credit Amount"
    amount_separator = 4
    print(f"En las mujeres el porcentaje de las que piden prestamos menores a la categoria {amount_separator} es: ", \
        round(len(female_data_df[female_data_df[amount_column] < amount_separator]) /\
            len(female_data_df),2))
    print(f"En los hombres el porcentaje de los que piden prestamos menores a la categoria {amount_separator} es: ", \
        round(len(male_data_df[male_data_df[amount_column] < amount_separator]) /\
            len(male_data_df),2))

    most_valuable_asset_column = "Most valuable available asset"    
    print(f"En las mujeres el porcentaje de las que no tienen most valuable asset es: ", \
        round(len(female_data_df[female_data_df[most_valuable_asset_column] <= 1]) /\
            len(female_data_df),2))
    print(f"En los hombres el porcentaje de los que no tienen most valuable asset es: ", \
        round(len(male_data_df[male_data_df[most_valuable_asset_column] <= 1]) /\
            len(male_data_df),2))

    age_column = "Age (years)"
    current_employment = "Length of current employment"
    saves_column = "Value Savings/Stocks"
    creditability_column = "Creditability"
    duration_column = "Duration of Credit (month)"
    print(female_data_df.groupby(duration_column).size())
    print(male_data_df.groupby(duration_column).size())

if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    #main_test_and_plot_cf_matrix_random_forest_trees(data_df, n_estimators=3)
    tree_type = TreeType.RANDOM_FOREST
    categorical_columns = {
        # Quantity picked arbitrarily for this dataset taking into account that it separates the data in categories of the closest amount
        "Duration of Credit (month)": 6,
        "Credit Amount": 10,
        "Age (years)": 7
    }

    data_df = categorize_data_with_equal_frequency(
        data_df, categorical_columns)

    # print(data_df["Duration of Credit (month)"].value_counts())
    # print(data_df["Credit Amount"].value_counts())
    # print(data_df["Age (years)"].value_counts())

    # gender_study(data_df)
    # main(data_df, tree_type)
    main_k_fold(data_df)
