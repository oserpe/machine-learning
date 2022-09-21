import pandas as pd
from .models.DecisionTree import DecisionTree


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

def main(dataset: pd.DataFrame):
    # drop continous columns
    continuous_columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    dataset = dataset.drop(continuous_columns, axis=1)

    class_column = "Creditability"
    decision_tree = DecisionTree()

    decision_tree.train(dataset, class_column)

    decision_tree.draw()




if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    main(data_df)

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
