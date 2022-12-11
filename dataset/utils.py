import pandas as pd
from sklearn.preprocessing import StandardScaler

result_column = "diagnosis"
result_column_labels = ["M", "B"]


def prepare_dataset(standardize=True, convert_to_binary=True):
    data_df = pd.read_csv(
        "./machine-learning/dataset/breast_cancer_wisconsin_data.csv", header=0, sep=',')

    # drop duplicates by id
    data_df.drop_duplicates(subset=["id"], keep="first", inplace=True)

    # drop non-interesting columns
    data_df.drop("id", inplace=True, axis=1)

    # drop rows with missing values
    data_df.dropna(inplace=True)

    y = data_df[result_column]
    X = data_df.drop(result_column, axis=1)

    if convert_to_binary:
        y = y.replace("M", 1)
        y = y.replace("B", 0)

    if standardize:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X, y
