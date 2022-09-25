import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .models.KNN import KNN


def main(dataset):
    class_column = "Star Rating"
    X = dataset.drop(class_column, axis=1)
    Y = dataset[[class_column]]

    other = dataset.iloc[[0]]
    other = other.drop([class_column], axis=1)
    for i in [0, 50, 100, 200]:
        sample = dataset.iloc[[i]]
        dataset.drop([i], inplace=True)
        knn = KNN(X, Y, 2, weighted=False)

        sample_without_class = sample.drop([class_column], axis=1)
        print("sample: ")
        print(sample_without_class)
        print("rating expected: ", knn.classify(sample_without_class))

        # restore dataset
        dataset = pd.concat([dataset, sample])
    return


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej2/dataset/reviews_sentiment.csv", header=0, sep=';')

    # TODO: CAMBIAR CRITERIO PARA REVIEWS CON DATOS FALTANTES?
    data_df.dropna(inplace=True)

    data_df.loc[data_df["titleSentiment"] == "positive", "titleSentiment"] = 1
    data_df.loc[data_df["titleSentiment"] == "negative", "titleSentiment"] = -1
    data_df.loc[data_df["textSentiment"] == "positive", "textSentiment"] = 1
    data_df.loc[data_df["textSentiment"] == "negative", "textSentiment"] = -1

    # drop text columns from df
    text_columns = ["Review Title", "Review Text"]
    numerical_data_df = data_df.drop(text_columns, axis=1)

    main(numerical_data_df)
