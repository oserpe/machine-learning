import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..main import knn_impute


def plot_general_histogram(dataset: pd.DataFrame):
    print(dataset[["titleSentiment"]])
    df = dataset.apply(pd.to_numeric)
    df.hist(edgecolor='black', linewidth=1.0,
            xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej2/dataset/reviews_sentiment.csv", header=0, sep=';')

    print(data_df)

    data_df.loc[data_df["titleSentiment"] ==
                "positive", "titleSentiment"] = 1.0
    data_df.loc[data_df["titleSentiment"] ==
                "negative", "titleSentiment"] = -1.0
    data_df.loc[data_df["textSentiment"] == "positive", "textSentiment"] = 1.0
    data_df.loc[data_df["textSentiment"] == "negative", "textSentiment"] = -1.0

    # drop text columns from df
    text_columns = ["Review Title", "Review Text"]
    numerical_data_df = data_df.drop(text_columns, axis=1)

    imputed_df = knn_impute(numerical_data_df, "titleSentiment", 20)
    print(imputed_df[["titleSentiment"]].value_counts())
    print(type(imputed_df["titleSentiment"].sample(1).values[0]))
    print(imputed_df)
    plot_general_histogram(imputed_df)
    plt.show(block=True)
