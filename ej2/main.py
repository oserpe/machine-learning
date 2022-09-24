import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .models.KNN import KNN

def main(dataset):
    class_column = "Star Rating"

    knn = KNN(dataset, class_column, 15)
    other = dataset.iloc[[0]]
    other = other.drop([class_column], axis=1)
    for i in [0,50,100,150]:
        sample = dataset.iloc[[i]]
        dataset.drop([i], inplace=True)
        sample_without_class = sample.drop([class_column], axis=1)
        print(knn.test(sample_without_class))
        dataset = pd.concat([dataset,sample])
    return

if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej2/dataset/reviews_sentiment.csv", header=0, sep=';')

    data_df.dropna(inplace=True) # TODO: CAMBIAR CRITERIO?

    data_df.loc[data_df["titleSentiment"] == "positive", "titleSentiment"] = 1
    data_df.loc[data_df["titleSentiment"] == "negative", "titleSentiment"] = -1
    data_df.loc[data_df["textSentiment"] == "positive", "textSentiment"] = 1
    data_df.loc[data_df["textSentiment"] == "negative", "textSentiment"] = -1

    # drop text columns from df
    text_columns = ["Review Title", "Review Text"]
    numerical_data_df = data_df.drop(text_columns, axis=1)



    main(numerical_data_df)
