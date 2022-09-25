import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .models.KNN import KNN
from ..metrics import Metrics

def main_k_fold(dataset):
    # load the model
    knn = KNN([], [], k_neighbors=5, weighted=True)


    # get dataset x and y values
    x = dataset.drop(knn.classes_column_name, axis=1)
    y = dataset[[knn.classes_column_name]]

    results, errors, metrics, avg_metrics, std_metrics = Metrics.k_fold_cross_validation_eval(x.values.tolist(), y.values.tolist(
    ), model=knn, x_column_names=x.columns, y_column_names=y.columns, k=5)

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
    Metrics.avg_and_std_metrics_to_csv(knn.classes, avg_metrics, std_metrics, path="./machine-learning/ej2/dump/avg_std_metrics.csv")


def main(dataset):
    class_column = "Star Rating"

    # get train and test dataset
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.1)

    # get train dataset x and y values
    x_train = train_dataset.drop(class_column, axis=1)
    y_train = train_dataset[[class_column]]

    # load the model
    knn = KNN(x_train, y_train, k_neighbors=100, weighted=True)

    # test the model
    results = knn.test(test_dataset)

    # get confusion matrix
    y_predictions = results[knn.predicted_class_column_name].values.tolist()
    y = results[knn.classes_column_name].values.tolist()
    labels = [1, 2, 3, 4, 5]

    print("Y predictions:")
    print(y_predictions)

    print("Y:")
    print(y)

    cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)


    # X = dataset.drop(class_column, axis=1)
    # Y = dataset[[class_column]]

    # other = dataset.iloc[[0]]
    # other = other.drop([class_column], axis=1)
    # for i in [0, 50, 100, 200]:
    #     sample = dataset.iloc[[i]]
    #     dataset.drop([i], inplace=True)
    #     knn = KNN(X, Y, 2, weighted=False)

    #     sample_without_class = sample.drop([class_column], axis=1)
    #     print("sample: ")
    #     print(sample_without_class)
    #     print("rating expected: ", knn.classify(sample_without_class))

    #     # restore dataset
    #     dataset = pd.concat([dataset, sample])


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

    # main(numerical_data_df)
    main_k_fold(numerical_data_df)
