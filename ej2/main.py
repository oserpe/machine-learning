from itertools import groupby
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .models.KNN import KNN
from ..metrics import Metrics
import matplotlib.pyplot as plt


def create_rate_boxplot(dataset, class_column):
    groupby_data_df = dataset.groupby(class_column)

    classification = []
    labels = []
    for rate in groupby_data_df.groups.keys():
        classification.append(groupby_data_df.get_group(
            rate)['wordcount'].values)
        labels.append(rate)
        print(groupby_data_df.get_group(rate)['wordcount'].mean())

    fig, axes = plt.subplots(figsize=(10, 5))
    axes.boxplot(classification)
    axes.set_xticks(range(1, len(labels) + 1), labels)
    axes.set_title("Boxplot de cantidad de palabras por rate")

    plt.show()


def create_k_evolution_boxplot(dataset, class_column):
    # get train and test dataset
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.3)

    # get train dataset x and y values
    x_train = train_dataset.drop(class_column, axis=1)
    y_train = train_dataset[[class_column]]

    x_test = test_dataset.drop(class_column, axis=1)
    y_test = test_dataset[[class_column]]

    labels = [1, 2, 3, 4, 5]
    k_range = range(1, 70, 2)
    # colors for methods
    colors = ["red", "blue", "green", "orange", "purple",
              "brown", "pink", "gray", "olive", "cyan"]
    for index, weighted in enumerate([True, False]):
        precision = []
        for K in k_range:
            # load the model
            knn = KNN(x_train, y_train, k_neighbors=K, weighted=weighted)
            # test the model
            results = knn.test(x_test)
            # get metrics
            y_predictions = results[knn.predicted_class_column_name].values.tolist(
            )
            y = y_test[knn.class_column].values.tolist()
            cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
            metrics, metrics_df = Metrics.get_metrics_per_class(cf_matrix)

            precision.append(metrics[1]["precision"])

        plt.plot(list(k_range), precision, label="Weighted" if weighted else "Not weighted",
                 linestyle='--', marker='o', color=colors[(index*2) % len(colors)])

    plt.legend()
    plt.xlabel("Neighbours considered")
    plt.ylabel("Precision")
    plt.show()


def main_k_fold(dataset):
    # load the model
    knn = KNN([], [], k_neighbors=20, weighted=False)
    rand = random.randint(1, 10000)
    dataset = dataset.sample(frac=1, random_state=rand).reset_index(drop=True)

    # get dataset x and y values
    x = dataset.drop(knn.classes_column_name, axis=1)
    y = dataset[[knn.classes_column_name]]

    results, errors, metrics, avg_metrics, std_metrics = Metrics.k_fold_cross_validation_eval(
        x.values.tolist(), y.values.tolist(),
        model=knn, x_column_names=x.columns, y_column_names=y.columns, k=5)

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
        knn.classes, avg_metrics, std_metrics, path="./machine-learning/ej2/dump/avg_std_metrics.csv")


def main(dataset):
    class_column = "Star Rating"

    # get train and test dataset
    train_dataset, test_dataset = Metrics.holdout(dataset, test_size=0.3)

    # get train dataset x and y values
    x_train = train_dataset.drop(class_column, axis=1)
    y_train = train_dataset[[class_column]]

    # load the model
    knn = KNN(x_train, y_train, k_neighbors=20, weighted=True)

    x_test = test_dataset.drop(class_column, axis=1)
    y_test = test_dataset[[class_column]]

    # test the model
    results = knn.test(x_test)

    # get confusion matrix
    y_predictions = results[knn.predicted_class_column_name].values.tolist()
    y = y_test[knn.class_column].values.tolist()
    labels = [1, 2, 3, 4, 5]

    print("Y predictions:")
    print(y_predictions)

    print("Y:")
    print(y)

    cf_matrix = Metrics.get_confusion_matrix(y, y_predictions, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    # get metrics
    metrics, metrics_df = Metrics.get_metrics_per_class(cf_matrix)

    print("Metrics:")
    print(metrics)

    # save to csv
    metrics_df.to_csv("./machine-learning/ej2/dump/metrics.csv")

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


def knn_impute(dataset, class_column, k):
    complete_df = dataset[~dataset.isna().any(axis=1)]
    incomplete_df = dataset[dataset.isna().any(axis=1)]

    # get train dataset x and y values
    x_train = complete_df.drop(class_column, axis=1)
    y_train = complete_df[[class_column]]

    # load the model
    knn = KNN(x_train, y_train, k_neighbors=k, weighted=True,
              classes_column_name=class_column)

    # get incomplete without NaN column
    incomplete_df = incomplete_df.drop(class_column, axis=1)

    # add predicted class column
    incomplete_df[class_column] = knn.test(
        incomplete_df)[knn.predicted_class_column_name]

    # concat complete and incomplete
    imputed_df = pd.concat([complete_df, incomplete_df], axis=0)

    return imputed_df


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej2/dataset/reviews_sentiment.csv", header=0, sep=';')

    data_df.loc[data_df["titleSentiment"] == "positive", "titleSentiment"] = 1
    data_df.loc[data_df["titleSentiment"] == "negative", "titleSentiment"] = -1
    data_df.loc[data_df["textSentiment"] == "positive", "textSentiment"] = 1
    data_df.loc[data_df["textSentiment"] == "negative", "textSentiment"] = -1

    # drop text columns from df
    text_columns = ["Review Title", "Review Text"]
    numerical_data_df = data_df.drop(text_columns, axis=1)

    missing_data_column = "titleSentiment"

    imputed_df = knn_impute(numerical_data_df, missing_data_column, k=20)

    # main(data_completed_df)
    # main_k_fold(data_completed_df)
    # create_rate_boxplot(data_completed_df, "Star Rating")
    create_k_evolution_boxplot(imputed_df, "Star Rating")
