from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import re as regex
from .metrics import Metrics
import copy


def plot_n_k_fold_cv_eval(X, y, n, model, k: int, X_features: list = None, y_features: list = None, classes: list = None):
    print("Processing N-K-Fold CV evaluation...")
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(
        X, y, n, model, k, X_features, y_features, classes)

    print("Plotting N-K-Fold CV evaluation...")
    Metrics.plot_metrics_heatmap_std(
        avg_metrics, std_metrics, plot_title=f'K Fold Cross Validation Evaluation')


def plot_cf_matrix(y, y_predicted, labels=None):
    cf_matrix = Metrics.get_confusion_matrix(y, y_predicted, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)
    return cf_matrix


def plot_curves_with_legend(inputs, outputs, legends=None, X_label="X", Y_label="Y"):
    colors = sns.color_palette("hls", len(outputs))
    for i in range(len(outputs)):
        if legends is not None:
            plt.plot(inputs, outputs[i], color=colors[i],
                     label=legends[i])
        else:
            plt.plot(inputs, outputs[i], color=colors[i])

    plt.legend()
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()


def plot_kohonen_matrix_predictions(kohonen_model, y_values, kohonen_predictions, classes):
    K = kohonen_model.K

    x_indexes = [pos // K for pos in kohonen_predictions]
    y_indexes = [pos % K for pos in kohonen_predictions]

    # Initialize the dict matrix
    keys = classes
    kohonen_matrix = np.empty((K, K), dtype=object)
    for x in range(K):
        for y in range(K):
            kohonen_matrix[x][y] = {key: 0 for key in keys}

    # Fill the matrix
    for x, y, genre in zip(x_indexes, y_indexes, y_values):
        kohonen_matrix[x][y][genre] += 1

    hit_matrix = copy.deepcopy(kohonen_matrix)

    # Convert every dict to str
    for x in range(K):
        for y in range(K):
            # Get the percentage for each genre
            total = sum(kohonen_matrix[x][y].values())
            for idx, key in enumerate(keys):
                probability = 0
                percentage = probability

                if total != 0:
                    probability = kohonen_matrix[x][y][key] / total
                    percentage = round(probability * 100, 2)

                kohonen_matrix[x][y][
                    key] = f'{kohonen_matrix[x][y][key]} - {percentage}%'

            value = str(kohonen_matrix[x][y]).replace(",", "\n")
            value = regex.sub(r"[{}']", "", value)
            kohonen_matrix[x][y] = value

    # Plot heatmap
    kohonen_matrix_dummy = np.ones((K, K))
    sns.heatmap(kohonen_matrix_dummy, cmap="Greys_r", annot=kohonen_matrix, fmt="s",
                cbar=False, linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()

    # Plot hit heatmaps
    plot_kohonen_heatmaps_hit(hit_matrix)


def plot_kohonen_heatmaps_hit(hit_matrix):
    # Given a matrix where each cell is a dict with the number of hits for each genre, plot a heatmap
    # for each genre
    K = hit_matrix.shape[0]
    keys = list(hit_matrix[0][0].keys())

    fig, axs = plt.subplots(1, len(keys))
    # fig.suptitle('Genres Kohonen Heatmaps')

    # colors = ["OrRd", "BuGn", "PuBu"]

    for i in range(len(keys)):
        genre = keys[i]
        genre_matrix = np.zeros((K, K))

        for x in range(K):
            for y in range(K):
                genre_matrix[x][y] = hit_matrix[x][y][genre]

        sns.heatmap(genre_matrix, ax=axs[i], cmap="coolwarm",
                    linewidths=0.5, xticklabels=False, yticklabels=False)
        axs[i].set_title(genre)

    plt.show()


def plot_metrics_heatmap(cf_matrix):
    metrics_dict, metrics_df = Metrics.get_metrics_per_class(cf_matrix)
    Metrics.plot_metrics_heatmap(metrics_dict)
    return metrics_dict, metrics_df
