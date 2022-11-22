from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import re as regex
from .metrics import Metrics

def plot_n_k_fold_cv_eval(X, y, n, model, k: int, X_features: list = None, y_feature: str = None, classes: list = None):
    print("Processing N-K-Fold CV evaluation...")
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(
        X, y, n, model, k, X_features, y_feature, classes)
    
    print("Plotting N-K-Fold CV evaluation...")
    Metrics.plot_metrics_heatmap_std(
        avg_metrics, std_metrics, plot_title=f'K Fold Cross Validation Evaluation')

def plot_cf_matrix(y, y_predicted, labels = None): 
    cf_matrix = Metrics.get_confusion_matrix(y, y_predicted, labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

def plot_kohonen_matrix_predictions(kohonen_model, y_predictions, kohonen_predictions, classes):
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
    for x, y, genre in zip(x_indexes, y_indexes, y_predictions):
        kohonen_matrix[x][y][genre] += 1

    # Convert every dict to str
    for x in range(K):
        for y in range(K):
            value = str(kohonen_matrix[x][y]).replace(",", "\n")
            value = regex.sub(r"[{}']", "", value)
            kohonen_matrix[x][y] = value

    # Plot heatmap
    kohonen_matrix_dummy = np.ones((K, K))
    sns.heatmap(kohonen_matrix_dummy, cmap="Greys_r", annot=kohonen_matrix, fmt="s",
                cbar=False, linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()
