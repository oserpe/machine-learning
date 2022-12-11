from ..metrics.metrics import Metrics
from ..utils.estimator import Estimator
from ..main import result_column_labels

def plot_metrics(y_test, y_pred):
    cf_matrix = Metrics.get_confusion_matrix(
        y_test, y_pred, result_column_labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    metrics, metrics_df = Metrics.get_metrics_per_class(cf_matrix)
    Metrics.plot_metrics_heatmap(metrics)


def n_k_fold_plot(estimator: Estimator, X, y, k):
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(
        X, y, 5, estimator.value["estimator"], k, X.columns.to_list(), result_column_labels)
    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics)