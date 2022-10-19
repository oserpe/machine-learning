from ..models.Metrics import Metrics


def plot_n_k_fold_cv_eval(x, y, n, model, k: int, x_column_names: list = None, y_column_names: list = None):
    # Get ImageClasses possible values
    model.classes = [0,1,2]
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(x, y, n, model, k, x_column_names, y_column_names)
    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics, plot_title=f'K Fold Cross Validation Evaluation')