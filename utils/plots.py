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