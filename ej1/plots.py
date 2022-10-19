from ..models.Metrics import Metrics


def plot_n_k_fold_cv_eval(x, y, n, model, k: int, x_column_names: list = None, y_column_names: list = None):
    model.classes = [1, -1] # FIXME: HARDCODED
    avg_metrics, std_metrics = Metrics.n_k_fold_cross_validation_eval(x, y, n, model, k, x_column_names, y_column_names)
    Metrics.plot_metrics_heatmap_std(avg_metrics, std_metrics, plot_title=f'K Fold Cross Validation Evaluation')

# TODO: move to utils.py
def generate_line_interval(m, b, interval):
    return [m * interval[0] + b, m * interval[1] + b]

def get_animation_function(model, X, y, interval, ax, title):
    def animate(i):
        ax.clear()
        w_0 = model.w_list[i][0]
        w_1 = model.w_list[i][1]
        b = model.b_list[i]

        # (w0, w1) * (x, y) + b = 0
        # w0 x + w1 y + b = 0
        # y = -w0/w1 x - b/w1
        ax.scatter(X[:, 0], X[:, 1], c=y)

        m_hat = -w_0 / w_1
        b_hat = -b / w_1

        ax.set_title(title)
        ax.plot(interval, generate_line_interval(
            m_hat, b_hat, interval), color='green', label='Predicted')

        ax.set_xlim(interval)
        ax.set_ylim(interval)
    
    return animate