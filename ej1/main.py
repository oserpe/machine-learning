import math
from turtle import left
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import itertools
from sklearn.metrics import classification_report, confusion_matrix

from ..models.Metrics import Metrics

from ..utils import execute_grid_search_cv

from .plots import get_animation_function, plot_n_k_fold_cv_eval, plot_error_by_epoch

from ..models.SVM import SVM
from ..models.ThreePointsSVM import ThreePointsSVM
from ..models.SimplePerceptron import SimplePerceptron
from .dataset.generate_dataset import generate_linearly_separable, generate_not_linearly_separable
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# TODO: move to utils.py
def generate_line_interval(m, b, interval):
    return [m * interval[0] + b, m * interval[1] + b]


def plot_ej_a(X_train, y_train, X_test, y_test, m, b, interval, seed, animate=False):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=1000, max_epochs=1000,
                                  tol=0.01, random_state=seed, verbose=False)

    perceptron.fit(X_train, y_train)

    print("Error: ", perceptron.error_)
    print("W: ", perceptron.w_)
    print("B: ", perceptron.b_)

    if animate:
        fig, ax = plt.subplots(1, 1)

        # Plot the hyperplane animation
        animate_func = get_animation_function(perceptron, X_train, y_train, interval, ax, "Perceptron classification with linearly separable dataset")
        anim = animation.FuncAnimation(fig, animate_func, frames=len(
            perceptron.w_list), repeat=False, interval=25)
        
        anim.save('./machine-learning/ej1/dump/linearly_dataset_perceptron_animation.gif', fps=2)
        plt.show()   
    else:
        plot_data(X_train, y_train, interval, [[*perceptron.w_, perceptron.b_]],
              title="Perceptron classification with linearly separable dataset", colors=['green'], labels=['Predicted']) 

    # PLOT: n k fold cross validation
    # plot_n_k_fold_cv_eval(X, y, 5, perceptron, k=5)

    # PLOT: error by epoch
    # y_test_error = perceptron.compute_error_by_epoch(X_test, y_test)
    # # Add epochs number 0
    # epochs = np.arange(0, perceptron.epochs + 1)
    # plot_error_by_epoch(epochs, perceptron.error_epoch_list, y_test_error)

    # PLOT: confusion matrix
    y_pred = perceptron.predict(X_test)
    cf_matrix = Metrics.get_confusion_matrix(y_test, y_pred, [-1, 1])
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)

    # PLOT: metrics
    # metrics_per_class = Metrics.get_metrics_per_class(cf_matrix)[0]
    # Metrics.plot_metrics_heatmap(metrics_per_class)


def plot_ej_b(X_train, X_test, y_train, y_test, interval, seed):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=10000, max_epochs=1000,
                                  tol=0.01, random_state=seed, verbose=False)

    perceptron.fit(X_train, y_train)
    m = -perceptron.w_[0] / perceptron.w_[1]
    b = (-perceptron.b_ / perceptron.w_[1])[0]

    three_points_svm = ThreePointsSVM()
    three_points_svm.fit(X_train, y_train, [m, b])

    plt.scatter(three_points_svm.support_points_x, three_points_svm.support_points_y, c="violet", s=170)

    # find accuracy
    y_pred = three_points_svm.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print("test accuracy: ", accuracy)

    # TRAIN HYPERPLANE
    plot_data(X_train, y_train, interval, [three_points_svm.optimum_hyperplane_data, *three_points_svm.support_hyperplanes_data, [m, -1, b]],
              title="Optimal hyperplane", colors=['green', 'blue', 'blue', 'red'], 
              styles=['-', '--', '--', '-'], with_support_hyperplanes=True,
              labels=['Optimal hyperplane', 'Support hyperplane', 'Support hyperplane', 'Perceptron'])

    print("Optimal hyperplane confidence: ", three_points_svm.confidence_margin)
    print("Perceptron confidence: ", three_points_svm.perceptron_confidence)
    # TEST
    # plot_data(X_test, y_test, interval, [three_points_svm.optimum_hyperplane_data, *three_points_svm.support_hyperplanes_data, [m, -1, b]],
    #           title="Optimal hyperplane with test dataset", colors=['green', 'blue', 'blue', 'red'], 
    #           styles=['-', '--', '--', '-'], with_support_hyperplanes=True,
    #           labels=['Optimal hyperplane', 'Support hyperplane', 'Support hyperplane', 'Perceptron'])


def plot_ej_c(X, y, interval, seed):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=1000, max_epochs=1000,
                                  tol=0.01, random_state=seed)

    perceptron.fit(X, y)

    print("Error: ", perceptron.error_)
    print("W: ", perceptron.w_)
    print("B: ", perceptron.b_)

    fig, ax = plt.subplots(1, 1)

    # Plot the hyperplane animation
    animate = get_animation_function(perceptron, X, y, interval, ax, "Perceptron classification with not linearly separable dataset")
    anim = animation.FuncAnimation(fig, animate, frames=len(
        perceptron.w_list), repeat=False, interval=0)
    
    anim.save('./machine-learning/ej1/dump/not_linearly_dataset_perceptron_animation.mp4', fps=5)
    plt.show()


def plot_ej_d_sep_grid_search(X_train, X_test, y_train, y_test, m, b, interval, seed):
    # defining parameter range
    param_grid = {'c': [0.01, 0.1, 1, 5, 10, 100],
                  'max_iter': [1000, 10000],
                  'eta_w': [0.005, 0.01, 0.1, 1],
                  'eta_b': [0.005, 0.01, 0.1, 1],
                  'eta_decay_rate': [0.005, 0.01, 0.1, 1]}

    grid = GridSearchCV(SVM(c=1, max_iter=10000, random_state=seed, tol=0.01,
                            eta_w=0.01, eta_b=0.005, eta_decay_rate=0.005, verbose=False),
                        param_grid, refit=True, verbose=3, n_jobs=-1, cv=5)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


def plot_ej_d_sep(X, y, m, b, interval, seed):
    # Classify the points using the hinge loss SVM
    svm = SVM(c=5, max_iter=1000, random_state=seed, tol=0.01,
              eta_w=0.005, eta_b=1, eta_decay_rate=0.01, verbose=True)

    svm.fit(X, y)

    print("Error: ", svm.error_)
    print("W: ", svm.w_)
    print("B: ", svm.b_)

    # (w0, w1) * (x, y) + b = 0
    # w0 x + w1 y + b = 0
    # y = -w0/w1 x - b/w1

    plot_data(X, y, interval, [[*svm.w_, svm.b_], [m, -1, b]],
              title="SVM classification with linearly separable dataset", colors=['green', 'red'], labels=['Predicted', 'Real'])


def plot_ej_d_non_sep(X, y, interval, seed):
    # Classify the points using the hinge loss SVM
    svm = SVM(c=1, max_iter=10000, random_state=seed, tol=0.01,
              eta_w=0.01, eta_b=0.01, eta_decay_rate=0.1, verbose=True)

    svm.fit(X, y)

    print("Error: ", svm.error_)
    print("W: ", svm.w_)
    print("B: ", svm.b_)

    # (w0, w1) * (x, y) + b = 0
    # w0 x + w1 y + b = 0
    # y = -w0/w1 x - b/w1

    plot_data(X, y, interval, [[*svm.w_, svm.b_]],
              title="SVM classification with not linearly separable dataset", labels=['Predicted'])


def plot_data(X, y, interval, hyperplanes: list[list[float]], labels: list[str] = None, title: str = None,
              colors: list[str] = None, styles: list[str] = None, with_support_hyperplanes: bool = False):
    plt.scatter(X[:, 0], X[:, 1], c=y)

    for i, hyperplane in enumerate(hyperplanes):
        m_hat = -hyperplane[0] / hyperplane[1]
        b_hat = -hyperplane[2] / hyperplane[1]

        if with_support_hyperplanes and i == 1:
            plt.plot(interval, generate_line_interval(
                m_hat, b_hat, interval), color=colors[i], linestyle='--')
        else:
            plt.plot(interval, generate_line_interval(
                m_hat, b_hat, interval), color=colors[i] if colors else 'green', label=labels[i] if labels else None,
                linestyle=styles[i] if styles else '-')

    if labels:
        plt.legend()

    if title:
        plt.title(title)

    plt.ylim(top=interval[1])
    plt.ylim(bottom=interval[0])
    plt.xlim(right=interval[1])
    plt.xlim(left=interval[0])
    plt.show()


if __name__ == "__main__":
    seed = 4
    interval = [0, 5]
    n = 200

    X, y, m, b = generate_linearly_separable(
        n, interval, seed, clustering=True, hyperplane_margin=0.5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    # perceptron_grid_search_params = {
    #     "eta": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    # execute_grid_search_cv(SimplePerceptron(eta=0.001, max_iter=1000, max_epochs=1000,
    #                                         tol=0.01, random_state=seed, verbose=False), X_train, X_test,
    #                        y_train, y_test, perceptron_grid_search_params)

    plot_ej_a(X_train, y_train, X_test, y_test, m, b, interval, seed)
    # plot_ej_b(X, X_test, y, y_test, interval, seed)
    

    # noise_prox = 0.2
    # noise_prob = 0.5
    # non_sep_X, non_sep_y = generate_not_linearly_separable(
    #     n, interval, noise_proximity=noise_prox, noise_probability=noise_prob, seed=seed)

    # plot_ej_c(non_sep_X, non_sep_y, interval, seed)

    # plot_ej_d_sep(X, y, m, b, interval, seed)
    # plot_ej_d_sep_grid_search(X_train, X_test, y_train, y_test, m, b, interval, seed)
    # plot_ej_d_non_sep(non_sep_X, non_sep_y, interval, seed)

# seed 1, prox 0.1, prob 0.5 - parece no linealmente separable
# seed 2, prox 0.1, prob 0.5 idem, mejor que 1
# seed 2, prox 0.2, prob 0.5 idem, mejor que 1 y 2
