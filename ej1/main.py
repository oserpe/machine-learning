import math
from turtle import left
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import itertools
from sklearn.metrics import classification_report, confusion_matrix 

from ..models.SVM import SVM
from ..models.ThreePointsSVM import ThreePointsSVM
from ..models.SimplePerceptron import SimplePerceptron
from .dataset.generate_dataset import generate_linearly_separable, generate_not_linearly_separable
from matplotlib import pyplot as plt


def generate_line_interval(m, b, interval):
    return [m * interval[0] + b, m * interval[1] + b]


def plot_ej_a(X, y, m, b, interval, seed):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=1000, max_epochs=1000,
                                  tol=0.01, random_state=seed, verbose=False)

    perceptron.fit(X, y)

    print("Error: ", perceptron.error_)
    print("W: ", perceptron.w_)
    print("B: ", perceptron.b_)

    # (w0, w1) * (x, y) + b = 0
    # w0 x + w1 y + b = 0
    # y = -w0/w1 x - b/w1
    plot_data(X, y, interval, [[*perceptron.w_, perceptron.b_], [m, -1, b]],
              title="Perceptron classification with linearly separable dataset", colors=['green', 'red'], labels=['Predicted', 'Real'])

def plot_ej_b(X_train, X_test, y_train, y_test, interval):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=1000, max_epochs=1000,
                                  tol=0.01, random_state=seed, verbose=False)

    perceptron.fit(X_train, y_train)
    m = -perceptron.w_[0] / perceptron.w_[1]
    b = (-perceptron.b_ / perceptron.w_[1])[0]

    three_points_svm = ThreePointsSVM()
    three_points_svm.fit(X_train, y_train, [m, b])

    plt.scatter(three_points_svm.support_points_x, three_points_svm.support_points_y, c="orange", s=90)

    # find accuracy
    y_pred = three_points_svm.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print("test accuracy: ", accuracy)

    # TRAIN
    plot_data(X_train, y_train, interval, [three_points_svm.optimum_hyperplane_data, *three_points_svm.support_hyperplanes_data, [m, -1, b]],
              title="New hyperplane", colors=['green', 'blue', 'blue', 'red'], 
              labels=['Optimum', 'Support line', 'Support line', 'Perceptron'])

    # TEST
    plot_data(X_test, y_test, interval, [three_points_svm.optimum_hyperplane_data, *three_points_svm.support_hyperplanes_data, [m, -1, b]],
              title="New hyperplane", colors=['green', 'blue', 'blue', 'red'], 
              labels=['Optimum', 'Support line', 'Support line', 'Perceptron'])



def plot_ej_c(X, y, interval, seed):
    # Classify the points using the perceptron
    perceptron = SimplePerceptron(eta=0.01, max_iter=1000, max_epochs=1000,
                                  tol=0.01, random_state=seed)

    perceptron.fit(X, y)

    print("Error: ", perceptron.error_)
    print("W: ", perceptron.w_)
    print("B: ", perceptron.b_)

    # (w0, w1) * (x, y) + b = 0
    # w0 x + w1 y + b = 0
    # y = -w0/w1 x - b/w1

    plot_data(X, y, interval, [[*perceptron.w_, perceptron.b_]],
              title="Perceptron classification with not linearly separable dataset", labels=['Predicted'])

# TODO: agregar animación de recta por epoca/iteración

def plot_ej_d_sep_grid_search(X_train, X_test, y_train, y_test, m, b, interval, seed):
    # defining parameter range 
    param_grid = {'c': [0.01, 0.1, 1, 5, 10, 100],
                    'max_iter': [1000, 10000],
                    'eta_w': [0.005, 0.01, 0.1, 1],
                    'eta_b': [0.005, 0.01, 0.1, 1],
                    'eta_decay_rate': [0.005, 0.01, 0.1, 1]}  
    
    grid = GridSearchCV(SVM(c=1, max_iter=10000, random_state=seed, tol=0.01,
              eta_w=0.01, eta_b=0.005, eta_decay_rate=0.005, verbose=False), 
              param_grid, refit = True, verbose = 3, n_jobs=-1, cv=5) 
    
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


def plot_data(X, y, interval, hyperplanes: list[list[float]], labels: list[str] = None, title: str = None, colors: list[str] = None):
    plt.scatter(X[:, 0], X[:, 1], c=y)

    for i, hyperplane in enumerate(hyperplanes):
        m_hat = -hyperplane[0] / hyperplane[1]
        b_hat = -hyperplane[2] / hyperplane[1]

        plt.plot(interval, generate_line_interval(
            m_hat, b_hat, interval), color=colors[i] if colors else 'green', label=labels[i] if labels else None)

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
    seed = 2
    interval = [0, 5]
    n = 200

    X, y, m, b = generate_linearly_separable(n, interval, seed, clustering=True, hyperplane_margin=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    # plot_ej_a(X, y, m, b, interval, seed)
    # plot_ej_b(X_train, X_test, y_train, y_test, interval)
    

    noise_prox = 0.2
    noise_prob = 0.5
    non_sep_X, non_sep_y = generate_not_linearly_separable(
        n, interval, noise_proximity=noise_prox, noise_probability=noise_prob, seed=seed)

    # plot_ej_c(non_sep_X, non_sep_y, interval, seed)

    plot_ej_d_sep(X, y, m, b, interval, seed)
    # plot_ej_d_sep_grid_search(X_train, X_test, y_train, y_test, m, b, interval, seed)
    # plot_ej_d_non_sep(non_sep_X, non_sep_y, interval, seed)

# seed 1, prox 0.1, prob 0.5 - parece no linealmente separable
# seed 2, prox 0.1, prob 0.5 idem, mejor que 1
# seed 2, prox 0.2, prob 0.5 idem, mejor que 1 y 2
