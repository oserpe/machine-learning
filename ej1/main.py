import math
from turtle import left
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import itertools
from ..models.SVM import SVM
from .dataset.generate_dataset import generate_linearly_separable, generate_not_linearly_separable
from matplotlib import pyplot as plt
from ..models.SimplePerceptron import SimplePerceptron


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

    # Find distances of points of each class to the hyperplane of the perceptron
    points_distance_positive_class = []
    points_distance_negative_class = []

    m = -perceptron.w_[0] / perceptron.w_[1]
    b = (-perceptron.b_ / perceptron.w_[1])[0]
    for x_i in X_train:
        y_i_predicted = perceptron.predict([x_i])[0]
        distance = abs((m * x_i[0] - x_i[1] + b) / np.sqrt(m ** 2 + 1))
        
        point_with_distance = (distance, x_i)
        if y_i_predicted == 1:
            points_distance_positive_class.append(point_with_distance)
        else:
            points_distance_negative_class.append(point_with_distance)

    # sort distances arrays
    points_distance_negative_class.sort(key=lambda x: x[0])
    points_distance_positive_class.sort(key=lambda x: x[0])


    # Find the hyperplane with the biggest margin
    max_confidence_margin = 0
    best_hyperplane_data = {
        'support_points_x': [],
        'support_points_y': [],
        'optimum_hyperplane_data': [],
        'support_hyperplanes_data': [],
    } 

    # Iterate over all possible combinations of support points
    for two_points_class_val in [-1, 1]:
        if two_points_class_val < 0:
            two_points_class = points_distance_negative_class
            one_point_class = points_distance_positive_class
        else:
            two_points_class = points_distance_positive_class
            one_point_class = points_distance_negative_class

        # Get the two closest points of the two_points_class and calculate first support hyperplane
        line_support_point1, line_support_point2 = list(map(lambda x: x[1], two_points_class[:2]))
        new_m = (line_support_point1[1] - line_support_point2[1]) / (line_support_point1[0] - line_support_point2[0])
        new_b = line_support_point1[1] - new_m * line_support_point1[0]

        first_support_hyp = [new_m, -1, new_b]
        
        # Get the closest point of the other class and calculate second support hyperplane using the previous one
        other_support_point = one_point_class[0][1]
        other_support_hyp = [new_m, -1, other_support_point[1] - new_m * other_support_point[0]]
        other_hyp_b = other_support_point[1] - new_m * other_support_point[0]
        # Find the optimum hyperplane using previous support hyperplanes
        optimum_hyperplane = [new_m, -1, (new_b + other_hyp_b) / 2]

        # Find if this hyperplane margin is better than the previous one, if so, save its data
        new_confidence_margin = abs(new_b - other_hyp_b) / 2
        if new_confidence_margin > max_confidence_margin:
            max_confidence_margin = new_confidence_margin
            best_hyperplane_data = {
                'support_points_x': [line_support_point1[0], line_support_point2[0], other_support_point[0]],
                'support_points_y': [line_support_point1[1], line_support_point2[1], other_support_point[1]],
                'optimum_hyperplane_data': optimum_hyperplane,
                'support_hyperplanes_data': [first_support_hyp, other_support_hyp],
            }


    plt.scatter(best_hyperplane_data['support_points_x'], best_hyperplane_data['support_points_y'], c="orange", s=90)
    plot_data(X_train, y_train, interval, [best_hyperplane_data['optimum_hyperplane_data'], *best_hyperplane_data['support_hyperplanes_data'], [m, -1, b]],
              title="New hyperplane", colors=['green', 'blue', 'blue', 'red'], 
              labels=['Optimum', 'Support line', 'Support line', 'Perceptron'])


    plot_data(X_test, y_test, interval, [best_hyperplane_data['optimum_hyperplane_data'], *best_hyperplane_data['support_hyperplanes_data'], [m, -1, b]],
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


def plot_ej_d_sep(X, y, m, b, interval, seed):
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
    plot_ej_b(X_train, X_test, y_train, y_test, interval)
    

    # noise_prox = 0.2
    # noise_prob = 0.5
    # non_sep_X, non_sep_y = generate_not_linearly_separable(
    #     n, interval, noise_proximity=noise_prox, noise_probability=noise_prob, seed=seed)

    # plot_ej_c(non_sep_X, non_sep_y, interval, seed)

    # plot_ej_d_sep(X, y, m, b, interval, seed)
    # plot_ej_d_non_sep(non_sep_X, non_sep_y, interval, seed)

# seed 1, prox 0.1, prob 0.5 - parece no linealmente separable
# seed 2, prox 0.1, prob 0.5 idem, mejor que 1
# seed 2, prox 0.2, prob 0.5 idem, mejor que 1 y 2
