import numpy as np
from sklearn.model_selection import GridSearchCV
from .dataset.generate_dataset import generate_linearly_separable, generate_not_linearly_separable
from matplotlib import pyplot as plt
from ..models.SimplePerceptron import SimplePerceptron


def generate_line_interval(m, b, interval):
    return [m * interval[0] + b, m * interval[1] + b]


def plot_ej_a():
    seed = 2

    interval = [-1, 1]
    n = 100
    noise_prox = 0.2
    noise_prob = 0.5

    X, y, m, b = generate_linearly_separable(n, interval, seed)

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

    m_hat = -perceptron.w_[0] / perceptron.w_[1]
    b_hat = -perceptron.b_ / perceptron.w_[1]

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(interval, generate_line_interval(m, b, interval), color='red')
    plt.plot(interval, generate_line_interval(
        m_hat, b_hat, interval), color='green')

    plt.show()


if __name__ == "__main__":
    seed = 2

    interval = [-1, 1]
    n = 100
    noise_prox = 0.2
    noise_prob = 0.5

    X_non_separable, y_non_separable = generate_not_linearly_separable(
        n, interval, noise_proximity=noise_prox, noise_probability=noise_prob, seed=seed)

    plot_ej_a()

# seed 1, prox 0.1, prob 0.5 - parece no linealmente separable
# seed 2, prox 0.1, prob 0.5 idem, mejor que 1
# seed 2, prox 0.2, prob 0.5 idem, mejor que 1 y 2
