import numpy as np


def generate_linearly_separable(n, interval, seed=None):
    np.random.seed(seed)
    # Generate random samples
    X = np.random.uniform(interval[0], interval[1], (n, 2))

    # Generate random line
    m, b = np.random.uniform(interval[0], interval[1], 2)

    # Generate labels
    y = np.array([1 if x[1] > m * x[0] + b else -1 for x in X])

    return X, y, m, b


def generate_not_linearly_separable(n, interval, noise_proximity, noise_probability, seed=None):

    # Generate random samples
    X, y, m, b = generate_linearly_separable(n, interval, seed)

    # Add noise in points at distance noise_proximity wrt the line to invert labels
    for i in range(len(y)):
        if np.random.uniform() < noise_probability:
            if abs(y[i] * (m * X[i][0] - X[i][1] + b) / np.sqrt(m ** 2 + 1)) < noise_proximity:
                y[i] = -y[i]

    return X, y
