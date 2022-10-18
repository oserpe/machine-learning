import math
from matplotlib import pyplot as plt
import numpy as np


def generate_linearly_separable(n, interval, seed=None, clustering=False, hyperplane_margin=0.2, required_distance_percentage=0.2, interval_compression_rate=0.9, n_compression_rate=0.8):
    np.random.seed(seed)

    # Generate random line
    m = np.random.uniform(interval[0], interval[1])
    b = np.random.uniform(interval[0], interval[1])

    # If the line is not near the middle, re randomize b and m
    middle = interval[1] / 2
    middle_points = [[interval[0], middle], [interval[1], middle], [
        middle, interval[0]], [middle, interval[1]]]
    done = False
    minimum_distance = ((interval[0] + interval[1]) / 2) * \
        (1 + required_distance_percentage)

    while not done:
        # Calculate line extreme points
        lower_point = np.array([interval[0], m * interval[0] + b])
        upper_point = np.array([interval[1], m * interval[1] + b])
        # Get line middle point
        line_middle_point = (lower_point + upper_point) / 2

        # Calculate the distance to the middle square points
        distances = [np.linalg.norm(middle_pt - line_middle_point)
                     for middle_pt in middle_points]

        # Check if distances fall inside the specified square
        if all([distance < minimum_distance for distance in distances]):
            done = True
        else:
            # Generate random line
            m = np.random.uniform(interval[0], interval[1])
            b = np.random.uniform(interval[0], interval[1])

    # Generate random samples
    # If clustering is True, the samples are generated in two clusters compressing the interval
    if clustering:
        upper_bound = interval[1] * interval_compression_rate
        lower_bound = interval[0] * \
            interval_compression_rate if interval[0] > 1 else 1 * interval_compression_rate
        n_clustered = int(n * n_compression_rate)
        X = np.random.uniform(
            lower_bound, upper_bound, (n_clustered, 2))
        X = np.concatenate((X, np.random.uniform(
            interval[0], interval[1], (n - n_clustered, 2))))
    else:
        X = np.random.uniform(interval[0], interval[1], (n, 2))

    # Generate labels
    y = np.array([1 if x[1] > m * x[0] + b else -1 for x in X])

    if clustering:
        # Remove the points near the hyperplane
        X_new = []
        y_new = []
        for point, feature in zip(X, y):
            distance_to_line = abs(
                feature * (m * point[0] - point[1] + b) / np.sqrt(m ** 2 + 1))

            if distance_to_line > hyperplane_margin:
                X_new.append(point)
                y_new.append(feature)

        X = np.array(X_new)
        y = np.array(y_new)

    return X, y, m, b


def generate_not_linearly_separable(n, interval, noise_proximity, noise_probability, seed=None, clustering=True, hyperplane_margin=0.2, required_distance_percentage=0.2):

    # Generate random samples
    X, y, m, b = generate_linearly_separable(
        n, interval, seed, clustering, hyperplane_margin, required_distance_percentage)

    # Add noise in points at distance noise_proximity wrt the line to invert labels
    for i in range(len(y)):
        if np.random.uniform() < noise_probability:
            if abs(y[i] * (m * X[i][0] - X[i][1] + b) / np.sqrt(m ** 2 + 1)) < noise_proximity:
                y[i] = -y[i]

    return X, y
