import math
import numpy as np
from sklearn.base import BaseEstimator


class SVM(BaseEstimator):
    def __init__(self, c, max_iter, random_state, tol, eta_w, eta_b, eta_decay_rate, verbose=True):
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.eta_w = eta_w
        self.eta_b = eta_b
        self.eta_decay_rate = eta_decay_rate
        self.verbose = verbose

    def fit(self, X, y):
        iter = 0
        error = math.inf

        # Initialize weights and bias
        random_state = np.random.RandomState(self.random_state)
        self.w_ = random_state.uniform(-1, 1, X.shape[1])
        self.b_ = random_state.uniform(-1, 1)

        # Initialize learning rates (they are updated in each iteration)
        eta_w_t = self.eta_w
        eta_b_t = self.eta_b

        best_w = self.w_
        best_b = self.b_
        best_error = error

        while iter < self.max_iter and error > self.tol:
            random_sample = random_state.randint(0, X.shape[0])
            x_i = X[random_sample]
            y_i = y[random_sample]

            t = y_i * (np.dot(self.w_, x_i) + self.b_)

            if t >= 1:
                self.w_ = (1 - eta_w_t) * self.w_
            else:
                dw, db = self.compute_error_derivative(x_i, y_i)
                self.w_ -= eta_w_t * dw
                self.b_ -= eta_b_t * db

            # update learning rates
            eta_w_t = self.compute_learning_rate(iter, self.eta_w)
            eta_b_t = self.compute_learning_rate(iter, self.eta_b)

            error = self.compute_error(X, y)

            if error < best_error:
                best_w = self.w_
                best_b = self.b_
                best_error = error

            if self.verbose:
                print("Iter: ", iter, "Error: ", error,
                      "Best error: ", best_error)

            # Update iteration
            iter += 1

        self.w_ = best_w
        self.b_ = best_b
        self.error_ = best_error

        return self

    def compute_error_derivative(self, X, y):
        return self.w_ - self.c * np.sum(np.dot(X, y), axis=0), self.eta_b * self.c * (-np.sum(y, axis=0))

    def compute_error(self, X, y):
        # Compute hinge loss (average)
        t = y * (X @ self.w_ + self.b_)
        return np.sum(np.maximum(0, 1 - t)) / X.shape[0] + 0.5 * np.sum(self.w_ ** 2)

    def compute_learning_rate(self, t, eta):
        # TODO: Check for underflow?
        return eta * math.exp(-self.eta_decay_rate * t)

    def predict(self, X):
        return np.sign(X @ self.w_ + self.b_)
