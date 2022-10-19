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
        epoch = 0
        error = math.inf

        # Initialize weights and bias
        random_state = np.random.RandomState(self.random_state)
        self.w_ = random_state.uniform(-1, 1, X.shape[1])
        self.b_ = random_state.uniform(-1, 1, size=1)

        # By iteration
        self.w_list = np.array([self.w_])
        self.b_list = np.array(self.b_)

        # By epoch
        self.w_epoch_list = np.array([self.w_])
        self.b_epoch_list = np.array(self.b_)
        self.error_epoch_list = np.array(self.compute_error(X, y))

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

            self.w_list = np.append(self.w_list, [self.w_], axis=0)
            self.b_list = np.append(self.b_list, self.b_, axis=0)

            # Update iteration
            iter += 1
            if iter % X.shape[0] == 0:
                self.w_epoch_list = np.append(
                    self.w_epoch_list, [self.w_], axis=0)
                self.b_epoch_list = np.append(
                    self.b_epoch_list, self.b_, axis=0)
                self.error_epoch_list = np.append(
                    self.error_epoch_list, self.compute_error(X, y))
                epoch += 1

        self.w_ = best_w
        self.b_ = best_b
        self.error_ = best_error
        self.epochs = epoch

        return self

    def compute_error_derivative(self, X, y):
        dw = self.w_ - self.c * np.dot(X, y)
        db = self.c * (-np.sum(y, axis=0))
        return dw, db

    def compute_error(self, X, y):
        # Compute hinge loss
        err = 0

        for i in range(X.shape[0]):
            x_i = X[i]
            y_i = y[i]

            t = y_i * (np.dot(self.w_, x_i) + self.b_)

            if t < 1:
                err += 1 - t

        return self.c * err + 0.5 * np.sum(self.w_ ** 2)

    def compute_learning_rate(self, t, eta):
        # TODO: Check for underflow?
        return eta * math.exp(-self.eta_decay_rate * t)

    def predict(self, X, w=None, b=None):
        if w is None:
            w = self.w_

        if b is None:
            b = self.b_

        h = X @ w + b
        y_hat = [1 if x >= 0 else -1 for x in h]

        return y_hat

    def predict_by_epoch(self, X):
        # For every epoch, calculate all the predictions for each sample in X
        y_hat_by_epoch = []
        for i in range(self.epochs + 1):
            y_hat = self.predict(X, self.w_epoch_list[i], self.b_epoch_list[i])
            y_hat_by_epoch.append(y_hat)

        return np.array(y_hat_by_epoch)

    def compute_error_by_epoch(self, X, y):
        y_hat = self.predict_by_epoch(X)
        err = 0.5 * np.sum(np.square(y_hat - y), axis=1)

        return err

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
