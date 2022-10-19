import math
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class SimplePerceptron(BaseEstimator):
    def __init__(self, eta, max_iter, max_epochs, tol, random_state, verbose=True):
        self.eta = eta
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.random_state = check_random_state(self.random_state)
        self.w_ = self.random_state.uniform(size=X.shape[1], low=-1, high=1)
        self.b_ = self.random_state.uniform(size=1, low=-1, high=1)

        epoch = 0
        iter = 0
        error = math.inf
        best_w = self.w_
        best_b = self.b_
        best_error = math.inf

        self.w_list = np.array([self.w_])
        self.b_list = np.array(self.b_)

        while epoch < self.max_epochs and iter < self.max_iter and error > self.tol:
            random_sample = self.random_state.randint(0, X.shape[0])
            x = X[random_sample]

            y_hat = self.predict([x])[0]

            update = self.eta * (y[random_sample] - y_hat)
            self.w_ += update * x
            self.b_ += update
            self.w_list = np.append(self.w_list, [self.w_], axis=0)
            self.b_list = np.append(self.b_list, self.b_, axis=0)

            iter += 1
            if iter % X.shape[0] == 0:
                epoch += 1

            error = self.compute_error(X, y)

            if error < best_error:
                best_error = error
                best_w = self.w_
                best_b = self.b_

            if self.verbose:
                print("Epoch: ", epoch, "Iter: ", iter, "Error: ",
                      error, "Best Error: ", best_error)

        self.w_ = best_w
        self.b_ = best_b
        self.error_ = best_error

        return self

    def predict(self, X):
        h = X @ self.w_ + self.b_
        y_hat = [1 if x >= 0 else -1 for x in h]
        
        return y_hat

    def compute_error(self, X, y):
        y_hat = self.predict(X)
        err = 0.5 * sum(np.square(y_hat - y))

        return err

    def score(self, X, y):
        return np.mean(self.predict(X) == y)