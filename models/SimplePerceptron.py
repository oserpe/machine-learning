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

        # By iteration
        self.w_list = np.array([self.w_])
        self.b_list = np.array(self.b_)
        
        # By epoch
        self.w_epoch_list = np.array([self.w_])
        self.b_epoch_list = np.array(self.b_)
        self.error_epoch_list = np.array(self.compute_error(X, y))

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
                self.w_epoch_list = np.append(self.w_epoch_list, [self.w_], axis=0)
                self.b_epoch_list = np.append(self.b_epoch_list, self.b_, axis=0)
                self.error_epoch_list = np.append(self.error_epoch_list, self.compute_error(X, y))
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
        self.epochs = epoch

        return self

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

    def compute_error(self, X, y):
        y_hat = self.predict(X)
        err = 0.5 * sum(np.square(y_hat - y))
        return err

    def score(self, X, y):
        return np.mean(self.predict(X) == y)