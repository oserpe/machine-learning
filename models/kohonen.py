import math
import numpy as np
from sklearn.base import BaseEstimator

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def exp_decay(t, t_max, initial_value):
    return initial_value * math.exp(-t / t_max)

def linear_decay(t, initial_value):
    return initial_value / (t + 1)

class Kohonen(BaseEstimator):
    def __init__(self, max_iter, random_state, initial_radius, initial_lr, K, 
            distance_fn = euclidean_distance, lr_fn = exp_decay, radius_fn = exp_decay, verbose=True):
        self.verbose = verbose
        self.K = K
        self.random_state = random_state
        self.max_iter = max_iter
        self.initial_lr = initial_lr
        self.initial_radius = initial_radius
        self.distance_fn = distance_fn
        self.lr_fn = lr_fn
        self.radius_fn = radius_fn
        
    def find_closest_neuron(self, x):
        min_distance = math.inf
        closest_neuron = None

        for i in range(self.K):
            for j in range(self.K):
                distance = self.distance_fn(x, self.w[i][j])
                if distance < min_distance:
                    min_distance = distance
                    closest_neuron = (i, j)

        if closest_neuron is None:
            raise Exception("No closest neuron found")

        return closest_neuron

    def fit(self, X):
        random_state = np.random.RandomState(self.random_state)
        self.w = np.zeros((self.K, self.K, X.shape[1]))
        
        # initialize weights with random samples from X
        for i in range(self.K):
            for j in range(self.K):
                self.w[i][j] = X[random_state.randint(0, len(X))]
        
        iter = 0
        while iter < self.max_iter:
            radius = math.floor(self.radius_fn(iter, self.max_iter, self.initial_radius))
            for i in range(len(X)):
                x_i = X[random_state.randint(0, len(X))]

                # find the winning neuron
                winner = self.find_closest_neuron(x_i)

                # update the weights of the winning neuron and its neighbors                
                for i in range(winner[0] - radius, winner[0] + radius + 1):
                    for j in range(winner[1] - radius, winner[1] + radius + 1):
                        if i >= 0 and i < self.K and j >= 0 and j < self.K:
                            self.w[i][j] += self.lr_fn(iter, self.max_iter, self.initial_lr) * (x_i - self.w[i][j])

                
            iter += 1
            
        return

    def predict(self, X):
        # find the winning weights for each sample
        y = []
        
        for x in X:
            winner = self.find_closest_neuron(x)
            y.append(self.w[winner[0]][winner[1]])

        return y


    def get_mean_neighborhood_distance(self, x, y):
        radius = 1
        distances = []
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if i >= 0 and i < self.K and j >= 0 and j < self.K and i != x and j != y:
                    distances.append(np.linalg.norm(self.w[x][y] - self.w[i][j]))

        return np.mean(distances)

    def get_u_matrix(self):
        u_matrix = np.zeros((self.K, self.K))
        # for each neuron compute the distance to its neighbors
        for x in range(self.K):
            for y in range(self.K):
                u_matrix[x][y] = self.get_mean_neighborhood_distance(x, y)

        return u_matrix