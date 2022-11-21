import math
import numpy as np
from sklearn.base import BaseEstimator

from .cluster import Cluster

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
        
        # create the empty clusters
        self.clusters = [Cluster([]) for i in range(self.K * self.K)]
        
        # initialize weights with random samples from X
        for i in range(self.K):
            for j in range(self.K):
                self.w[i][j] = X[random_state.randint(0, len(X))]
        
        iter = 0
        while iter < self.max_iter:
            radius = math.floor(self.radius_fn(iter, self.max_iter, self.initial_radius))
            X_shuffled = random_state.permutation(X)
            for x_i in X_shuffled:
                # find the winning neuron
                winner = self.find_closest_neuron(x_i)

                # update the weights of the winning neuron and its neighbors                
                for i in range(winner[0] - radius, winner[0] + radius + 1):
                    for j in range(winner[1] - radius, winner[1] + radius + 1):
                        if i >= 0 and i < self.K and j >= 0 and j < self.K:
                            self.w[i][j] += self.lr_fn(iter, self.max_iter, self.initial_lr) * (x_i - self.w[i][j])

                # add the sample into its corresponding cluster, only if it's the last iteration
                if iter == self.max_iter - 1:
                    pos = winner[0] * self.K + winner[1]
                    self.clusters[pos].add_points([x_i])
            
            iter += 1
            
        return

    def predict(self, X):
        # find the winning neuron for each sample and return its cluster index
        clusters_index = []

        for x in X:
            winner = self.find_closest_neuron(x)
            pos = winner[0] * self.K + winner[1]
            clusters_index.append(pos)

        return clusters_index


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
    
    def get_feature_weights(self, feature_index):
        if feature_index >= self.w.shape[2]:
            raise Exception("Feature index out of bounds")

        feature_weights = np.zeros((self.K, self.K))
        for x in range(self.K):
            for y in range(self.K):
                feature_weights[x][y] = self.w[x][y][feature_index]
        
        return feature_weights
    
    def clusters_to_matrix(self):
        clusters_matrix = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                clusters_matrix[i][j] = len(self.clusters[i * self.K + j])
        
        return clusters_matrix

    def get_clusters(self, iteration):
        return self.clusters