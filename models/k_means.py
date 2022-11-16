import math
import numpy as np
from sklearn.base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, K, max_iter, random_state, tol = 0.0001, verbose=True):
        self.K = K
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.prototypes = [] # can be centroids, modes, medians, etc.
        self.groups = [[] for _ in range(self.K)] # groups of data points

    def fit(self, X):
        iter = 0

        # Initialize by assigning samples to random groups
        random_state = np.random.RandomState(self.random_state)
        for x_i in X:
            self.groups[random_state.randint(0, self.K)].append(x_i) 

        self.variation = math.inf
        last_variation = 0
        while iter < self.max_iter and self.variation != last_variation:
            self.variation = last_variation

            # Find centroids
            self.prototypes = []
            for i in range(self.K):
                self.prototypes.append(np.mean(self.groups[i], axis=0))

            # Assign samples to their closest prototype's group
            self.groups = [[] for _ in range(self.K)]
            for x_i in X:
                # find index of closest prototype
                closest_prototype_index = np.argmin([np.linalg.norm(x_i - prototype) for prototype in self.prototypes])
                self.groups[closest_prototype_index].append(x_i)

            # Calculate sum of groups variation of W
            last_variation = 0 
            for group in self.groups:
                if len(group) > 0:
                    last_variation += self.calculate_W_vartiation(group)
            
            print(self.variation)
            
        return self.variation

    def calculate_W_vartiation(self, X):
        W = 0
        for i in range(len(X)):
                for j in range(i+1, len(X)):
                    W += np.linalg.norm(X[i] - X[j], axis=0)
                    # FIXME: esto es euclidean distance pero en el ppt pareceria ser que es euclidean distance al cuadrado
                    # print(W, np.linalg.norm(X[i] - X[j], axis=0), X[j])
                    # if j>10:
                    #     exit()
        
        return W/len(X)