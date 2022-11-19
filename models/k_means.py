import math
import numpy as np
from sklearn.base import BaseEstimator

class DotWithID:
    # Class to store a dot with its id, so we know which row of the distances matrix corresponds to this dot
    def __init__(self, id, dot):
        self.id = id
        self.dot = dot

class KMeans(BaseEstimator):
    def __init__(self, K, max_iter, random_state, tol = 0.0001, verbose=True):
        self.K = K
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.centroids = []
        self.groups = [[] for _ in range(self.K)]

    def fit(self, X):

        # Precompute points distances for W variation calculation
        points_distances = np.zeros((len(X), len(X)))
        dots_with_id = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                    # FIXME: esto es euclidean distance pero en el ppt pareceria ser que es euclidean distance al cuadrado
                points_distances[i][j] = np.linalg.norm(X[i] - X[j], axis=0)
                points_distances[j][i] = points_distances[i][j] # symmetric matrix
            
            # create a DotWithID object for each dot, to remeber which row of the distances matrix corresponds to this dot
            dots_with_id.append( DotWithID(i, X[i]) ) 


        # Initialize kmeans by assigning samples to random groups
        random_state = np.random.RandomState(self.random_state)
        for dot_with_id in dots_with_id:
            self.groups[random_state.randint(0, self.K)].append(dot_with_id) 

        iter = 0
        self.variation = math.inf
        last_variation = 0
        while iter < self.max_iter and self.variation != last_variation:
            iter +=1
            self.variation = last_variation

            # Find centroids
            self.centroids = [0 for _ in range(self.K)]
            for i in range(self.K):
                self.centroids[i] = np.mean(
                    list(map(lambda dot_data: dot_data.dot , self.groups[i]))
                    , axis=0)

            # Assign samples to their closest prototype's group
            self.groups = [[] for _ in range(self.K)]
            for dot_with_id in dots_with_id:
                # find index of closest prototype
                closest_prototype_index = np.argmin([np.linalg.norm(dot_with_id.dot - prototype) for prototype in self.centroids])
                self.groups[closest_prototype_index].append(dot_with_id)

            # Calculate sum of groups variation of W
            last_variation = 0 
            for group in self.groups:
                if len(group) > 0:
                    last_variation += self.calculate_W_variation(points_distances, group)
                else: print("WARNING: empty group")
            

        # FIXME: should we return the best variation or the last variation?            
        return self.variation

    def calculate_W_variation(self, points_distances, dots_with_id):
        W = 0
        for i in range(len(dots_with_id)):
            for j in range(i+1, len(dots_with_id)):
                W += points_distances[dots_with_id[i].id][dots_with_id[j].id]
        
        return W/len(dots_with_id)
