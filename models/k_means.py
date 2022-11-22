import math
import numpy as np
from .cluster import Cluster
from sklearn.base import BaseEstimator


class Point:
    def __init__(self, point, id):
        self.id = id
        self.point = point


class KMeans(BaseEstimator):
    def __init__(self, K, max_iter, random_state, verbose=True):
        self.K = K
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.clusters = [[] for _ in range(self.K)]
        self.variations = []

    def compute_X_distances(self, X):
        X_distances_matrix = np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(i+1, len(X)):
                X_distances_matrix[i][j] = np.linalg.norm(X[i] - X[j], axis=0)
                # symmetric matrix
                X_distances_matrix[j][i] = X_distances_matrix[i][j]

        return X_distances_matrix

    def choose_initial_clusters(self, X):
        random_state = np.random.RandomState(self.random_state)
        clusters = [[] for _ in range(self.K)]
        for x in X:
            clusters[random_state.randint(0, self.K)].append(x)

        return clusters

    def compute_centroids(self, clusters):
        return [np.mean(cluster, axis=0) for cluster in self.map_clusters_to_numpy(clusters)]

    def find_closest_centroid(self, x, centroids):
        # returns the index of the closest centroid to x
        return np.argmin([np.linalg.norm(x - centroid) for centroid in centroids])

    def map_clusters_to_numpy(self, clusters):
        return [[x.point for x in cluster] for cluster in clusters]

    def fit(self, X):
        # Precompute points distances for computing variation within clusters
        X_distances_matrix = self.compute_X_distances(X)
        points = [Point(x, i) for i, x in enumerate(X)]

        # Initialize kmeans by assigning samples to random groups
        clusters = self.choose_initial_clusters(points)
        centroids = self.compute_centroids(clusters)

        iter = 0
        prev_centroids = None

        while iter < self.max_iter and np.not_equal(centroids, prev_centroids).any():
            iter += 1

            # Assign samples to their closest centroid
            clusters = [[] for _ in range(self.K)]
            for x in points:
                # find index of closest prototype
                closest_centroid_index = self.find_closest_centroid(
                    x.point, centroids)
                clusters[closest_centroid_index].append(x)

            # Find new centroids
            centroids = self.compute_centroids(clusters)

            # Calculate sum of groups variation of W
            self.variations.append(self.compute_average_variation(
                X_distances_matrix, clusters)
            )

        for i, cluster in enumerate(clusters):
            self.clusters[i] = Cluster([x.point for x in cluster])

    def compute_average_variation(self, X_distances_matrix, clusters):
        total_variation = 0
        for cluster in clusters:
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    total_variation += X_distances_matrix[cluster[i].id][cluster[j].id]

            if len(cluster) == 0:
                print("WARNING: empty cluster")

        return total_variation/len(clusters)

    def predict(self, X):
        # find the winning cluster for each sample and return its cluster index
        clusters_index = []

        for x in X:
            closest_centroid_index = self.find_closest_centroid(
                x, [cluster.centroid for cluster in self.clusters])
            clusters_index.append(closest_centroid_index)

        return clusters_index

    def get_clusters(self):
        return self.clusters
