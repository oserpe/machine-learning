import math
import numpy as np
from sklearn.base import BaseEstimator
import copy
from .cluster import ClusteringDistance, Cluster


class HierarchicalClustering(BaseEstimator):
    def __init__(self, K, distance_metric: ClusteringDistance = None, verbose=True):
        self.verbose = verbose
        self.K = K
        self.distance_metric = distance_metric
        self.clusters_evolution = []
        self.distance_evolution = []
        self.variations = []

    def fit(self, X):
        # Initialize clusters for every point
        current_clusters = [Cluster([point]) for point in X]
        cluster_distances = np.zeros(
            (len(current_clusters), len(current_clusters)))

        for i in range(len(current_clusters)):
            for j in range(i+1, len(current_clusters)):
                cluster_distances[i][j] = current_clusters[i]\
                    .distance_to_cluster(current_clusters[j], self.distance_metric)

        while len(current_clusters) > self.K:
            if len(current_clusters) % 100 == 0:
                print(len(current_clusters))

            min_cluster_distance = math.inf
            cluster_point_1 = None
            cluster_point_2 = None

            for i in range(len(current_clusters)):
                for j in range(i+1, len(current_clusters)):
                    if cluster_distances[i][j] < min_cluster_distance:
                        min_cluster_distance = cluster_distances[i][j]
                        cluster_point_1 = i
                        cluster_point_2 = j

            # Merge clusters
            new_cluster = Cluster(
                current_clusters[cluster_point_1].points + current_clusters[cluster_point_2].points)

            # Remove merged clusters
            # first the second one because j is always > i when making the matrix
            current_clusters.pop(cluster_point_2)
            current_clusters.pop(cluster_point_1)

            current_clusters.insert(cluster_point_1, new_cluster)

            # add distances to new cluster
            cluster_distances = np.delete(
                cluster_distances, cluster_point_2, axis=0)  # remove row
            cluster_distances = np.delete(
                cluster_distances, cluster_point_2, axis=1)  # remove column

            # reuse the other row and column that weren't removed (cluster_point_1)
            for i in range(len(current_clusters)):
                cluster_distances[cluster_point_1][i] = new_cluster\
                    .distance_to_cluster(current_clusters[i], self.distance_metric)
                cluster_distances[i][cluster_point_1] = cluster_distances[cluster_point_1][i]

            self.clusters_evolution.append(copy.copy(current_clusters))
            self.distance_evolution.append(min_cluster_distance)

            # if len(current_clusters) <= 25: # TODO: FIX MAGIC NUMBER
            #     self.variations.append(self.compute_average_variation(current_clusters))

        self.clusters = copy.copy(current_clusters)

    def find_closest_cluster_to_point(self, point):
        point_cluster = Cluster([point])

        closest_cluster_index = np.argmin(
            [point_cluster.distance_to_cluster(cluster, self.distance_metric) for cluster in self.clusters])

        return closest_cluster_index

    def predict(self, X):
        # find the winning cluster for each sample and return its cluster index
        clusters_index = []

        for x in X:
            closest_centroid_index = self.find_closest_cluster_to_point(x)
            clusters_index.append(closest_centroid_index)

        return clusters_index

    def compute_average_variation(self, clusters):
        total_variation = 0
        for cluster in clusters:
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    total_variation += np.linalg.norm(cluster.points[i] - cluster.points[j], axis=0)

            if len(cluster) == 0:
                print("WARNING: empty cluster")

        return total_variation/len(clusters)
    
    def get_clusters(self):
        return self.clusters
