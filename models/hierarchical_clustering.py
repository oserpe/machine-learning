import math
import numpy as np
from sklearn.base import BaseEstimator
import copy

class ClusteringDistance():
    CENTROID = 0
    MINIMUM = 1
    MAXIMUM = 2
    AVERAGE = 3

class Cluster:
    def __init__(self, points):
        self.points = points
        self.centroid = np.mean(points, axis=0)

    def __str__(self) -> str:
        return f'Cluster with {len(self.points)} points and centroid {self.centroid}'
    
    def distances_to_cluster_points(self, cluster):
        distances = []
        for point in self.points:
            for point2 in cluster.points:
                distances.append(np.linalg.norm(point - point2))
        return distances

    def distance_to_cluster(self, cluster, distance_type: ClusteringDistance):
        if distance_type == ClusteringDistance.CENTROID:
            return self.centroid_distance(cluster)
        elif distance_type == ClusteringDistance.MINIMUM:
            return self.min_distance(cluster)
        elif distance_type == ClusteringDistance.MAXIMUM:
            return self.max_distance(cluster)
        elif distance_type == ClusteringDistance.AVERAGE:
            return self.mean_distance(cluster)
        else:
            return self.centroid_distance(cluster)

    def max_distance(self, cluster):
        return max(self.distances_to_cluster_points(cluster))

    def min_distance(self, cluster):
        return min(self.distances_to_cluster_points(cluster))
    
    def mean_distance(self, cluster):
        return np.mean(self.distances_to_cluster_points(cluster))
    
    def centroid_distance(self, cluster):
        return np.linalg.norm(self.centroid - cluster.centroid)

class HierarchichalClustering(BaseEstimator):
    def __init__(self, distance_metric: ClusteringDistance = None, verbose=True):
        self.verbose = verbose
        self.distance_metric = distance_metric
        self.clusters_evolution = []
        self.distance_evolution = []

    def fit(self, X):
        # Initialize clusters for every point
        current_clusters = [Cluster([point]) for point in X]
        
        while len(current_clusters) > 1:
            cluster_distances = np.zeros((len(current_clusters), len(current_clusters)))
            min_cluster_distance = math.inf
            min_cluster_distance_index = (0, 0)
            for i in range(len(current_clusters)):
                for j in range(i+1, len(current_clusters)):
                    cluster_distances[i][j] = current_clusters[i]\
                            .distance_to_cluster(current_clusters[j], self.distance_metric)

                    if cluster_distances[i][j] < min_cluster_distance:
                        min_cluster_distance = cluster_distances[i][j]
                        min_cluster_distance_index = (i, j)

            # Merge clusters
            new_cluster = Cluster(current_clusters[min_cluster_distance_index[0]].points + current_clusters[min_cluster_distance_index[1]].points)

            # Remove merged clusters
            current_clusters.pop(min_cluster_distance_index[1]) # first the second one because j is always > i when making the matrix
            current_clusters.pop(min_cluster_distance_index[0])

            current_clusters.append(new_cluster)

            self.clusters_evolution.append(copy.copy(current_clusters))
            self.distance_evolution.append(min_cluster_distance)
        
        


