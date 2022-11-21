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

    def add_points(self, points):
        self.points.extend(points)
        self.centroid = np.mean(self.points, axis=0)

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


class HierarchicalClustering(BaseEstimator):
    def __init__(self, distance_metric: ClusteringDistance = None, verbose=True):
        self.verbose = verbose
        self.distance_metric = distance_metric
        self.clusters_evolution = []
        self.distance_evolution = []

    def fit(self, X):
        # Initialize clusters for every point
        current_clusters = [Cluster([point]) for point in X]
        cluster_distances = np.zeros(
            (len(current_clusters), len(current_clusters)))
        for i in range(len(current_clusters)):
            for j in range(i+1, len(current_clusters)):
                cluster_distances[i][j] = current_clusters[i]\
                    .distance_to_cluster(current_clusters[j], self.distance_metric)
                    
        while len(current_clusters) > 1:
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

        return

