import numpy as np


class ClusteringDistance():
    CENTROID = 0
    MINIMUM = 1
    MAXIMUM = 2
    AVERAGE = 3


class Cluster:
    def __init__(self, points):
        self.points = points
        self.centroid = np.mean(points, axis=0) if len(points) > 0 else None

    def __str__(self) -> str:
        return f'Cluster with {len(self.points)} points and centroid {self.centroid}'

    def add_points(self, points):
        self.points.extend(points)
        self.centroid = np.mean(self.points, axis=0) if len(self.points) > 0 else None

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

    def __len__(self):
        return len(self.points)