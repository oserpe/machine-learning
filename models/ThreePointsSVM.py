import math
import numpy as np
from sklearn.base import BaseEstimator


class ThreePointsSVM(BaseEstimator):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, X, y, initial_hyperplane):
        # Find distances of points of each class to the hyperplane of the perceptron
        points_distance_positive_class = []
        points_distance_negative_class = []

        m = initial_hyperplane[0]
        b = initial_hyperplane[1]

        for x_i, y_i in zip(X,y):
            distance = abs((m * x_i[0] - x_i[1] + b) / np.sqrt(m ** 2 + 1))
            
            point_with_distance = (distance, x_i)
            if y_i == 1:
                points_distance_positive_class.append(point_with_distance)
            else:
                points_distance_negative_class.append(point_with_distance)

        # sort distances arrays
        points_distance_negative_class.sort(key=lambda x: x[0])
        points_distance_positive_class.sort(key=lambda x: x[0])

        # Save the perceptron confidence for study purposes
        self.perceptron_confidence = min(points_distance_positive_class[0][0], points_distance_negative_class[0][0])
        
        # Find the hyperplane with the biggest margin
        self.confidence_margin = 0

        # Iterate over all possible combinations of support points
        for two_points_class_val in [-1, 1]:
            if two_points_class_val < 0:
                two_points_class = points_distance_negative_class
                one_point_class = points_distance_positive_class
            else:
                two_points_class = points_distance_positive_class
                one_point_class = points_distance_negative_class

            # Get the two closest points of the two_points_class and calculate first support hyperplane
            line_support_point1, line_support_point2 = list(map(lambda x: x[1], two_points_class[:2]))
            new_m = (line_support_point1[1] - line_support_point2[1]) / (line_support_point1[0] - line_support_point2[0])
            new_b = line_support_point1[1] - new_m * line_support_point1[0]

            first_support_hyp = [new_m, -1, new_b]
            
            # Get the closest point of the other class and calculate second support hyperplane using the previous one
            other_support_point = one_point_class[0][1]
            other_support_hyp = [new_m, -1, other_support_point[1] - new_m * other_support_point[0]]
            other_hyp_b = other_support_point[1] - new_m * other_support_point[0]
            # Find the optimum hyperplane using previous support hyperplanes
            optimum_hyperplane = [new_m, -1, (new_b + other_hyp_b) / 2]

            # Find if this hyperplane margin is better than the previous one, if so, save its data
            new_confidence_margin = abs(new_b - other_hyp_b) / 2
            if new_confidence_margin > self.confidence_margin:
                self.confidence_margin = new_confidence_margin
                self.support_points_x = [line_support_point1[0], line_support_point2[0], other_support_point[0]]
                self.support_points_y = [line_support_point1[1], line_support_point2[1], other_support_point[1]]
                self.optimum_hyperplane_data = optimum_hyperplane
                self.support_hyperplanes_data = [first_support_hyp, other_support_hyp]


    def predict(self, X):
        predictions = []
        for x_i, y_i in X:
            # Find if x_i is above or below the hyperplane
            line_y_at_x_i = self.optimum_hyperplane_data[0] * x_i + self.optimum_hyperplane_data[2]
            predictions.append(-1*np.sign(line_y_at_x_i - y_i))

        return predictions
