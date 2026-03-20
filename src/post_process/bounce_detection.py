import cv2
import numpy as np
import os

class Bounce_Detection:
    def __init__(self, table_corners):
        """
        Initialize Ball_Detection with table corners.
        :param table_corners: A list of four tuples representing the corners of the table [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        """
        self.table_corners = np.array(table_corners, dtype=np.int32)
        self.table_mask =  np.zeros((1080, 1920), dtype=np.uint8)  # Assuming a 1920x1080 image
        cv2.fillPoly(self.table_mask, [self.table_corners], 1)

    def point_in_table(self, point):
        """
        Check if a point is inside the table area defined by the corners.

        :param point: A tuple (x, y) representing the ball's position.
        :return: True if the point is inside the table, False otherwise.
        """
        # Create a mask for the table polygon
        return self.table_mask[point[1], point[0]] == 1

    def bounce_detection(self, ball_coordinates):
        """
        Detect bounces based on ball coordinates within the table area.

        :param ball_coordinates: List of tuples [(x1, y1), (x2, y2), ...] representing ball positions over time.
        :return: A list of indices where the ball bounces within the table area.
        """
        bounces = []

        for i in range(1, len(ball_coordinates) - 1):
            prev_y = ball_coordinates[i - 1][1]
            curr_y = ball_coordinates[i][1]
            next_y = ball_coordinates[i + 1][1]
            y_change_prev = abs(curr_y - prev_y)
            y_change_next = abs(next_y - curr_y)

            # Check if the current point is within the table area
            if self.point_in_table(ball_coordinates[i]):
                # Case 1: Bounce on one side of the table
                if prev_y < curr_y >= next_y and y_change_prev < 25 and y_change_next < 25:
                    bounces.append(i)
                # Case 2: Bounce on the other side of the table
                elif prev_y > curr_y <= next_y and y_change_prev < 25 and y_change_next < 25:
                    bounces.append(i)
                elif prev_y == curr_y == next_y:
                    bounces.append(i)

        return bounces
    

    def detect_bounce(self, ball_coordinates, window_size=5, x_smooth_threshold=5, y_smooth_threshold=10):
        """
        Detect bounces based on ball trajectory (X and Y coordinates).

        :param ball_coordinates: List of tuples [(x1, y1), (x2, y2), ...] representing ball positions over time.
        :param window_size: Number of points to analyze in the sliding window.
        :param x_smooth_threshold: Maximum allowable X deviation for smooth motion.
        :param y_smooth_threshold: Maximum allowable deviation from parabolic fit for Y motion.
        :return: List of indices where bounces are detected.
        """
        bounces = []

        if len(ball_coordinates) < window_size:
            return bounces  # Not enough points for analysis

        for i in range(window_size, len(ball_coordinates)):
            if self.point_in_table(ball_coordinates[i]):
                # Extract the sliding window
                window = ball_coordinates[i - window_size:i]
                x_coords = [pos[0] for pos in window]
                y_coords = [pos[1] for pos in window]

                # Fit a parabola to Y-coordinates: y = ax^2 + bx + c
                x_indices = np.arange(len(y_coords))
                coefficients = np.polyfit(x_indices, y_coords, 2)
                a, b, c = coefficients

                # Find the vertex of the parabola
                vertex_x = -b / (2 * a)
                vertex_y = a * vertex_x**2 + b * vertex_x + c

                # Ensure the vertex is within the current window
                if 0 <= vertex_x < len(y_coords):
                    # Check if the Y motion is smooth (close to the parabolic fit)
                    y_fit = a * x_indices**2 + b * x_indices + c
                    y_deviation = np.max(np.abs(np.array(y_coords) - y_fit))

                    # Check if the X motion is smooth
                    x_diff = np.diff(x_coords)
                    x_deviation = np.max(np.abs(x_diff - np.mean(x_diff)))

                    if y_deviation < y_smooth_threshold and x_deviation < x_smooth_threshold:
                        # Detect a bounce at the vertex
                        bounce_index = i - window_size + int(round(vertex_x))
                        bounces.append(bounce_index)

        return bounces


    def detect_bounce_with_y_outliers(self, ball_coordinates, method="zscore", threshold=2):
        """
        Detect bounces based on Y-coordinate outliers.

        Args:
            ball_coordinates (list): List of tuples [(x1, y1), (x2, y2), ...] representing ball positions over time.
            method (str): Method to detect outliers. Options: "zscore", "iqr".
            threshold (float): Threshold for detecting outliers.
                - For "zscore", it represents the number of standard deviations.
                - For "iqr", it represents the multiplier for the IQR range.

        Returns:
            list: Indices of detected bounces in the input ball_coordinates.
        """
        if not ball_coordinates:
            return []  # No data to analyze

        # Extract Y-coordinates from ball positions
        y_coords = [pos[1] for pos in ball_coordinates if pos[1]!=-1]
        y_coords = np.array(y_coords)

        outlier_indices = []

        if method == "zscore":
            # Calculate mean and standard deviation
            y_mean = np.mean(y_coords)
            y_std = np.std(y_coords)

            # Calculate z-scores
            z_scores = (y_coords - y_mean) / y_std

            # Identify indices where z-scores exceed the threshold
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

        elif method == "iqr":
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            q1 = np.percentile(y_coords, 25)
            q3 = np.percentile(y_coords, 75)

            # Calculate IQR (Interquartile Range)
            iqr = q3 - q1

            # Determine bounds for outliers
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            # Identify indices where Y-coordinates are outliers
            outlier_indices = np.where((y_coords < lower_bound) | (y_coords > upper_bound))[0]

        else:
            raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")

        return list(outlier_indices)



    


# Example usage
if __name__ == "__main__":
    table_corners = [(734, 397), (1119, 399), (1150, 581), (742, 577)]
    table_corners_game2 = [(724, 339), (1113, 335), (1214, 539), (783, 545)]
    ball_detector = Bounce_Detection(table_corners)

    ball_positions = [(800, 350), (900, 450), (950, 500), (1000, 450), (1100, 400)]  # Example coordinates
    bounces = ball_detector.bounce_detection(ball_positions)
    print("Bounce indices:", bounces)
