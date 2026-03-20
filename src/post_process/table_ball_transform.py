import cv2
import sys
import os
import numpy as np
sys.path.append('../')

import matplotlib.pyplot as plt


class Table_ball_transform:
    def __init__(self, output_folder, img_corners, table_corners = [(0, 0), (1525, 0), (1525, 2740), (0, 2740)]):
        self.output_folder = output_folder
        self.selected_corners = self.order_corners(img_corners)
        self.corners_image_path = os.path.join(output_folder, "corners_images.jpg")
        self.table_corners = table_corners

    def map_ball_to_table(self, ball_positions):
        """
        Map a ball's position from image frame to table coordinates.
        
        Args:
            ball_positions (tuple): a list (x, y) position of the ball in the image frame.

        Returns:
            list: (x, y) position of the ball in table coordinates.
        """
        # Convert corners to numpy arrays
        image_corners_np = np.array(self.selected_corners, dtype="float32")
        table_corners_np = np.array(self.table_corners, dtype="float32")

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(image_corners_np, table_corners_np)

        # Transform the ball position
        transformed_positions = []
        for ball_pos in ball_positions:
            ball_position_np = np.array([[ball_pos]], dtype="float32")  # Shape (1, 1, 2)
            transformed_position = cv2.perspectiveTransform(ball_position_np, M)
            transformed_positions.append(transformed_position[0, 0])  # Store as [x, y]

        return transformed_positions
        
    def draw_ball_positions(self, ball_positions, vertical_split=(0.25, 0.5, 0.25), horizontal_split=(0.25, 0.5, 0.75)):
        """
        Draws the ball positions on the table view and divides the table into nine regions:
        - Top Upper Left, Top Upper Middle, Top Upper Right
        - Top Lower Left, Top Lower Middle, Top Lower Right
        - Bottom Upper Left, Bottom Upper Middle, Bottom Upper Right
        - Bottom Lower Left, Bottom Lower Middle, Bottom Lower Right

        Args:
            ball_positions (list): List of ball positions in the original image.
            vertical_split (tuple): Fractions for (left, middle, right) sections.
                                    - Default (0.25, 0.5, 0.25) → 25% left, 50% middle, 25% right
            horizontal_split (tuple): Fractions for horizontal division.
                                    - Default (0.25, 0.5, 0.75) → 25% upper, 25% lower, 50% bottom
        """
        # Create a blank image for the table view
        table_view = np.zeros((2740, 1525, 3), dtype=np.uint8)

        # Transform ball positions to table coordinates
        transformed_table_pos = self.map_ball_to_table(ball_positions)

        # Draw the table outline
        cv2.polylines(table_view, [np.array(self.table_corners, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

        # Get table dimensions
        table_height, table_width = table_view.shape[:2]

        # **Horizontal 4-Way Split**
        top_upper_boundary = int(table_height * horizontal_split[0])  # 25% height (Top Upper)
        top_lower_boundary = int(table_height * horizontal_split[1])  # 50% height (Net Line)
        bottom_upper_boundary = int(table_height * horizontal_split[2])  # 75% height (Bottom Upper)
        
        # **Vertical 3-Way Split**
        left_boundary = int(table_width * vertical_split[0])  # Left limit
        right_boundary = int(table_width * (vertical_split[0] + vertical_split[1]))  # Right limit

        # Initialize counters for 9 regions
        counts = {
            "top_upper_left": 0, "top_upper_middle": 0, "top_upper_right": 0,
            "top_lower_left": 0, "top_lower_middle": 0, "top_lower_right": 0,
            "bottom_upper_left": 0, "bottom_upper_middle": 0, "bottom_upper_right": 0,
            "bottom_lower_left": 0, "bottom_lower_middle": 0, "bottom_lower_right": 0
        }

        # Draw ball positions and count them in each region
        for table_pos in transformed_table_pos:
            x, y = int(table_pos[0]), int(table_pos[1])  # Extract coordinates
            cv2.circle(table_view, (x, y), 10, (0, 255, 0), -1)  # Draw ball
            cv2.putText(table_view, "Ball", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # **Determine which of the 9 regions the ball is in**
            if y < top_upper_boundary:  # **Top Upper Section**
                if x < left_boundary:
                    counts["top_upper_left"] += 1
                elif x < right_boundary:
                    counts["top_upper_middle"] += 1
                else:
                    counts["top_upper_right"] += 1
            elif y < top_lower_boundary:  # **Top Lower Section**
                if x < left_boundary:
                    counts["top_lower_left"] += 1
                elif x < right_boundary:
                    counts["top_lower_middle"] += 1
                else:
                    counts["top_lower_right"] += 1
            elif y < bottom_upper_boundary:  # **Bottom Upper Section**
                if x < left_boundary:
                    counts["bottom_upper_left"] += 1
                elif x < right_boundary:
                    counts["bottom_upper_middle"] += 1
                else:
                    counts["bottom_upper_right"] += 1
            else:  # **Bottom Lower Section**
                if x < left_boundary:
                    counts["bottom_lower_left"] += 1
                elif x < right_boundary:
                    counts["bottom_lower_middle"] += 1
                else:
                    counts["bottom_lower_right"] += 1

        # **Draw Horizontal Net (Middle Line) and Extra Dividers**
        cv2.line(table_view, (0, top_upper_boundary), (table_width, top_upper_boundary), color=(255, 255, 255), thickness=2)
        cv2.line(table_view, (0, top_lower_boundary), (table_width, top_lower_boundary), color=(255, 255, 255), thickness=2)
        cv2.line(table_view, (0, bottom_upper_boundary), (table_width, bottom_upper_boundary), color=(255, 255, 255), thickness=2)

        # **Draw Vertical Split Lines**
        cv2.line(table_view, (left_boundary, 0), (left_boundary, table_height), color=(255, 255, 255), thickness=2)  # Left boundary
        cv2.line(table_view, (right_boundary, 0), (right_boundary, table_height), color=(255, 255, 255), thickness=2)  # Right boundary

        # Save the result
        cv2.imwrite("table_with_ball_9_regions.jpg", table_view)

        # Print ball count information
        print(f"Ball positions and table split into 9 regions (4 Horizontal × 3 Vertical).")
        for key, value in counts.items():
            print(f"⚪ {key.replace('_', ' ').title()}: {value}")





    def order_corners(self, corners):
        """
        Order the corners in the correct sequence: top-left, top-right, bottom-right, bottom-left.
        Args:
            corners (list of tuples): List of 4 (x, y) coordinates.
        Returns:
            list of tuples: Ordered corners.
        """
        corners = np.array(corners)

        # Sum of x and y (top-left will have the smallest sum, bottom-right the largest)
        s = corners.sum(axis=1)
        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]

        # Difference of x and y (top-right will have the smallest difference, bottom-left the largest)
        diff = np.diff(corners, axis=1)
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]

        return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]

    def get_user_selected_corners_with_mouse(self, image):
        """Allow the user to click on four corners of the table."""
        global points, temp_image
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                points.append((x, y))
                print(f"Point selected: {x}, {y}")
                cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Corners", temp_image)

        temp_image = image.copy()
        cv2.imshow("Select Corners", temp_image)
        cv2.setMouseCallback("Select Corners", click_event)

        print("Click on four corners of the table.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(points) != 4:
            print("Error: You must select exactly 4 points.")
            return None
        
        return points


if __name__ == "__main__":
    print("ok")