import cv2
import math
import numpy as np

def find_line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the slope
    slope = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept
    intercept = y1 - slope * x1
    
    return slope, intercept

def find_parallel_line_equation(point):
    x, y = point

    # Since the line is parallel to the X-axis, the slope is 0
    slope = 0

    # The y-intercept will be the y-coordinate of the given point
    intercept = y

    return slope, intercept

def find_intersection_point(slope1, intercept1, slope2, intercept2):
    # Calculate the x-coordinate of the intersection point
    x = (intercept2 - intercept1) / (slope1 - slope2)

    # Calculate the y-coordinate of the intersection point
    y = slope1 * x + intercept1

    return round(x), round(y)

def find_required_points(coordinates):
    # Lines equation 
    slope1, intercept1 = find_line_equation(coordinates[0], coordinates[1])
    slope2, intercept2 = find_line_equation(coordinates[2], coordinates[3])

    # Sort the coordinates based on the y-coordinate
    sorted_coordinates = sorted(coordinates, key=lambda c: c[1])

    # Get the two points in the middle
    middle_points = sorted_coordinates[1:3]

    # Find the indices of the middle points in the original list
    indices = [coordinates.index(point) for point in middle_points]

    # Create an array containing only the middle points with the same indices as the input
    result = [point if i in indices else (0, 0) for i, point in enumerate(coordinates)]

    if (0 in indices):
        slope3, intercept3 = find_parallel_line_equation(middle_points[indices.index(0)])
        result[2] = find_intersection_point(slope1, intercept1, slope3, intercept3) 
    if (1 in indices):
        slope3, intercept3 = find_parallel_line_equation(middle_points[indices.index(1)])
        result[3] = find_intersection_point(slope1, intercept1, slope3, intercept3) 
    if (2 in indices):
        slope3, intercept3 = find_parallel_line_equation(middle_points[indices.index(2)])
        result[0] = find_intersection_point(slope1, intercept1, slope3, intercept3) 
    if (3 in indices):
        slope3, intercept3 = find_parallel_line_equation(middle_points[indices.index(3)])
        result[1] = find_intersection_point(slope1, intercept1, slope3, intercept3) 

    return result

def get_area(lanes):
    result = []
    for lane in lanes:
        points = find_required_points(lane)
        result.append(points)

    return result

def split_line_segment(point1, point2, n, m):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the distance between the two points
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Calculate the percentage increment between each segment
    increment = m/100

    # Calculate the increment in distance for each segment
    distance_increment = distance * increment

    # Initialize the list of points
    points = [point1]

    # Calculate and add the intermediate points
    for i in range(1, n):
        # Calculate the distance for the current segment
        segment_distance = distance_increment * i

        # Calculate the ratio of the segment distance to the total distance
        ratio = segment_distance / distance

        # Calculate the x and y coordinates of the point on the line
        x = x1 + ratio * (x2 - x1)
        y = y1 + ratio * (y2 - y1)

        # Add the point to the list
        points.append((x, y))

    # Add the end point
    points.append(point2)

    return points

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the differences in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the squared distances in x and y directions
    squared_distance = dx ** 2 + dy ** 2

    # Calculate the distance by taking the square root of the squared distance
    distance = math.sqrt(squared_distance)

    return distance

def split_number(number, n, m):
    # Calculate and add the remaining parts
    num = 1
    for i in range(1, n):
        # Calculate the size of the next part
        num += pow(1+m/100, i)

    return num


print(split_number(100,5,5))





# Draw
def draw_lanes(frame, lanes):
    print(lanes) 
    for lane in lanes:
        top_left = lane[0]
        bottom_left = lane[1]
        top_right = lane[2]
        bottom_right = lane[3]
        cv2.line(frame, top_left, top_right, (0, 255, 255), 2)
        cv2.line(frame, top_left, bottom_left, (0, 255, 255), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 255, 255), 2)
        cv2.line(frame, top_right, bottom_right, (0, 255, 255), 2)
    return frame

def draw_points(frame, points):
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)  # Draw a red dot at each point

    return frame

def draw_rectangle(frame, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw a green rectangle

    return frame

def draw_dot_segment(frame, points):
    for point in points:
        cv2.circle(frame, (round(x), round(y)), 2, (0, 0, 255), -1)

# point1 = (295, 547)
# point2 =  (436, 206)

# split_points = split_line_segment(point1, point2, 5, 50)

# print(split_points)
# frame = cv2.imread('lane.png')

# # Draw the points on the frame
# frame_with_dots = draw_points(frame, split_points)

# # Display the frame with dots
# cv2.imshow('Frame', frame_with_dots)
# cv2.waitKey(0)
# cv2.destroyAllWindows()