
import cv2
import math
import numpy as np

def find_line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the slope
    if x1 == x2:
        x1+=1
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
        result[2] = find_intersection_point(slope2, intercept2, slope3, intercept3) 
    if (1 in indices):
        slope3, intercept3 = find_parallel_line_equation(middle_points[indices.index(1)])
        result[3] = find_intersection_point(slope2, intercept2, slope3, intercept3) 
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

def get_BOI(areas, frame, segment, increment):
    result = []
    for area in areas:
        left_point_list, right_point_list = split_line_segment(area[0], area[3], segment, increment, area)
        # draw_points(frame, left_point_list)
        # draw_points(frame, right_point_list)
        list_BOI = []
        for i in range(len(left_point_list)-1):
            list_BOI.append([left_point_list[i],right_point_list[i+1]])
            draw_rectangle(frame, left_point_list[i], right_point_list[i+1])
        

        result.append(list_BOI)

    return frame, result


def split_line_segment(point1, point2, n, m, coordinates):
    slope1, intercept1 = find_line_equation(coordinates[0], coordinates[1])
    slope2, intercept2 = find_line_equation(coordinates[2], coordinates[3])

    x1, y1 = point1
    x2, y2 = point2

    distance = y2-y1
    list_num = split_number(distance, n, m)
    # print("Distance between y2, y1: " + str(distance));
    # print("split to " + str(n) + " part with increment " + str(m) + "%")
    # print(list_num)
    point_list = [(x1, y1)]
    distance_temp = y1

    for i in range(len(list_num)):
        distance_temp += list_num[i]
        point_list.append((x1,distance_temp))

    right_point_list = []
    left_point_list = []
    for i in range(len(point_list)):
        slope3, intercept3 = find_parallel_line_equation(point_list[i])
        left_point = find_intersection_point(slope1, intercept1, slope3, intercept3)
        right_point = find_intersection_point(slope2, intercept2, slope3, intercept3)
        left_point_list.append(left_point)
        right_point_list.append(right_point)
    
    return left_point_list, right_point_list

    # return point

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
    powe = 1
    for i in range(1, n):
        # Calculate the size of the next part
        powe += pow(1+m/100, i)

    first_num = round(number/powe)
    list_num = [first_num]

    incre = first_num
    for i in range(1, n):
        incre = (incre*(1+m/100))
        list_num.append(round(incre))

    return list_num

# Draw
def draw_lanes(frame, lanes):
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