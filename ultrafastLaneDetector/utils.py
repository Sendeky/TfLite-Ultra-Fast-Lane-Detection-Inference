import math
import matplotlib.pyplot as plt
import numpy as np

# averages the line points
def average_points(point1, point2):
	x = (point1[0] + point2[0]) /2
	y = (point1[1] + point2[1]) /2
	avg_point = [x, y]
	return avg_point

# shortens array, keeps every Nth element
def shorten_array(arr, step):
    ret_arr = []
    for i in range(0, len(arr), step):
        ret_arr.append(arr[i])
    return ret_arr

def calculate_turn_radius(points):
     points = np.array(points)
     x_points = points[:, 0]
     y_points = points[:, 1]
     
     # perform polynomial regression
     coefficients = np.polyfit(x_points, y_points, 3)

     # Extract the slope and intercept from the coefficients
     slope = coefficients[0]
     intercept = coefficients[1]

     # Calculate the radius (assuming distance between points is 3 meters)
     radius = 1 / abs(slope)  # Radius of the curve in meters

     # Adjust for scaling factor
     scaling_factor = 3  # Distance between points in meters
     adjusted_radius = radius / scaling_factor

     # Print the radius and adjusted radius
     print("Radius:", radius)
     print("Adjusted Radius:", adjusted_radius)

     f = np.poly1d(coefficients)

     print("f: ", f)
     print("slope: ", slope)
     print("intercept: ", intercept)

     # calculate new x's and y's
     x_new = np.linspace(x_points[0], x_points[-1], 50)
     y_new = f(x_points)

     plt.plot(x_points, y_new)
     plt.xlim([x_points[0]-1, x_points[-1] + 1 ])
     plt.show()
     
     return adjusted_radius


# calculates steering angle when givern turn/curve radius
def calculate_steering_angle(turn_radius):
     wheelbase = 0.8 # wheelbase of cart in meters
     steering_angle = math.atan(wheelbase / turn_radius)
     steering_angle_deg = math.degrees(steering_angle)
    #  print("steering_angle_deg: ", steering_angle_deg)
     return steering_angle_deg