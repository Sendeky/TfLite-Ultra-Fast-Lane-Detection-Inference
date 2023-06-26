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