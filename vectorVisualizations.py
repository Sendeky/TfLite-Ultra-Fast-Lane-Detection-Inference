# import open3d as o3d

lane_dict = {
    "lane0": [],
    "lane1": [],
    "lane2": [],
    "lane3": []
    }
# number of lanes tells us how many lane array we need, lanes is where we receive the array with all the lanes
def input_lanes(num_lanes, lanes):
    print("num lanes: ", num_lanes)
    print("lanes: ", lanes)


    for i in range(num_lanes):
        print(f"lane{i}:", lanes[i])

        lane_dict.__setitem__(f"lane{i}", lanes[i])
        print("lane_dict: ", lane_dict)

    make_points(lane_dict=lane_dict)

def make_points(lane_dict):
    print("")
    for i, (key, value) in enumerate(lane_dict.items()):
        print("i: ", i)
        print("key: ", key)
        print("value: ", value)