"""
---------------HOW THIS WORKS---------------
"class ModelConfig" has init parameters for models, ie. Culane and Tusimple
"class UltrafastLaneDetector" init func makes variables for tracking fps, time, etc. and calls "initialize_model()"
"initialize_model()" creates TFLite interpretor + gets model input & output deteails
^^^^^^
All that happens when the class gets initialized (in this case, in videoLaneDetection.py)


All this happens when "detect_lanes()" is called from instance of UltraFastLaneDetector
˅˅˅˅˅˅
"def detect_lanes()" gets frame and prepares it by sending it to prepare_input()
	"prepare_input()"  converts RGB2BGR, gets height, width, channels of image, and scales/normalized the pixel values
"input_tensor" from "prepare_input()" gets passed to "inference()"
	"inference()" sets tensor depending on modeltype, performs infeerence, gets output of inference model, and reshapes output to 3D array (anchors, lanes, points)
"process_output" takes in output ^^ and returns lane points and lanes detected
	parses "output" of model, normalizes it, reshapes to 3D array, gets location of lane points
	checks for points and iterates, appending lane points, returns lane points matrix, lane presence (true/false, out of 4)
"draw_lanes" then takes image, lane points matrix, lane presence, config, draw_points bool
	resizes image to model input size, adds mask for inner lanes, draws points and lines if draw_points bool is true
"""









import time
import cv2
import scipy.special
from enum import Enum
import numpy as np

#utils
# from utils import average_points
from .utils import average_points

#Checks if tflite runtime is installed
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

# Colors for lanes/dots on visualization
lane_colors = [(0,0,255),(255,0,0),(255,0,0),(0,255,255)]

# Row anchors for the two models
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

# Model configuration
class ModelConfig():

	def __init__(self, model_type):

		# Different configs depending on model
		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	# Tusimple model configuration
	# 1280 x 720 image, 100 griding number, 56 cls_num_per_lane
	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	# Culane model configuration
	# 1640 x 590 image, 200 griding number, 18 cls_num_per_lane
	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE):
		
		# Variables for tracking fps, time, and frame count
		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path)

	def initialize_model(self, model_path):

		# creates interpreter that uses TFLite model
		self.interpreter = Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()

		# Get model info
		self.getModel_input_details()
		self.getModel_output_details()
		
	def detect_lanes(self, image, draw_points=True):
		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 288 x 800 pixels
		img_input = cv2.resize(img, (self.input_width,self.input_height)).astype(np.float32)
		
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]

		img_input = ((img_input/ 255.0 - mean) / std).astype(np.float32)
		img_input = img_input[np.newaxis,:,:,:]

		return img_input

	def getModel_input_details(self):
		self.input_details = self.interpreter.get_input_details()
		input_shape = self.input_details[0]['shape']
		self.input_height = input_shape[1]
		self.input_width = input_shape[2]
		self.channels = input_shape[3]

	def getModel_output_details(self):
		self.output_details = self.interpreter.get_output_details()
		output_shape = self.output_details[0]['shape']
		self.num_anchors = output_shape[1]
		self.num_lanes = output_shape[2]	
		self.num_points = output_shape[3]

	def inference(self, input_tensor):
		# Different models have different input and output tensor parameters
		# In this case Culane and Tusimple have slighly varying parameters. More info in this paper: https://arxiv.org/abs/2004.11757
		# Peform inference
		self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)		# sets tensor depending on model and input tensor (prepared image)
		self.interpreter.invoke()
		output = self.interpreter.get_tensor(self.output_details[0]['index'])			# gets tensor from model

		output = output.reshape(self.num_anchors, self.num_lanes, self.num_points)		# reshapes output tensor to 3D array (anchors, lanes, points)
		return output


	@staticmethod
	def process_output(output, cfg):		

		# Parse the output of the model to get the lane information
		processed_output = output[:, ::-1, :]

		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)		# applies softmax fucntion to outpu tensor (normalizes it)
		idx = np.arange(cfg.griding_num) + 1									# creates array of numbers from 1 to 100/200 (depending on model, tusimple/culane)
		idx = idx.reshape(-1, 1, 1)												# reshapes array to 3D array (100,200, 1, 1) (tusimple/culane)
		loc = np.sum(prob * idx, axis=0)										# multiplies prob and idx arrays and sums them up; gets location of lane points
		processed_output = np.argmax(processed_output, axis=0)					# gets index of maximum value in output tensor 
		loc[processed_output == cfg.griding_num] = 0							# sets location of lane points to 0 if index of maximum value in output tensor is 100/200 (tusimple/culane)
		processed_output = loc													# sets processed output to location of lane points

		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lane_points_mat = []				# matrix of lane points
		lanes_detected = []					# list of lanes detected (rightmost, right, left, leftmost)

		max_lanes = processed_output.shape[1]
		# print("max lanes: ", max_lanes)
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lane_points_mat.append(lane_points)
			# print("Lane points mat: ", lane_points_mat)
			# print("Lane points mat type: ", type(lane_points_mat))
		return np.array(lane_points_mat), np.array(lanes_detected)
	


	@staticmethod
	def draw_lanes(input_img, lane_points_mat, lanes_detected, cfg, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			
			lane_segment_img = visualization_img.copy()
			
			cv2.fillPoly(lane_segment_img, pts = [np.vstack((lane_points_mat[1],np.flipud(lane_points_mat[2])))], color =(255,128,0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		if(draw_points):
			for lane_num,lane_points in enumerate(lane_points_mat):
				for lane_point in lane_points:

					if lane_num == 0 or lane_num == 3:
						# draw circles for outer lanes
						cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
					
					if lane_num == 1 or lane_num == 2:
						# draw continous lines for inner lanes
						print("lane points: ", lane_points)
						print("lane points len: ", len(lane_points))

						# draw dashed lines for inner lanes
						for i in range(0, len(lane_points) - 1, 1):
							print("i range: ", i)
							print(f"{i} lane_points: ", lane_points[i])
							print(f"{i + 1} lane_points: ", lane_points[i + 1])
							print("avg points: ")
							point1 = lane_points[i]
							point2 = lane_points[i + 1]
							cv2.line(img=visualization_img, pt1=point1, pt2=point2, color=lane_colors[lane_num], thickness=3)
						
						#print(f"min: 0")
						#print(f"max: {len(lane_points)}")

						# draws straight line from bottommost to topmost
						# minimum and maximum end point (ie. topmost and bottommost)
						# point1 = lane_points[0]
						# point2 = lane_points[(len(lane_points) - 1)]	# -1 because len is 1...n, not 0...n
						# cv2.line(img=visualization_img, pt1=point1, pt2=point2, color=lane_colors[lane_num], thickness=3)

					# lane_points = np.array(lane_points)
					# cv2.drawContours(image=visualization_img, contours=[lane_points], contourIdx=-1, color=lane_colors[lane_num], thickness=3)
			# print("lane_points_mat: ", lane_points_mat)
			converted_lane_points_mat = np.array(lane_points_mat).tolist()
			print("converted lane_points_mat: ", converted_lane_points_mat)

			# checks for 2 center lanes in lane_points matrix
			if converted_lane_points_mat[1] and [2]:
				print("Center lanes exist")
				print("right lane: ", converted_lane_points_mat[1])
				print("left lane: ", converted_lane_points_mat[2])

				length = min(len(converted_lane_points_mat[1]), len(converted_lane_points_mat[2]))
				print("min length lanes: ", length)


				# gets center points between left and right
				for i in range(0, length - 1, 1):
					# gets center point of lane 1 and lane 2 point "i"
					center_point1 = average_points(converted_lane_points_mat[1][i], converted_lane_points_mat[2][i])
					center_point2 = average_points(converted_lane_points_mat[1][i + 1], converted_lane_points_mat[2][i + 1])

					# points need to be integers before given to cv2.line
					center_x1 = int(center_point1[0])
					center_y1 = int(center_point1[1])
					center_x2 = int(center_point2[0])
					center_y2 = int(center_point2[1])

					point1 = [center_x1, center_y1]
					point2 = [center_x2, center_y2]
					cv2.line(img=visualization_img, pt1=point1, pt2=point2, color=lane_colors[lane_num], thickness=8)
					# center_points.append([center_x, center_y])
				
				# print("center points: ", center_points)
				# center_points = np.array(center_points)
				# cv2.drawContours(visualization_img, [center_points], -1, lane_colors[lane_num], 3)
				# cv2.drawContours(image=visualization_img, contours=[lane_points], contourIdx=-1, color=lane_colors[lane_num], thickness=3)


		return visualization_img


	







