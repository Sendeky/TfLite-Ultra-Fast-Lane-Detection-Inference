import cv2
import os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from vectorVisualizations import input_lanes

model_path = "models/model_float32.tflite"
model_type = ModelType.TUSIMPLE


# get current directory
print("Current Directory: ", os.getcwd())
# Initialize video
cap = cv2.VideoCapture("test.mp4")

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

# user input to continue
input("Press Enter to continue...")

debug = input("y/n Debug")
if debug == "y":
	debug = True
else: 
	debug = False

if __name__ == "__main__":
	while cap.isOpened():
		try:
			# Read frame from the video
			ret, frame = cap.read()
		except:
			continue

		if ret:	

			# Detect the lanes
			# use debug if you want to see pyplot of smoothed center points
			output_img, lanes_array = lane_detector.detect_lanes(frame, debug=debug)

			cv2.imshow("Detected lanes", output_img)

			# give lanes_array to vectorVisualizations.py for drawing in visualizations space
			# print("len lanes:", len(lanes_array))
			# print("lanes array:", lanes_array)
			input_lanes(len(lanes_array), lanes_array)

		else:
			break

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()