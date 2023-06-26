import cv2
import os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

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
			output_img = lane_detector.detect_lanes(frame, debug=False)

			cv2.imshow("Detected lanes", output_img)

		else:
			break

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()