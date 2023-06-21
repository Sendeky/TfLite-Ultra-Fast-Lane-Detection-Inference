import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "models/model_float32.tflite"
model_type = ModelType.TUSIMPLE

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()