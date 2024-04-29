from ultralytics import YOLO
from Processing.process_frame import *

working_dir = os.getcwd()
"""
define model 
"""
model_path = os.path.join(working_dir, "saved_models", "yolov8n_float32.tflite")
model = YOLO(model_path)

camera = cv2.VideoCapture(0)  # 0 is usually the default camera

try:
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color and run detections
        detections = process_frame(frame_arr=frame, model=model)
        annotated_frame = annotate_frame(detections=detections, frame=frame)
        # Display the resulting frame
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
