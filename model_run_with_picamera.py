from ultralytics import YOLO
from Processing.process_frame import *

working_dir = os.getcwd()
"""
define model 
"""
model_path = os.path.join(working_dir, "saved_models", "yolov8n_float32.tflite")
model = YOLO(model_path)

picam2 = Picamera2()
picam2.preview_configuration.main.size=(1920,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.start()

try:
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()
        if not frame:
            print("frame not found!")
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
    picam2.stop()
    cv2.destroyAllWindows()
