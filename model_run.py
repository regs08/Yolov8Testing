from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
# Load the model
working_dir = os.getcwd()

model_path = os.path.join(working_dir, "saved_models", "yolov8n_float32.tflite")
print(working_dir)
print(model_path)
LABEL_MAP_PATH = os.path.join(working_dir, "label_maps", "coco_labels.txt")
model = YOLO(model_path)
"""
trying to manually set names
"""
import ast

def load_labels(filename):
    with open(filename, 'r') as file:
        data = file.read()
        labels_dict = ast.literal_eval(data)
    return labels_dict

# Load the labels
label_map = load_labels(LABEL_MAP_PATH)

camera = cv2.VideoCapture(0)  # 0 is usually the default camera

try:
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Run detection
        results = model.predict(source=pil_image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(pil_image)
        for box, mask, conf, class_id, tracker_id, class_name in detections:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)
            draw.text((box[0], box[1]), label_map[class_id], fill="red")

        # Convert PIL image back to array
        rgb_image = np.array(pil_image)

        # Convert RGB back to BGR for displaying
        final_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Frame', final_frame)

        # Press 'q' to break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
