import cv2
from PIL import Image
import supervision as sv
import os
from Processing.utils.load_labels import load_labels
working_dir = os.getcwd()
LABEL_MAP_PATH = os.path.join(working_dir, "label_maps", "coco_labels.txt")

# note must be box annotator, only one that has labels as an arg
bounding_box_annotator = sv.BoxAnnotator()
label_map = load_labels(LABEL_MAP_PATH)

def process_frame(frame_arr, model):
    #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_arr)

    # Run detection
    results = model.predict(source=pil_image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections

def annotate_frame(detections, frame):

    #extract labels from detections class ids
    labels = [label_map[id] for id in detections.class_id]
    #create a string and format the label with the conf
    labels_with_conf = [f"{l} {c:.2f}" for l, c in zip(labels, detections.confidence)]

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels_with_conf)
    return annotated_frame