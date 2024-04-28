import os
from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils

# Path to the TFLite model and label file
TFLITE_MODEL_PATH = "/Users/cole/PycharmProjects/YoloV8Testing/saved_models/yolov8n_float32.tflite"
LABEL_FILE = "/label_files/coco_labels.txt"
OUTPUT_DIR = "/Users/cole/PycharmProjects/YoloV8Testing/saved_models"
TFLITE_MODEL_WITH_METADATA = "yolov8n_with_metadata.tflite"

out_path = os.path.join(OUTPUT_DIR, TFLITE_MODEL_WITH_METADATA)
# Create the metadata writer
writer = image_classifier.MetadataWriter.create_for_inference(
    writer_utils.load_file(TFLITE_MODEL_PATH), input_norm_mean=[0], input_norm_std=[255], label_file_paths=[LABEL_FILE])

# Add metadata to the model
writer_utils.save_file(writer.populate(), out_path)
