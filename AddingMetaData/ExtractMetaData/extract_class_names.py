import os
from ultralytics import YOLO

# Define paths and load the original model
orig_model_dir = "/Users/cole/PycharmProjects/YoloV8Testing/saved_models"
orig_model_filename = "yolov8n.pt"
orig_model_path = os.path.join(orig_model_dir, orig_model_filename)
orig_model = YOLO(orig_model_path)

# Extract class names (labels)
class_names = orig_model.names

# Define paths for the new files
label_file_dir = "/Users/cole/PycharmProjects/YoloV8Testing/label_files"
label_map_dir = "/Users/cole/PycharmProjects/YoloV8Testing/label_maps"

label_file = "coco_labels.txt"
label_map_filename = "coco_label_map.txt"

labels_file_path = os.path.join(label_file_dir,label_file)
label_map_file_path = os.path.join(orig_model_dir, label_map_filename)

# Write the labels file (just the labels)
with open(labels_file_path, 'w') as file:
    for index in class_names:
        file.write(f"{class_names[index]}\n")

# Write the dictionary file (indices and labels)
with open(label_map_file_path, 'w') as file:
    file.write("{\n")
    for index in class_names:
        file.write(f"    {index}: '{class_names[index]}',\n")
    file.write("}\n")

print(f"Label file created: {labels_file_path}")
print(f"Dictionary file created: {label_map_file_path}")
