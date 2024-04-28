import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/Users/cole/PycharmProjects/YoloV8Testing/saved_models/yolov8n_with_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image = Image.open('/Users/cole/PycharmProjects/YoloV8Testing/saved_models/test.jpeg').resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
image = np.array(image, dtype=np.float32)
image = image / 255.0  # Normalize the image as per metadata
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Get the results
output_data = interpreter.get_tensor(output_details[0]['index'])

# Load labels
labels = {}
with open("label_maps/coco_labels.txt", "r") as f:
    for index, line in enumerate(f.readlines()):
        label_name = line.strip()  # Remove any leading/trailing whitespace
        labels[index] = label_name

# Process and print the results
predicted_label_index = np.argmax(output_data)
print("Predicted Label:", labels[predicted_label_index])
