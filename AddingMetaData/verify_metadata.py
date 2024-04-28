from tflite_support import metadata as _metadata

model_with_metadata_path = "/Users/cole/PycharmProjects/YoloV8Testing/saved_models/yolov8n_with_metadata.tflite"
displayer = _metadata.MetadataDisplayer.with_model_file(model_with_metadata_path)
print("Metadata populated:")
print(displayer.get_metadata_json())
