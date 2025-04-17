import os
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("yolo11n-cls-trained-synth.pt")

# Path to the validation folder
val_folder = "/Users/rahual/Documents/parts_classification/val"

# Get all image paths in the validation folder
image_paths = [
    os.path.join(val_folder, image_name)
    for image_name in os.listdir(val_folder)
    if os.path.isfile(os.path.join(val_folder, image_name)) and image_name.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Predict on each image and print the results
for image_path in image_paths:
    results = model(image_path)
    prediction = results[0].probs.top1 if results[0].probs is not None else "No prediction"
    print(f"{os.path.basename(image_path)}: {prediction}")