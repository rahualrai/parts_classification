import os
from ultralytics import YOLO

model = YOLO("yolo11n-cls-trained-synth.pt")

class_names = model.names

image_path = "eval_data/val_no_bg/spring_02.jpg"

if not os.path.isfile(image_path):
    print(f"no file")
else:
    results = model(image_path)
    res = results[0]

    if res.probs is not None:
        class_id = int(res.probs.top1)
        class_name = class_names[class_id]
        print(f"Prediction: {class_name} (Index: {class_id})")
    else:
        print("No prediction could be made.")