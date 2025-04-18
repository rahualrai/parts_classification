import os
from ultralytics import YOLO

model = YOLO("yolo11n-cls-trained-synth.pt")

class_names = model.names

val_folder = "eval_data/val_no_bg"

image_paths = [
    os.path.join(val_folder, fname)
    for fname in os.listdir(val_folder)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(val_folder, fname))
]

for image_path in image_paths:
    results = model(image_path)
    res = results[0]
    if res.probs is not None:
        class_id = int(res.probs.top1)
        class_name = class_names[class_id]
    else:
        class_name = "No prediction"
    print(f"{os.path.basename(image_path)}: {class_name}")
