import os
from ultralytics import YOLO

import os
import rembg
import numpy as np
from PIL import Image

def remove_background(input_image):
    input_array = np.array(input_image)
    output_array = rembg.remove(input_array)
    output_image = Image.fromarray(output_array)
    return output_image

model = YOLO("yolo11n-cls-trained-synth.pt")

class_names = model.names

image_name = "shaft_01.jpg"
image_path = os.path.join("eval_data/val", image_name)
image_folder_path_no_bg = os.path.join("eval_data/val_no_bg", image_name)

if not os.path.isfile(image_path):
    print(f"no file")
else:
    # remove background
    input_image = Image.open(image_path)
    output_image = remove_background(input_image)
    output_image = output_image.convert("RGB")
    # Save the processed image
    output_image.save(image_folder_path_no_bg)

    results = model(image_folder_path_no_bg)
    res = results[0]

    if res.probs is not None:
        class_id = int(res.probs.top1)
        class_name = class_names[class_id]
        print(f"Prediction: {class_name} (Index: {class_id})")
    else:
        print("No prediction could be made.")

