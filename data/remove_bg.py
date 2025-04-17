#!/usr/bin/env python3
import os
import rembg
import numpy as np
from PIL import Image

def remove_background(input_image):
    # Convert the image to a NumPy array
    input_array = np.array(input_image)

    # Apply background removal using rembg
    output_array = rembg.remove(input_array)

    # Create a PIL Image from the output array
    output_image = Image.fromarray(output_array)

    return output_image

def process_all_folders(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                output_folder = os.path.dirname(output_path)
                os.makedirs(output_folder, exist_ok=True)

                print(f"Processing {input_path} -> {output_path}")

                # Open the input image
                input_image = Image.open(input_path)

                # Remove the background
                output_image = remove_background(input_image)

                # Convert to RGB mode if saving as JPEG
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    output_image = output_image.convert("RGB")

                # Save the processed image
                output_image.save(output_path)

if __name__ == "__main__":
    input_dir = "data/new_split_parts"
    output_dir = "data/parts"
    process_all_folders(input_dir, output_dir)