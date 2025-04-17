import rembg
import numpy as np
from PIL import Image

# Load the input image (replace with the actual path)
try:
    input_image = Image.open('data/new_split_parts/test/connector/image_01.jpg')
except FileNotFoundError:
    print("Error: Input image file not found. Please check the file path.")
    exit()

input_array = np.array(input_image)

# Apply background removal using rembg
try:
    output_array = rembg.remove(input_array)
except Exception as e:
    print(f"Error during background removal: {e}")
    exit()

# Create a PIL Image from the output array
output_image = Image.fromarray(output_array)

# Convert to RGB if saving as JPEG (optional, if you need to remove transparency)
# output_image = output_image.convert('RGB')

# Save the output image as PNG (to preserve transparency)
output_image.save('data/new_split_parts/test/connector/image_01_bg.jpg')